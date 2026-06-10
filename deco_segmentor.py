import random
import os
import torch
from omegaconf import OmegaConf
from src.models.autoencoder.base import fp2uint8
from src.diffusion.base.guidance import simple_guidance_fn
from wrapper_adam_sampling import WrapperAdamLMSampler
from src.diffusion.flow_matching.scheduling import LinearScheduler
from PIL import Image
import tempfile
from huggingface_hub import snapshot_download
from PIL import Image
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config as local_config


def instantiate_class(config):
    kwargs = config.get("init_args", {})
    class_module, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(**kwargs)

def load_model(weight_dict, denoiser):
    prefix = "ema_denoiser."
    for k, v in denoiser.state_dict().items():
        try:
            v.copy_(weight_dict["state_dict"][prefix + k])
        except:
            print(f"Failed to copy {prefix + k} to denoiser weight")
    return denoiser

class Pipeline:
    def __init__(self, vae, denoiser, conditioner, resolution, device, num_steps, guidance, timeshift, order, save_maps=False):
        self.vae = vae.to(device)
        self.denoiser = denoiser.to(device)
        self.conditioner = conditioner.to(device)
        self.conditioner.compile()
        self.resolution = resolution
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="traj_gifs_")
        # self.denoiser.compile()

        self.diffusion_sampler = WrapperAdamLMSampler(
            order=order,
            scheduler=LinearScheduler(),
            guidance_fn=simple_guidance_fn,
            num_steps=num_steps,
            guidance=guidance,
            timeshift=timeshift,
            save_maps=save_maps
        )

    def __del__(self):
        self.tmp_dir.cleanup()

    @torch.no_grad()
    def __call__(self, x, y, neg_prompt, num_images, image_height, image_width):
        image_height = image_height // 32 * 32 # TODO: related to attention maps resolution ? could get finer maps?
        image_width = image_width // 32 * 32
        self.denoiser.decoder_patch_scaling_h = image_height / 512
        self.denoiser.decoder_patch_scaling_w = image_width / 512

        xT = torch.stack([x] * num_images, dim=0)
        xT = (xT.float() / 127.5) - 1
        xT = xT.to(local_config.device)
        with torch.no_grad():
            condition, uncondition = self.conditioner([y,]*num_images, {"negative_prompt": neg_prompt})
            attention_maps = self.diffusion_sampler(self.denoiser, xT, condition, uncondition, return_x_trajs=True)
        return attention_maps[1]
    


class DeCoSegmentor(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        ckpt_path = 'deco.ckpt'
        if not os.path.exists(ckpt_path):
            snapshot_download(repo_id='MCG-NJU/PixNerd-XXL-P16-T2I', local_dir=ckpt_path)
            ckpt_path = os.path.join(ckpt_path, "model.ckpt")
        else:
            ckpt_path = ckpt_path

        config_path = "./configs_t2i/sft_res512.yaml"
        config = OmegaConf.load(config_path)
        vae_config = config.model.vae
        denoiser_config = config.model.denoiser
        conditioner_config = config.model.conditioner

        vae = instantiate_class(vae_config)
        denoiser = instantiate_class(denoiser_config)
        conditioner = instantiate_class(conditioner_config)


        ckpt = torch.load(ckpt_path, map_location="cpu")
        denoiser = load_model(ckpt, denoiser)
        denoiser = denoiser.to(local_config.device)
        vae = vae.to(local_config.device)
        denoiser.eval()

        #TODO delete
        self.idx = 0
        import json
        with open('catseg_configs/ade150.json') as f:
            self.categs = json.load(f)
        self.categs_count = len(self.categs)

        # TODO None is instead of resolution
        # TODO: nums steps hardcoded
        self.pipeline = Pipeline(vae, denoiser, conditioner, None, local_config.device, 100, local_config.guidance,
                                 local_config.timeshift, local_config.order, local_config.save_maps)

    @staticmethod
    def get_gt_shape(x):
        gt_file_path = x[0]['file_name'].replace("images", "annotations").replace(".jpg", ".png")
        mask = Image.open(gt_file_path)
        mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0)
        return mask_tensor.shape
    
    def get_gt_labels(self, x):
        gt_file_path = x[0]['file_name'].replace("images", "annotations").replace(".jpg", ".png")
        mask = Image.open(gt_file_path)
        mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0)
        return [(i, self.categs[i]) for i in torch.unique(mask_tensor)]

    def call_with_defaults(self, x, prompt):
        return self.pipeline(
            x,
            prompt,
            local_config.neg_label,
            local_config.num_images,
            local_config.image_height,
            local_config.image_width
        )
    
    @staticmethod
    def resize_maps(attention_maps, new_shape):
        return F.interpolate(attention_maps, new_shape, mode="bilinear", align_corners=False)

    def forward(self, x):
        image_tensor = x[0]["image"]
        gt_idxs_and_labels = self.get_gt_labels(x)
        gt_shape = DeCoSegmentor.get_gt_shape(x)
        prompt_format = "a picture of a {target}"
        prediction = torch.zeros((self.categs_count+1, gt_shape[-2], gt_shape[-1])).to(local_config.device)
        # init background score TODO: do not hardcode treshold
        prediction[0] += 0.0

        # TODO: switch order of interpolate and argmax?
        for label_idx, label in gt_idxs_and_labels:
            prompt = prompt_format.format(target=label)
            attention_maps = self.call_with_defaults(image_tensor, prompt)
            # select slice corresponding to positive prompt
            attention_maps = attention_maps[0].unsqueeze(0).unsqueeze(0)
            prediction[label_idx+1] += DeCoSegmentor.resize_maps(attention_maps, gt_shape[1:])
        
        class_prediction = np.argmax(prediction.detach().cpu().numpy(), axis=0).astype(np.uint8)

        img_array = (image_tensor.cpu().numpy()).astype(np.uint8).transpose(1, 2, 0)
        Image.fromarray(img_array).save(f"z_output/{self.idx}_input_image.png")
        
        # Save class_prediction as heatmap
        plt.figure()
        plt.imshow(class_prediction, cmap='viridis')
        plt.colorbar(label='Class Index')
        plt.title('Class Prediction')
        plt.axis('off')
        plt.savefig(f"z_output/{self.idx}_class_prediction_heatmap.png")
        plt.close()

        with open('hello.txt', "w") as f:
            for ai in range(class_prediction.shape[0]):
                for bi in range(class_prediction.shape[1]):
                    f.write(f"{class_prediction[ai, bi]} ")
                f.write("\n")

            f.write(f"\n\nPREDICTS\n\n0: {prediction[:, 10, 30]}\n1: {prediction[:, 388, 399]}")
        self.idx += 1
        if self.idx == 1:
            exit(0)

        return [{"sem_seg": class_prediction}]