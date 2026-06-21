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


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def visualize_prediction(image, predictions, class_names, file_id, alpha=0.6):
    """
    image: torch.Tensor
        Shape (3,H,W) or (1,3,H,W)

    predictions: np.ndarray
        Shape (num_classes,H,W)
        logits or probabilities
    """

    # --------------------------
    # Convert image
    # --------------------------

    if image.ndim == 4:
        image = image.squeeze(0)

    image = image.detach().cpu().permute(1, 2, 0).numpy()

    # normalize for display
    image = image.astype(np.float32)

    if image.max() > 1:
        image /= 255.0

    image = np.clip(image, 0, 1)

    # --------------------------
    # Predicted class map
    # --------------------------

    pred_mask = np.argmax(predictions, axis=0)

    num_classes = predictions.shape[0]

    # ADE20K-style palette
    colors = np.random.RandomState(42).rand(num_classes, 3)

    cmap = ListedColormap(colors)

    # --------------------------
    # Overlay
    # --------------------------

    colored_mask = cmap(pred_mask)[..., :3]

    overlay = (
        (1 - alpha) * image
        + alpha * colored_mask
    )

    overlay = np.clip(overlay, 0, 1)

    # --------------------------
    # Plot
    # --------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(image)
    axes[0].set_title("Image")

    axes[1].imshow(
        pred_mask,
        cmap=cmap,
        interpolation="nearest"
    )
    axes[1].set_title("Prediction")

    # Add legend for class IDs and their colors
    unique_classes = np.unique(pred_mask)
    legend_elements = [
        Patch(facecolor=colors[i], label=f"{class_names[i]}")
        for i in unique_classes
    ]
    axes[1].legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize="small"
    )

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    # TODO
    plt.savefig(f"z_output/{file_id}.png")


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
    def __call__(self, x, y, neg_prompt, num_images, image_height, image_width, extra_dict):
        image_height = image_height // 32 * 32 # TODO: related to attention maps resolution ? could get finer maps?
        image_width = image_width // 32 * 32
        self.denoiser.decoder_patch_scaling_h = image_height / 512
        self.denoiser.decoder_patch_scaling_w = image_width / 512

        xT = torch.stack([x] * num_images, dim=0)
        xT = (xT.float() / 127.5) - 1
        xT = xT.to(local_config.device)
        condition, uncondition = self.conditioner([y,]*num_images, {"negative_prompt": neg_prompt})
        attention_maps = self.diffusion_sampler(self.denoiser, xT, condition, uncondition, extra_dict=extra_dict)
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
            self.categs = ["background"] + json.load(f)
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
        return [(i, self.categs[i]) for i in torch.unique(mask_tensor).to(torch.uint8) if i > 0]

    def call_with_defaults(self, x, prompt, extra_dict):
        return self.pipeline(
            x,
            prompt,
            local_config.neg_label,
            local_config.num_images,
            local_config.image_height,
            local_config.image_width,
            extra_dict
        )
    
    @staticmethod
    def resize_maps(attention_maps, new_shape):
        return F.interpolate(attention_maps, new_shape, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def forward_no_grad(self, x):
        image_tensor = x[0]["image"]
        gt_idxs_and_labels = self.get_gt_labels(x)
        gt_shape = DeCoSegmentor.get_gt_shape(x)
        prompt_format = "a picture of a {target} whitin a complex scene"
        prediction = torch.zeros((self.categs_count+1, gt_shape[-2], gt_shape[-1])).to(local_config.device)
        # init background score TODO: do not hardcode treshold
        # 0.00002 too little
        prediction[0] += local_config.background_threshold

        # TODO: switch order of interpolate and argmax?
        for label_idx, label in gt_idxs_and_labels:
            prompt = prompt_format.format(target=label)
            print(self.idx, "LABEL", label)
            extra_dict = {
                "prompt": prompt,
                "eval_mode": local_config.eval,
                "img_id": self.idx
            }
            
            attention_maps = self.call_with_defaults(image_tensor, prompt, extra_dict)
            # select slice corresponding to positive prompt
            attention_maps = attention_maps[0].unsqueeze(0).unsqueeze(0)
            resized_map = DeCoSegmentor.resize_maps(attention_maps, gt_shape[1:]).squeeze(0).squeeze(0)
            prediction[label_idx.item()] += resized_map

            # DeCoSegmentor.save_attn_map(resized_map, label)

        # class_prediction = np.argmax(prediction.detach().cpu().numpy(), axis=0).astype(np.uint8)

        # img_array = (image_tensor.cpu().numpy()).astype(np.uint8).transpose(1, 2, 0)
        # Image.fromarray(img_array).save(f"z_output/{self.idx}_input_image.png")
        
        # # Save class_prediction as heatmap
        # plt.figure()
        # plt.imshow(class_prediction, cmap='viridis')
        # plt.colorbar(label='Class Index')
        # plt.title('Class Prediction')
        # plt.axis('off')
        # plt.savefig(f"z_output/{self.idx}_class_prediction_heatmap.png")
        # plt.close()

        visualize_prediction(self.resize_maps(image_tensor.unsqueeze(0), gt_shape[1:]), prediction.detach().cpu(), 
                             self.categs,
                             x[0]['file_name'].split("/")[-1].replace(".jpg", ""))
        self.idx += 1
        return [{"sem_seg": prediction[1:, :, :]}]
    
    def forward(self, x):
        return self.forward_no_grad(x)