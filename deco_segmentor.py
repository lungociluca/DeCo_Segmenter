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
import scipy.io as sio

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
    plt.close()


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
    def __init__(self, vae, denoiser, conditioner, resolution, device, num_steps, guidance, timeshift, order, save_maps=False, prompt_format='', labels=[]):
        # self.vae = vae.to(device)
        self.denoiser = denoiser.to(device)
        self.conditioner = conditioner.to(device)
        self.conditioner.compile()
        self.resolution = resolution
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="traj_gifs_")
        # self.denoiser.compile()

        self.prompt_embeddings, self.token_lenghts = self.compute_prompt_embs(prompt_format, labels)

        self.diffusion_sampler = WrapperAdamLMSampler(
            order=order,
            scheduler=LinearScheduler(),
            guidance_fn=simple_guidance_fn,
            num_steps=num_steps,
            guidance=guidance,
            timeshift=timeshift,
            save_maps=save_maps
        )
        
        self.conditioner.to("cpu")

    def __del__(self):
        self.tmp_dir.cleanup()

    @torch.no_grad()
    def compute_prompt_embs(self, prompt_format, labels):
        embeddings_and_length_list = [self.conditioner(prompt_format.format(prep="an" if x[0] in "aeiou" else "a", target=x) if x != "something" else "The image depicts a something") for x in labels]
        return torch.cat([x[0] for x in embeddings_and_length_list],dim=0), [x[1].item() for x in embeddings_and_length_list]

    @torch.no_grad()
    def __call__(self, x, label_ids, neg_prompt, num_images, image_height, image_width, extra_dict):
        image_height = image_height // 32 * 32 # TODO: related to attention maps resolution ? could get finer maps?
        image_width = image_width // 32 * 32
        self.denoiser.decoder_patch_scaling_h = image_height / 512
        self.denoiser.decoder_patch_scaling_w = image_width / 512

        xT = torch.stack([x] * num_images, dim=0)
        xT = (xT.float() / 127.5) - 1
        xT = xT.to(local_config.device)
        condition = torch.stack([self.prompt_embeddings[lidx.item()] for lidx in label_ids], dim=0)
        token_lengths = [self.token_lenghts[lidx.item()] for lidx in label_ids]
        attention_maps = self.diffusion_sampler(self.denoiser, xT, label_ids, condition, condition, token_lengths, extra_dict=extra_dict)
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
        # vae_config = config.model.vae
        denoiser_config = config.model.denoiser
        conditioner_config = config.model.conditioner

        # vae = instantiate_class(vae_config)
        denoiser = instantiate_class(denoiser_config)
        conditioner = instantiate_class(conditioner_config)


        ckpt = torch.load(ckpt_path, map_location="cpu")
        denoiser = load_model(ckpt, denoiser)
        denoiser = denoiser.to(local_config.device)
        # vae = vae.to(local_config.device)
        denoiser.eval()

        #TODO delete
        self.idx = 0
        import json
        self.dataset_config = local_config.datasets[local_config.eval_dataset]
        with open(self.dataset_config["json"]) as f:
            self.categs = json.load(f)
        if local_config.eval_dataset == local_config.EvalDatasets.VOC12:
            self.categs = ["something"] + self.categs
        else:
            self.categs = ["background"] + self.categs
        self.categs_count = len(self.categs)

        # TODO: nums steps hardcoded
        prompt_format = "The object in the image depicts {prep} {target}"
        self.pipeline = Pipeline(None, denoiser, conditioner, None, local_config.device, 100, local_config.guidance,
                                 local_config.timeshift, local_config.order, local_config.save_maps, prompt_format=prompt_format, labels=self.categs)
        denoiser.set_default_prompt_emb(self.pipeline.prompt_embeddings, self.pipeline.token_lenghts)

        for i in range(16):
            if not os.path.isdir(f"data/{i}"):
                os.mkdir(f"data/{i}")
            for j in range(22):
                if not os.path.isdir(f"data/{i}/{j}"):
                    os.mkdir(f"data/{i}/{j}")
        os.mkdir("data/gt")

    def get_gt_shape(self, x):
        gt_file_path = x[0]['file_name'].replace(self.dataset_config["img_dir"], self.dataset_config["gt_dir"]) \
            .replace(".jpg", self.dataset_config["extention"])
        mask = Image.open(gt_file_path)
        mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0)
        return mask_tensor.shape

    def get_gt(self, x):
        gt_file_path = x[0]['file_name'].replace(self.dataset_config["img_dir"], self.dataset_config["gt_dir"]) \
            .replace(".jpg", self.dataset_config["extention"])
        mask = Image.open(gt_file_path)
        return torch.from_numpy(np.array(mask)).unsqueeze(0)
    
    def get_gt_labels(self, x):
        gt_file_path = x[0]['file_name'].replace(self.dataset_config["img_dir"], self.dataset_config["gt_dir"]) \
            .replace(".jpg", self.dataset_config["extention"])
        mask = Image.open(gt_file_path)
        mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0)
        if local_config.eval_dataset == local_config.EvalDatasets.VOC12:
            filter_condition = lambda idx: idx > 0 and idx < 255
        else:
            filter_condition = lambda idx: idx > 0 # TODO: check for ade150
        return [(i, self.categs[i]) for i in torch.unique(mask_tensor).to(torch.uint8) if filter_condition(i)]

    def call_with_defaults(self, x, label_ids, extra_dict):
        return self.pipeline(
            x,
            label_ids,
            local_config.neg_label,
            local_config.num_images,
            local_config.image_height,
            local_config.image_width,
            extra_dict
        )
    
    @staticmethod
    def resize_maps(attention_maps, new_shape):
        return F.interpolate(attention_maps, new_shape, mode="bilinear", align_corners=False)
    
    def postprocess_voc12(self, prediction):
        # final_prediction = torch.zeros((C, H, W),dtype=prediction.dtype).to(prediction.device)
        # for i in range(1, C):
        #     final_prediction[i-1] += prediction[i]
        # final_prediction[-1] += prediction[0]
        # return final_prediction
        return prediction[1:, :, :] # TODO: check
    
    def postprocess_ade150(self, prediction):
        return prediction[1:, :, :] # TODO: check
    
    @torch.no_grad()
    def forward_no_grad(self, x):
        image_tensor = x[0]["image"]
        gt_idxs_and_labels = self.get_gt_labels(x)
        gt_shape = self.get_gt_shape(x)
        prompt_format = "The image depicts {prep} {target}"
        # TODO: was a +1: len of categs+1
        prediction = torch.zeros((self.categs_count, gt_shape[-2], gt_shape[-1])).to(local_config.device)
        # init background score TODO: do not hardcode treshold
        background_idx = 0
        prediction[background_idx] += local_config.background_threshold

        prompts = [prompt_format.format(prep="an" if idx_and_label[1][0] in "aeiou" else "a", target=idx_and_label[1]) for idx_and_label in gt_idxs_and_labels]# + [local_config.neg_label]
        label_ids = [idx_and_label[0] for idx_and_label in gt_idxs_and_labels]
        extra_dict = {
            "prompts": prompts,
            "eval_mode": local_config.eval,
            "img_id": self.idx,
        }

        gt_save_path = f"data/gt/{self.idx}.pt"
        gt = self.get_gt(x)
        torch.save(gt[0], gt_save_path)
        
        attention_maps = self.call_with_defaults(image_tensor, label_ids, extra_dict)
        # select slice corresponding to positive prompt
        attention_maps = attention_maps.unsqueeze(1)
        resized_map = DeCoSegmentor.resize_maps(attention_maps, gt_shape[1:]).squeeze(1)

        cam_dict = {}
        for i, label_and_idx in enumerate(gt_idxs_and_labels):
            image_label = label_and_idx[0].item() - 1
            cam_dict[str(image_label)] = (resized_map[i] * 255).cpu().numpy()
        
        get_fid = lambda x: x.split('/')[-1].replace('.jpg', '')
        save_path = os.path.join("sio_maps", "images", f'{get_fid(x[0]["file_name"])}.mat')
        sio.savemat(save_path, cam_dict, do_compression=True)
        
        for i, label_and_idx in enumerate(gt_idxs_and_labels):
            label_idx = label_and_idx[0]
            prediction[label_idx.item()] += resized_map[i]

        # visualize_prediction(self.resize_maps(image_tensor.unsqueeze(0), gt_shape[1:]), prediction.detach().cpu(), 
        #                      self.categs,
        #                      x[0]['file_name'].split("/")[-1].replace(".jpg", ""))
        self.idx += 1
        postprocess_func = self.postprocess_ade150 if local_config.eval_dataset == local_config.EvalDatasets.ADE150 else self.postprocess_voc12
        return [{"sem_seg": postprocess_func(prediction)}] # TODO: verify slicing
    
    def forward(self, x):
        return self.forward_no_grad(x)