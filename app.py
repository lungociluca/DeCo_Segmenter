#   vae:
#     class_path: src.models.vae.LatentVAE
#     init_args:
#       precompute: true
#       weight_path: /mnt/bn/wangshuai6/models/sd-vae-ft-ema/
#   denoiser:
#     class_path: src.models.denoiser.decoupled_improved_dit.DDT
#     init_args:
#       in_channels: 4
#       patch_size: 2
#       num_groups: 16
#       hidden_size: &hidden_dim 1152
#       num_blocks: 28
#       num_encoder_blocks: 22
#       num_classes: 1000
#   conditioner:
#     class_path: src.models.conditioner.LabelConditioner
#     init_args:
#       null_class: 1000
#   diffusion_sampler:
#     class_path: src.diffusion.stateful_flow_matching.sampling.EulerSampler
#     init_args:
#       num_steps: 250
#       guidance: 3.0
#       state_refresh_rate: 1
#       guidance_interval_min: 0.3
#       guidance_interval_max: 1.0
#       timeshift: 1.0
#       last_step: 0.04
#       scheduler: *scheduler
#       w_scheduler: src.diffusion.stateful_flow_matching.scheduling.LinearScheduler
#       guidance_fn: src.diffusion.base.guidance.simple_guidance_fn
#       step_fn: src.diffusion.stateful_flow_matching.sampling.ode_step_fn
import random
import os
import torch
import argparse
from omegaconf import OmegaConf
from src.models.autoencoder.base import fp2uint8
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.adam_sampling import AdamLMSampler
from src.diffusion.flow_matching.scheduling import LinearScheduler
from PIL import Image
import tempfile
from huggingface_hub import snapshot_download

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
    def __init__(self, vae, denoiser, conditioner, resolution, device):
        self.vae = vae.to(device)
        self.denoiser = denoiser.to(device)
        self.conditioner = conditioner.to(device)
        self.conditioner.compile()
        self.resolution = resolution
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="traj_gifs_")
        # self.denoiser.compile()

    def __del__(self):
        self.tmp_dir.cleanup()

    @torch.no_grad()
    @torch.autocast(device_type=local_config.device, dtype=torch.bfloat16)
    def __call__(self, y, neg_prompt, num_images, seed, image_height, image_width, num_steps, guidance, timeshift, order, save_maps=False):
        diffusion_sampler = AdamLMSampler(
            order=order,
            scheduler=LinearScheduler(),
            guidance_fn=simple_guidance_fn,
            num_steps=num_steps,
            guidance=guidance,
            timeshift=timeshift,
            save_maps=save_maps
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image_height = image_height // 32 * 32
        image_width = image_width // 32 * 32
        self.denoiser.decoder_patch_scaling_h = image_height / 512
        self.denoiser.decoder_patch_scaling_w = image_width / 512
        xT = torch.randn((num_images, 3, image_height, image_width), device="cpu", dtype=torch.float32,
                         generator=generator)
        xT = xT.to(local_config.device)
        with torch.no_grad():
            condition, uncondition = conditioner([y,]*num_images, {"negative_prompt": neg_prompt})


        # Sample images:
        samples, trajs = diffusion_sampler(denoiser, xT, condition, uncondition, return_x_trajs=True)

        def decode_images(samples):
            samples = vae.decode(samples)
            samples = fp2uint8(samples)
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()
            images = []
            for i in range(len(samples)):
                image = Image.fromarray(samples[i])
                images.append(image)
            return images

        def decode_trajs(trajs):
            cat_trajs = torch.stack(trajs, dim=0).permute(1, 0, 2, 3, 4)
            animations = []
            for i in range(cat_trajs.shape[0]):
                frames = decode_images(
                    cat_trajs[i]
                )
                # 生成唯一文件名（结合seed和样本索引，避免冲突）
                gif_filename = f"{random.randint(0, 100000)}.gif"
                gif_path = os.path.join(self.tmp_dir.name, gif_filename)
                frames[0].save(
                    gif_path,
                    format="GIF",
                    append_images=frames[1:],
                    save_all=True,
                    duration=200,
                    loop=0
                )
                animations.append(gif_path)
            return animations

        images = decode_images(samples)
        animations = decode_trajs(trajs)

        return images, animations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs_t2i/sft_res512.yaml")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model_id", type=str, default="MCG-NJU/PixNerd-XXL-P16-T2I")
    parser.add_argument("--ckpt_path", type=str, default="models")

    args = parser.parse_args()
    if not os.path.exists(args.ckpt_path):
        snapshot_download(repo_id=args.model_id, local_dir=args.ckpt_path)
        ckpt_path = os.path.join(args.ckpt_path, "model.ckpt")
    else:
        ckpt_path = args.ckpt_path

    config = OmegaConf.load(args.config)
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


    pipeline = Pipeline(vae, denoiser, conditioner, args.resolution, local_config.device)

    num_steps = local_config.num_steps
    guidance =local_config.guidance
    image_height = local_config.image_height
    image_width = local_config.image_width
    num_images = local_config.num_images
    label = local_config.label
    neg_label = local_config.neg_label
    seed = local_config.seed
    timeshift = local_config.timeshift
    order = local_config.order
    save_maps = local_config.save_maps


    images, animations = pipeline(
        label,
        neg_label,
        num_images,
        seed,
        image_height,
        image_width,
        num_steps,
        guidance,
        timeshift,
        order,
        save_maps
    )
    images[0].save(local_config.out_image_path)