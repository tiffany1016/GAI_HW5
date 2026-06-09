import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel
from torchvision import transforms
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(name: str, prediction_type: str, beta_schedule: str, use_karras_sigmas: bool, solver_order: int):
    train_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule=beta_schedule,
        clip_sample=False,
        prediction_type=prediction_type,
    )
    if name == "ddim":
        return DDIMScheduler.from_config(train_scheduler.config)
    if name == "dpm":
        return DPMSolverMultistepScheduler.from_config(
            train_scheduler.config,
            use_karras_sigmas=use_karras_sigmas,
            solver_order=solver_order,
        )
    if name == "ddpm":
        return DDPMScheduler.from_config(train_scheduler.config)
    raise ValueError(f"Unknown scheduler: {name}")


@torch.no_grad()
def generate_and_save_images(
    unet,
    vae,
    scheduler,
    device,
    dtype,
    save_folder,
    num_samples,
    batch_size,
    num_inference_steps,
    seed,
):
    unet.eval()
    vae.eval()
    os.makedirs(save_folder, exist_ok=True)
    to_pil = transforms.ToPILImage()
    generator = torch.Generator(device=device).manual_seed(seed)

    sample_idx = 0
    total_batches = (num_samples + batch_size - 1) // batch_size
    for _ in tqdm(range(total_batches), desc="generating"):
        scheduler.set_timesteps(num_inference_steps, device=device)
        current_batch = min(batch_size, num_samples - sample_idx)
        latents = torch.randn((current_batch, 4, 32, 32), generator=generator, device=device, dtype=dtype)

        for timestep in scheduler.timesteps:
            model_output = unet(latents, timestep).sample
            latents = scheduler.step(model_output, timestep, latents).prev_sample

        vae_dtype = next(vae.parameters()).dtype
        latents = (latents.float() / vae.config.scaling_factor).to(dtype=vae_dtype)
        images = vae.decode(latents, return_dict=False)[0]
        images = (images.clamp(-1, 1) + 1) / 2

        for image in images:
            output_path = Path(save_folder) / f"{sample_idx:04d}.png"
            to_pil(image.cpu()).save(output_path)
            sample_idx += 1

    print(f"Saved {sample_idx} samples to {save_folder}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HW5 submission images from a trained latent UNet.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=80)
    parser.add_argument("--scheduler", choices=["dpm", "ddim", "ddpm"], default="dpm")
    parser.add_argument("--use_karras_sigmas", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--solver_order", type=int, choices=[1, 2, 3], default=2)
    parser.add_argument("--prediction_type", choices=["epsilon", "v_prediction"], default="epsilon")
    parser.add_argument("--beta_schedule", choices=["scaled_linear", "squaredcos_cap_v2", "linear"], default="scaled_linear")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mixed_precision", choices=["fp16", "no"], default="fp16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and args.mixed_precision == "fp16"
    dtype = torch.float16 if use_amp else torch.float32

    print(f"Loading VAE and UNet on {device}")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)
    if use_amp:
        vae.to(dtype=torch.float16)

    unet = UNet2DModel.from_pretrained(args.checkpoint_dir).to(device)
    unet.requires_grad_(False)
    if use_amp:
        unet.to(dtype=torch.float16)

    scheduler = build_scheduler(
        args.scheduler,
        args.prediction_type,
        args.beta_schedule,
        args.use_karras_sigmas,
        args.solver_order,
    )
    generate_and_save_images(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        save_folder=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
