import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


@dataclass
class ModelPreset:
    block_out_channels: tuple
    layers_per_block: int
    down_block_types: tuple
    up_block_types: tuple
    attention_head_dim: int


PRESETS = {
    "small": ModelPreset(
        block_out_channels=(128, 192, 256, 384),
        layers_per_block=2,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    ),
    "base": ModelPreset(
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    ),
    "large": ModelPreset(
        block_out_channels=(192, 384, 512, 768),
        layers_per_block=2,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
    ),
}


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int = 256, random_flip: bool = True):
        self.image_dir = Path(image_dir)
        self.image_paths = sorted(p for p in self.image_dir.glob("*.png") if p.is_file())
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG files found in {self.image_dir}")

        ops: List[object] = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        ]
        if random_flip:
            ops.append(transforms.RandomHorizontalFlip(p=0.5))
        ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


class EMA:
    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay: float = 0.9999, warmup_steps: int = 2000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.params = [p for p in parameters if p.requires_grad]
        self.shadow = [p.detach().clone().float() for p in self.params]
        self.backup = None

    def _decay(self):
        if self.warmup_steps <= 0:
            return self.decay
        warmup_decay = 1.0 - 1.0 / (self.num_updates + 1)
        return min(self.decay, warmup_decay)

    @torch.no_grad()
    def update(self):
        self.num_updates += 1
        decay = self._decay()
        one_minus_decay = 1.0 - decay
        for shadow, param in zip(self.shadow, self.params):
            shadow.sub_(one_minus_decay * (shadow - param.detach().float()))

    @torch.no_grad()
    def store(self):
        self.backup = [p.detach().clone() for p in self.params]

    @torch.no_grad()
    def copy_to(self):
        for shadow, param in zip(self.shadow, self.params):
            param.copy_(shadow.to(dtype=param.dtype, device=param.device))

    @torch.no_grad()
    def restore(self):
        if self.backup is None:
            return
        for backup, param in zip(self.backup, self.params):
            param.copy_(backup)
        self.backup = None

    def state_dict(self):
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
            "shadow": [s.cpu() for s in self.shadow],
        }


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_unet(preset_name: str) -> UNet2DModel:
    preset = PRESETS[preset_name]
    return UNet2DModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=preset.layers_per_block,
        block_out_channels=preset.block_out_channels,
        down_block_types=preset.down_block_types,
        up_block_types=preset.up_block_types,
        attention_head_dim=preset.attention_head_dim,
        norm_num_groups=32,
        act_fn="silu",
    )


def build_train_scheduler(prediction_type: str) -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        prediction_type=prediction_type,
    )


def build_sample_scheduler(name: str, train_scheduler: DDPMScheduler):
    if name == "ddim":
        return DDIMScheduler.from_config(train_scheduler.config)
    if name == "dpm":
        return DPMSolverMultistepScheduler.from_config(train_scheduler.config)
    if name == "ddpm":
        return DDPMScheduler.from_config(train_scheduler.config)
    raise ValueError(f"Unknown sample scheduler: {name}")


def build_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def compute_snr(scheduler: DDPMScheduler, timesteps: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(device=timesteps.device)
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alpha_prod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    return (sqrt_alpha_prod / sqrt_one_minus_alpha_prod) ** 2


@torch.no_grad()
def sample_images(
    unet: UNet2DModel,
    vae: AutoencoderKL,
    train_scheduler: DDPMScheduler,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    num_inference_steps: int,
    scheduler_name: str,
    seed: int,
):
    unet.eval()
    vae.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    scheduler = build_sample_scheduler(scheduler_name, train_scheduler)
    scheduler.set_timesteps(num_inference_steps, device=device)

    latents = torch.randn((batch_size, 4, 32, 32), generator=generator, device=device, dtype=dtype)
    for timestep in scheduler.timesteps:
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.type == "cuda" and dtype == torch.float16):
            model_output = unet(latents, timestep).sample
        latents = scheduler.step(model_output, timestep, latents).prev_sample

    vae_dtype = next(vae.parameters()).dtype
    latents = (latents.float() / vae.config.scaling_factor).to(dtype=vae_dtype)
    images = vae.decode(latents, return_dict=False)[0]
    return (images.clamp(-1, 1) + 1) / 2


def save_checkpoint(unet, ema, output_dir: Path, step: int, save_ema: bool):
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = ckpt_dir / f"unet_{step}"
    unet.save_pretrained(raw_dir)
    if save_ema and ema is not None:
        ema.store()
        ema.copy_to()
        ema_dir = ckpt_dir / f"unet_ema_{step}"
        unet.save_pretrained(ema_dir)
        torch.save(ema.state_dict(), ckpt_dir / f"ema_{step}.pt")
        ema.restore()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a latent DDPM UNet for HW5 FID optimization.")
    parser.add_argument("--image_dir", type=str, default="public_data/images")
    parser.add_argument("--output_dir", type=str, default="outputs/fid_base")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="base")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--prediction_type", choices=["epsilon", "v_prediction"], default="epsilon")
    parser.add_argument("--latent_mode", choices=["sample", "mode"], default="sample")
    parser.add_argument("--snr_gamma", type=float, default=None)
    parser.add_argument("--noise_offset", type=float, default=0.0)
    parser.add_argument("--mixed_precision", choices=["fp16", "no"], default="fp16")
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_warmup_steps", type=int, default=2000)
    parser.add_argument("--random_flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--sample_every", type=int, default=2000)
    parser.add_argument("--preview_batch_size", type=int, default=16)
    parser.add_argument("--preview_steps", type=int, default=50)
    parser.add_argument("--preview_scheduler", choices=["ddim", "dpm", "ddpm"], default="dpm")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.mixed_precision == "fp16"
    amp_dtype = torch.float16 if use_amp else torch.float32
    print(f"device={device}, preset={args.preset}, amp={use_amp}")

    dataset = ImageDataset(args.image_dir, image_size=256, random_flip=args.random_flip)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)
    vae.eval()

    unet = build_unet(args.preset).to(device)
    unet.train()
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    train_scheduler = build_train_scheduler(args.prediction_type)
    total_update_steps = math.ceil(len(dataloader) / args.gradient_accumulation_steps) * args.epochs
    if args.max_train_steps is not None:
        total_update_steps = min(total_update_steps, args.max_train_steps)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=total_update_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMA(unet.parameters(), decay=args.ema_decay, warmup_steps=args.ema_warmup_steps) if args.use_ema else None

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(total=total_update_steps, desc="optimizer steps")
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for batch_idx, pixel_values in enumerate(dataloader, start=1):
            pixel_values = pixel_values.to(device, non_blocking=True)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    latent_dist = vae.encode(pixel_values).latent_dist
                    if args.latent_mode == "mode":
                        latents = latent_dist.mode()
                    else:
                        latents = latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            if args.noise_offset > 0:
                noise = noise + args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device,
                    dtype=latents.dtype,
                )
            timesteps = torch.randint(
                0,
                train_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = train_scheduler.add_noise(latents, noise, timesteps)
            target = noise
            if args.prediction_type == "v_prediction":
                target = train_scheduler.get_velocity(latents, noise, timesteps)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                model_pred = unet(noisy_latents, timesteps).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, loss.ndim)))
                if args.snr_gamma is not None:
                    snr = compute_snr(train_scheduler, timesteps)
                    min_snr = torch.minimum(snr, torch.full_like(snr, args.snr_gamma))
                    if args.prediction_type == "v_prediction":
                        loss_weight = min_snr / (snr + 1.0)
                    else:
                        loss_weight = min_snr / snr
                    loss = loss * loss_weight
                loss = loss.mean()
                loss_for_backward = loss / args.gradient_accumulation_steps

            scaler.scale(loss_for_backward).backward()
            running_loss += loss.item()

            do_update = batch_idx % args.gradient_accumulation_steps == 0
            if do_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                if ema is not None:
                    ema.update()
                global_step += 1
                pbar.update(1)
                pbar.set_postfix(loss=f"{running_loss / batch_idx:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                if global_step % args.sample_every == 0:
                    if ema is not None:
                        ema.store()
                        ema.copy_to()
                    images = sample_images(
                        unet=unet,
                        vae=vae,
                        train_scheduler=train_scheduler,
                        device=device,
                        dtype=amp_dtype,
                        batch_size=args.preview_batch_size,
                        num_inference_steps=args.preview_steps,
                        scheduler_name=args.preview_scheduler,
                        seed=args.seed + global_step,
                    )
                    save_image(make_grid(images.cpu(), nrow=4), samples_dir / f"step_{global_step:06d}.png")
                    if ema is not None:
                        ema.restore()
                    unet.train()

                if global_step % args.save_every == 0:
                    save_checkpoint(unet, ema, output_dir, global_step, save_ema=args.use_ema)

                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    break

        print(f"epoch={epoch:03d}, mean_loss={running_loss / max(1, len(dataloader)):.6f}, step={global_step}")
        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

    save_checkpoint(unet, ema, output_dir, global_step, save_ema=args.use_ema)
    print(f"Finished. Best inference candidate is usually: {output_dir / 'checkpoints' / f'unet_ema_{global_step}'}")


if __name__ == "__main__":
    main()
