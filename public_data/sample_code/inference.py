import os
import argparse
import torch
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler, AutoencoderKL
from tqdm import tqdm


def generate_and_save_images(unet, vae, scheduler, device, save_folder, num_samples, batch_size=8):
    unet.eval()
    vae.eval()

    os.makedirs(save_folder, exist_ok=True)
    to_pil = transforms.ToPILImage()

    total_batches = (num_samples + batch_size - 1) // batch_size
    sample_idx = 0

    with torch.no_grad():
        for batch_idx in tqdm(range(total_batches)):
            current_batch = min(batch_size, num_samples - sample_idx)
            """
            Your sampling code.
            """

            latents = latents / vae.config.scaling_factor
            images = vae.decode(latents, return_dict=False)[0]
            images = (images.clamp(-1, 1) + 1) / 2

            for i in range(current_batch):
                output_path = os.path.join(save_folder, f"{sample_idx:04d}.png")
                image = to_pil(images[i].cpu())
                image.save(output_path)
                sample_idx += 1

    print(f"Saved {sample_idx} samples to {save_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate samples from a trained diffusion UNet model.")
    parser.add_argument("--checkpoint_dir", type=str, default="./outputs/baseline3/checkpoints/unet_3000",
                        help="Path to the trained UNet checkpoint folder saved with save_pretrained().")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory where generated images will be saved.")
    parser.add_argument("--num_samples", type=int, default=3000,
                        help="Total number of images to generate.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for latent generation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading VAE and UNet on {device}")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)

    unet = UNet2DModel.from_pretrained(args.checkpoint_dir).to(device)
    unet.requires_grad_(False)

    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)

    generate_and_save_images(
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        device=device,
        save_folder=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )
