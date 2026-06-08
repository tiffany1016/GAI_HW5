import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from diffusers import UNet2DModel, DDPMScheduler, AutoencoderKL


class ImageDataset(Dataset):
    """Dataset for loading images from a directory."""
    
    def __init__(self, image_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.image_paths = sorted([p for p in self.image_dir.glob("*.png") if p.is_file()])
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        return image


@torch.no_grad()
def generate_and_save_images(unet, vae, scheduler, epoch, device, save_folder, enable_bar=False):
    """
    Your code here:
    Sample images using trained model.
    """
    pass


def train():
    # ========= Hyperparameters ==========
    train_epochs = 100
    batch_size = 8
    gradient_accumulation_steps = 1 
    # You can use gradients accumulation to simulate larger batch size if you have limited GPU memory.
    # Call optimizer.step() every `gradient_accumulation_steps` batches.
    lr = 1e-4
    eval_freq = 1000
    save_freq = 10000
    image_dir = "images"
    output_dir = "outputs"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== Saved folders ==========
    save_folder = os.path.join(output_dir, "samples")
    os.makedirs(save_folder, exist_ok=True)

    # ========== Load Pretrained Model ==========
    ## You cannot change this part
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.requires_grad_(False)

    # ========== Init ==========
    ## You should modify the model architecture by your self
    unet = UNet2DModel().to(device)
    unet.train()
    optimizer = torch.optim.Adam(list(unet.parameters()), lr=lr)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False)

    # ========== Dataset ==========
    dataset = ImageDataset(image_dir, image_size=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    # ========== Training ==========
    loss_accumulated = 0.0
    step = 0
    pbar = tqdm(total=train_epochs * len(dataloader), desc="Training")
    for epoch in range(train_epochs):
        for batch in dataloader:
            step += 1
            pixel_values = batch.to(device)

            # Encode images
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            """
            Your code here:
            1. Sample noise and timestep
            2. Add noise to the latents according to the noise scheduler
            3. Predict the noise using UNet
            4. Compute the loss (MSE between the predicted noise and the true noise)
            """
        pbar.update(1)

        if step % eval_freq == 0:
            generate_and_save_images(unet, vae, noise_scheduler, step, device, save_folder)
            unet.train()

    # Save final model
    unet.save_pretrained(os.path.join(save_folder, f"unet_final"))
