# HW5 FID Training Copy

This folder is a working copy. The original sample files under `public_data/sample_code/` are untouched.

## Recommended first run on RTX 3080

```powershell
python code/train_fid.py --output_dir outputs/fid_base --preset base --batch_size 12 --epochs 160 --sample_every 2000 --save_every 5000
```

Smoke test:

```powershell
python code/train_fid.py --output_dir outputs/smoke --preset small --batch_size 2 --epochs 1 --max_train_steps 2 --sample_every 1 --save_every 2 --preview_batch_size 2 --preview_steps 2 --num_workers 0
```

If CUDA runs out of memory:

```powershell
python code/train_fid.py --output_dir outputs/fid_small --preset small --batch_size 12 --epochs 180
```

If there is still not enough VRAM, keep `--preset small` and reduce `--batch_size` to `8`.

## Generate 3000 images

Use the EMA checkpoint first.

```powershell
python code/inference_fid.py --checkpoint_dir outputs/fid_base/checkpoints/unet_ema_50000 --output_dir results --scheduler dpm --num_inference_steps 80 --batch_size 32 --num_samples 3000
```

Good FID sweeps to try:

```powershell
python code/inference_fid.py --checkpoint_dir outputs/fid_base/checkpoints/unet_ema_50000 --output_dir results_dpm80 --scheduler dpm --num_inference_steps 80 --seed 2026
python code/inference_fid.py --checkpoint_dir outputs/fid_base/checkpoints/unet_ema_50000 --output_dir results_dpm120 --scheduler dpm --num_inference_steps 120 --seed 2026
python code/inference_fid.py --checkpoint_dir outputs/fid_base/checkpoints/unet_ema_50000 --output_dir results_ddim100 --scheduler ddim --num_inference_steps 100 --seed 2026
```

## Local FID

The scoring program expects an `input` folder with `ref` and `res`.

```powershell
New-Item -ItemType Directory -Force -Path local_score/input
Copy-Item -Recurse -Force ref local_score/input/ref
Copy-Item -Recurse -Force results_dpm80 local_score/input/res
New-Item -ItemType Directory -Force -Path local_score/output
python scoring_program/score.py --input_dir local_score/input --output_dir local_score/output --config config.json
Get-Content local_score/output/scores.json
```

## FID strategy

The model trains in the Stable Diffusion VAE latent space (`4x32x32`) instead of pixel space. This makes training much cheaper and lets the UNet spend capacity on distribution matching rather than raw RGB detail.

Use EMA checkpoints for submission. EMA smooths noisy late-stage updates and usually improves generated sample quality and FID.

Use only horizontal flip augmentation. Strong color or geometry augmentation can create images outside the real professor-photo distribution and may hurt FID.

Sweep checkpoints and samplers. The best FID is often not the final raw checkpoint. Test EMA checkpoints around `30k`, `40k`, `50k`, and later, then compare `dpm` 80/120 steps and `ddim` 100 steps.

Train long enough for diversity. On 5500 images, 100 epochs is a baseline; 160-220 epochs is more likely to lower FID if samples do not visibly overfit.
