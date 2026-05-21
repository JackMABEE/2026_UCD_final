"""DDIM inversion quality test.

Encodes an image, inverts to z_T, then denoises back with the same prompt
and no attention injection. A faithful reconstruction verifies the inversion
is cycle-consistent before injection is layered on top.

Usage:
    python scripts/inversion_test.py [--steps N] [--dtype fp16|fp32]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from loguru import logger
from PIL import Image

from attn_texture.utils.device import get_device_and_dtype, safe_to
from attn_texture.utils.io import load_image, pil_to_tensor, tensor_to_pil

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOURCE = Path("assets/inputs/41.png")
REF_PROMPT = "a fabric with floral pattern"
IMAGE_SIZE = (512, 512)
MODEL_ID = "runwayml/stable-diffusion-v1-5"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def encode_prompt(tokenizer, text_encoder, prompt: str, device, dtype) -> torch.Tensor:
    ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids.to(device)
    with torch.no_grad():
        return text_encoder(ids).last_hidden_state


def ddim_inversion(
    unet, scheduler, z_0: torch.Tensor, embeds: torch.Tensor
) -> torch.Tensor:
    """Deterministic DDIM forward: z_0 → z_T (ascending timesteps)."""
    timesteps_fwd = scheduler.timesteps.flip(0)
    alphas = scheduler.alphas_cumprod.to(z_0.device, z_0.dtype)

    z = z_0.clone()
    for i, t in enumerate(timesteps_fwd):
        alpha_t = alphas[t]
        with torch.inference_mode():
            noise_pred = unet(z, t, encoder_hidden_states=embeds).sample
        z_0_pred = (z - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        if i + 1 < len(timesteps_fwd):
            alpha_next = alphas[timesteps_fwd[i + 1]]
        else:
            alpha_next = torch.zeros_like(alpha_t)
        z = alpha_next.sqrt() * z_0_pred + (1.0 - alpha_next).sqrt() * noise_pred

    logger.info("Inversion complete, z_T shape={}", z.shape)
    return z


def ddim_denoise(
    unet, scheduler, z_T: torch.Tensor, embeds: torch.Tensor
) -> torch.Tensor:
    """Standard DDIM denoising: z_T → z_0 (descending timesteps, no injection)."""
    latents = z_T.clone()
    for t in scheduler.timesteps:
        with torch.inference_mode():
            noise_pred = unet(latents, t, encoder_hidden_states=embeds).sample
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    logger.info("Denoising complete, z_0' shape={}", latents.shape)
    return latents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    device, auto_dtype = get_device_and_dtype()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    num_steps = args.steps
    out_dir = args.out_dir or Path(f"experiments/inversion_test_{args.dtype}_{num_steps}steps")

    logger.info("device={}  dtype={}  steps={}", device, dtype, num_steps)

    # Load models
    logger.info("Loading SD 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, safety_checker=None
    ).to(device)
    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    scheduler.set_timesteps(num_steps)

    vae_scale: float = float(vae.config.scaling_factor)

    # Encode source image
    source = load_image(SOURCE, size=IMAGE_SIZE)
    pixel = pil_to_tensor(source).unsqueeze(0)           # (1, 3, H, W) [0,1]
    pixel = safe_to(pixel, device, dtype) * 2.0 - 1.0   # [-1, 1]

    with torch.inference_mode():
        z_0 = vae.encode(pixel).latent_dist.sample() * vae_scale
    logger.info("VAE encode done, z_0 shape={}", z_0.shape)

    # Encode prompt
    embeds = encode_prompt(tokenizer, text_encoder, REF_PROMPT, device, dtype)

    # DDIM inversion: z_0 → z_T
    logger.info("Running DDIM inversion ({} steps)...", num_steps)
    z_T = ddim_inversion(unet, scheduler, z_0, embeds)

    # DDIM denoising: z_T → z_0' (same prompt, no injection)
    logger.info("Denoising back ({} steps)...", num_steps)
    z_0_reconstructed = ddim_denoise(unet, scheduler, z_T, embeds)

    # Decode both
    with torch.inference_mode():
        decoded_orig = vae.decode(z_0 / vae_scale).sample
        decoded_recon = vae.decode(z_0_reconstructed / vae_scale).sample

    decoded_orig = decoded_orig / 2.0 + 0.5
    decoded_recon = decoded_recon / 2.0 + 0.5

    img_orig = tensor_to_pil(decoded_orig[0].float())
    img_recon = tensor_to_pil(decoded_recon[0].float())

    # Build side-by-side panel: original | VAE round-trip | reconstructed
    panel_w = IMAGE_SIZE[0] * 3
    panel_h = IMAGE_SIZE[1] + 24
    panel = Image.new("RGB", (panel_w, panel_h), color=(30, 30, 30))

    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    tag = f"{args.dtype} / {num_steps} steps"
    labels = ["original", "VAE round-trip", f"DDIM recon ({tag})"]
    images = [source, img_orig, img_recon]
    for i, (img, label) in enumerate(zip(images, labels)):
        panel.paste(img.convert("RGB"), (i * IMAGE_SIZE[0], 24))
        draw.text((i * IMAGE_SIZE[0] + 4, 4), label, fill=(255, 255, 255), font=font)

    out_dir.mkdir(parents=True, exist_ok=True)
    panel.save(out_dir / "reconstructed.png")
    logger.info("Saved → {}", out_dir / "reconstructed.png")

    # Pixel-level stats vs original
    import math
    import torch.nn.functional as F
    orig_t = pil_to_tensor(source).unsqueeze(0)
    recon_t = pil_to_tensor(img_recon).unsqueeze(0)
    mse = F.mse_loss(recon_t, orig_t).item()
    psnr_db = 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    logger.info("Reconstruction  MSE={:.6f}  PSNR={:.2f} dB", mse, psnr_db)


if __name__ == "__main__":
    main()
