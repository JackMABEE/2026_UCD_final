"""Phase 2 final composite: re-run 'ours' pipeline on a texture source, then
FFT-blend the result onto a target person image within a pre-computed mask.

Usage
-----
python -m attn_texture.cli.run_composite \\
    --texture-source  assets/textures/extracted_floral.png \\
    --person          assets/inputs/person_floral_street.png \\
    --mask            experiments/2026-05-25_phase2_floral_genmask/mask.png \\
    --ref-prompt      "a blue floral fabric texture" \\
    --gen-prompt      "a white cotton shirt fabric, same pattern layout" \\
    --exp-name        "2026-05-25_white_to_floral" \\
    [--config         configs/phase1_global.yaml]

What it does
------------
1. Re-run TwoPassPipeline (ours) on texture-source → ours_texture.png.
2. FFT-blend the result onto person within the mask:
     L channel: low_freq from person (lighting), high_freq from ours (texture).
     A/B chroma: kept from person (preserves original shirt colour).
3. Apply mask composite: blend inside mask, person unchanged outside.
4. Save final_composite.png and a 2-panel comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image

from attn_texture.core.fft_blend import _lab_to_rgb, _rgb_to_lab, blend_latents
from attn_texture.core.two_pass_pipeline import TwoPassPipeline
from attn_texture.utils.device import get_device_and_dtype
from attn_texture.utils.io import load_image
from attn_texture.utils.memory import with_isolated_model
from attn_texture.utils.seed import seed_everything

_DEFAULT_CONFIG = Path(__file__).parents[3] / "configs" / "phase1_global.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_sd_components(cfg):
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    logger.info("Loading SD 1.5 from '{}'…", cfg.model_id)
    device, dtype = get_device_and_dtype()
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe.unet, pipe.vae, scheduler, pipe.tokenizer, pipe.text_encoder


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB → (1, 3, H, W) float32 in [0, 1], on CPU."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """(1, 3, H, W) float in [0, 1] → PIL RGB."""
    arr = t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).float().cpu().numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))


def _build_comparison(left: Image.Image, right: Image.Image, labels: list[str]) -> Image.Image:
    """2-panel side-by-side with burned-in labels."""
    W, H = left.size
    label_h = 24
    panel = Image.new("RGB", (W * 2, H + label_h), (30, 30, 30))

    from PIL import ImageDraw

    draw = ImageDraw.Draw(panel)
    for i, (img, label) in enumerate(zip([left, right], labels)):
        panel.paste(img, (i * W, label_h))
        tw = draw.textlength(label)
        draw.text((i * W + (W - tw) // 2, 4), label, fill=(255, 255, 255))
    return panel


# ---------------------------------------------------------------------------
# Core composite logic
# ---------------------------------------------------------------------------


def _fft_blend_within_mask(
    person_t: torch.Tensor,
    ours_t: torch.Tensor,
    mask_t: torch.Tensor,
    cutoff_ratio: float,
) -> torch.Tensor:
    """FFT-blend person and ours in LAB space, keeping person's chroma.

    L channel: low_freq from person (original lighting), high_freq from ours
               (generated texture detail).
    A/B channels: always from person (preserves original shirt colour).

    Args:
        person_t:    (1, 3, H, W) float32 in [0, 1].
        ours_t:      (1, 3, H, W) float32 in [0, 1].
        mask_t:      (1, 1, H, W) float32 in [0, 1]; 1 = inside shirt.
        cutoff_ratio: low-freq radius fraction passed to blend_latents.

    Returns:
        Composited image, (1, 3, H, W) float32 in [0, 1].
    """
    person_lab = _rgb_to_lab(person_t)   # (1, 3, H, W)
    ours_lab = _rgb_to_lab(ours_t)

    # L: low_freq from person lighting, high_freq from ours texture
    blended_L = blend_latents(
        person_lab[:, 0:1], ours_lab[:, 0:1], cutoff_ratio
    )                                    # (1, 1, H, W)

    # A/B: keep person's original chroma (preserves the floral colour palette)
    blended_lab = torch.cat([blended_L, person_lab[:, 1:3]], dim=1)
    blend_t = _lab_to_rgb(blended_lab)  # (1, 3, H, W)

    # Spatial composite: blend inside mask, original person outside
    mask_3ch = mask_t.expand_as(person_t)  # (1, 3, H, W)
    return blend_t * mask_3ch + person_t * (1.0 - mask_3ch)


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 2 composite: paste 'ours' texture onto person within mask."
    )
    parser.add_argument("--texture-source", required=True, type=Path,
                        help="Source texture image for TwoPassPipeline.")
    parser.add_argument("--person", required=True, type=Path,
                        help="Target person image to composite onto.")
    parser.add_argument("--mask", required=True, type=Path,
                        help="Binary mask PNG (255=shirt, 0=background).")
    parser.add_argument("--ref-prompt", required=True, type=str,
                        help="Ref prompt describing texture-source.")
    parser.add_argument("--gen-prompt", required=True, type=str,
                        help="Gen prompt for TwoPassPipeline.")
    parser.add_argument("--exp-name", required=True, type=str,
                        help="Experiment name (output goes to experiments_root/<exp-name>/).")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG,
                        help=f"OmegaConf YAML config (default: {_DEFAULT_CONFIG}).")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    for p, name in [
        (args.texture_source, "--texture-source"),
        (args.person, "--person"),
        (args.mask, "--mask"),
        (args.config, "--config"),
    ]:
        if not p.exists():
            logger.error("{} not found: {}", name, p)
            raise SystemExit(1)

    cfg = OmegaConf.load(args.config)
    seed_everything(cfg.seed)

    exp_dir = Path(cfg.experiments_root) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, exp_dir / "config.yaml")

    image_size: tuple[int, int] = tuple(cfg.image_size)  # (W, H)

    logger.info("Phase 2 composite — exp: '{}'", args.exp_name)
    logger.info("  texture-source : {}", args.texture_source)
    logger.info("  person         : {}", args.person)
    logger.info("  mask           : {}", args.mask)

    # ── Step 1: run TwoPassPipeline on texture source ────────────────────────
    logger.info("─── Step 1: TwoPassPipeline (ours) on texture source ───")
    texture_img = load_image(args.texture_source, size=image_size)
    with with_isolated_model(lambda: _load_sd_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder = comps
        pipeline = TwoPassPipeline(unet, vae, scheduler, tokenizer, text_encoder, cfg.ours)
        ours_img: Image.Image = pipeline.run(
            source_image=texture_img,
            ref_prompt=args.ref_prompt,
            gen_prompt=args.gen_prompt,
            num_inference_steps=cfg.ours.num_inference_steps,
        )
    ours_path = exp_dir / "ours_texture.png"
    ours_img.save(ours_path)
    logger.info("Ours texture saved → {}", ours_path)

    # ── Step 2: load person + mask ───────────────────────────────────────────
    logger.info("─── Step 2: Load person image and mask ───")
    person_img = load_image(args.person, size=image_size)
    mask_pil = Image.open(args.mask).convert("L").resize(image_size, Image.NEAREST)
    coverage = np.array(mask_pil).mean() / 255.0 * 100.0
    logger.info("Mask coverage: {:.1f}%", coverage)

    person_t = _pil_to_tensor(person_img)                       # (1, 3, 512, 512)
    ours_t = _pil_to_tensor(ours_img)                           # (1, 3, 512, 512)
    mask_arr = np.array(mask_pil).astype(np.float32) / 255.0   # (512, 512)
    mask_t = torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 512, 512)

    # ── Step 3: FFT blend within mask ────────────────────────────────────────
    logger.info("─── Step 3: FFT blend (cutoff_ratio={}) ───", cfg.ours.fft_cutoff_ratio)
    final_t = _fft_blend_within_mask(
        person_t, ours_t, mask_t, cutoff_ratio=float(cfg.ours.fft_cutoff_ratio)
    )

    # ── Step 4: save artefacts ───────────────────────────────────────────────
    final_img = _tensor_to_pil(final_t)
    final_path = exp_dir / "final_composite.png"
    final_img.save(final_path)
    logger.info("Final composite saved → {}", final_path)

    comparison = _build_comparison(
        person_img, final_img,
        labels=["original", "composite"],
    )
    comparison_path = exp_dir / "comparison.png"
    comparison.save(comparison_path)
    logger.info("Comparison panel saved → {}", comparison_path)
    logger.info("All artefacts saved to {}", exp_dir)


if __name__ == "__main__":
    main()
