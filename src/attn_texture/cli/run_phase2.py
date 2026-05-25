"""CLI entry point for Phase 2: region-aware garment texture transfer.

Workflow (CLAUDE.md §5.3 — mask before render)
------------------------------------------------
**Step 1 — inspect the mask (required):**

    python -m attn_texture.cli.run_phase2 \\
        --source     path/to/person.jpg \\
        --ref-prompt "a person wearing a plain white shirt" \\
        --gen-prompt "a person wearing a striped silk shirt" \\
        --mask-word  shirt \\
        --exp-name   2026-05-25_person1_silk_stripe \\
        --dry-run-mask

    Saves experiments/<exp-name>/mask.png and mask_overlay.png.
    No generation happens.  Inspect the overlay and re-run without the flag.

**Step 2 — full render (only after mask is verified):**

    python -m attn_texture.cli.run_phase2 \\
        --source     path/to/person.jpg \\
        --ref-prompt "a person wearing a plain white shirt" \\
        --gen-prompt "a person wearing a striped silk shirt" \\
        --mask-word  shirt \\
        --exp-name   2026-05-25_person1_silk_stripe

    Requires mask.png to exist in the experiment directory.
    Saves outputs/result.png and comparison.png.

What it does
------------
Dry-run (--dry-run-mask):
  1. Load SD 1.5; encode source → VAE latent z_0.
  2. DDIM inversion (mask.dry_run_steps): z_0 → z_T.
  3. Register AttentionStore on every attn2 (cross-attention) layer.
  4. Denoising loop (mask.dry_run_steps): conditional pass accumulating
     cross-attention maps; store.between_steps() called after each step.
  5. Aggregate maps at 16×16 (mask.aggregate_res); build binary mask for
     --mask-word tokens; upsample to image resolution.
  6. Save grayscale mask.png + red-tint overlay on source (mask_overlay.png).

Full render (no flag):
  1. Verify mask.png exists — hard stop if absent (§5.3).
  2. Run TwoPassPipeline (PnP self-attention injection + FFT blend).
  3. Pixel-level composite: inside mask → generated, outside → source.
  4. Save outputs/result.png and side-by-side comparison.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image

from attn_texture.core.mask_extraction import (
    AttentionStore,
    aggregate_attention,
    build_mask,
    get_word_inds,
    register_attention_control,
)
from attn_texture.core.two_pass_pipeline import TwoPassPipeline
from attn_texture.utils.device import get_device_and_dtype, safe_to
from attn_texture.utils.io import load_image, pil_to_tensor, save_image
from attn_texture.utils.memory import with_isolated_model
from attn_texture.utils.seed import seed_everything

_DEFAULT_CONFIG = Path(__file__).parents[3] / "configs" / "phase2_local.yaml"


# ---------------------------------------------------------------------------
# Model factory (identical to run_phase1 — no ControlNet needed in Phase 2)
# ---------------------------------------------------------------------------


def _load_sd_components(cfg):
    """Load SD 1.5 UNet, VAE, DDIM scheduler, tokenizer, text encoder."""
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


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _encode_prompt(tokenizer, text_encoder, prompt: str, device: str) -> torch.Tensor:
    """Tokenize *prompt* and return text encoder hidden states (1, S, D)."""
    ids = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids.to(device)
    with torch.no_grad():
        return text_encoder(ids).last_hidden_state


def _ddim_inversion(
    unet,
    scheduler,
    z_0: torch.Tensor,
    ref_embeds: torch.Tensor,
) -> torch.Tensor:
    """Deterministic DDIM forward process: z_0 → z_T.

    Mirrors TwoPassPipeline._ddim_inversion but operates on externally-
    provided components so it can be reused inside the dry-run context.
    scheduler.set_timesteps() must have been called before this function.

    Args:
        unet:       UNet2DConditionModel.
        scheduler:  DDIMScheduler with .alphas_cumprod and .timesteps.
        z_0:        VAE-encoded source latent (1, 4, H, W).
        ref_embeds: text embeddings for the source prompt (1, S, D).

    Returns:
        Inverted latent z_T, same shape as z_0.
    """
    timesteps_fwd = scheduler.timesteps.flip(0)   # ascending for inversion
    alphas: torch.Tensor = scheduler.alphas_cumprod.to(z_0.device, z_0.dtype)

    z = z_0.clone()
    for i, t in enumerate(timesteps_fwd):
        alpha_t = alphas[t]
        noise_pred = unet(z, t, encoder_hidden_states=ref_embeds).sample
        z_0_pred = (z - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        if i + 1 < len(timesteps_fwd):
            alpha_next = alphas[timesteps_fwd[i + 1]]
        else:
            alpha_next = torch.zeros_like(alpha_t)
        z = alpha_next.sqrt() * z_0_pred + (1.0 - alpha_next).sqrt() * noise_pred

    logger.debug("_ddim_inversion: complete, z_T shape={}", z.shape)
    return z


# ---------------------------------------------------------------------------
# Mask visualization helper
# ---------------------------------------------------------------------------


def _make_mask_overlay(source: Image.Image, mask_arr: np.ndarray) -> Image.Image:
    """Overlay a semi-transparent red tint on masked pixels of *source*.

    Args:
        source:   RGB PIL Image.
        mask_arr: uint8 array (H, W); 255 = inside mask, 0 = outside.
                  Must match source dimensions.

    Returns:
        RGBA PIL Image with red overlay inside the mask.
    """
    overlay = source.copy().convert("RGBA")
    red = Image.new("RGBA", source.size, (220, 30, 30, 130))
    mask_pil = Image.fromarray(mask_arr, mode="L")
    overlay.paste(red, mask=mask_pil)
    return overlay


# ---------------------------------------------------------------------------
# Dry-run: mask extraction  (CLAUDE.md §5.3)
# ---------------------------------------------------------------------------


def _run_dry_run_mask(
    cfg,
    source_image: Image.Image,
    ref_prompt: str,
    gen_prompt: str,
    mask_word: str,
    exp_dir: Path,
) -> None:
    """Extract cross-attention mask and save visualisation. No generation.

    Saves:
      mask.png        — grayscale binary mask (255=garment, 0=background).
      mask_overlay.png — red tint over source at masked pixels for inspection.

    Args:
        cfg:          Full OmegaConf config (phase2_local.yaml).
        source_image: RGB PIL Image already resized to cfg.image_size.
        ref_prompt:   Source description prompt (used for DDIM inversion).
        gen_prompt:   Target generation prompt; must contain mask_word.
        mask_word:    Single word identifying the garment region.
        exp_dir:      Experiment directory; files written here.
    """
    logger.info("─── Dry-run: cross-attention mask for '{}' ───", mask_word)
    mask_cfg = cfg.mask
    W_img, H_img = source_image.size   # cfg.image_size is [W, H]

    mask_arr: np.ndarray  # set inside with_isolated_model; used for I/O outside

    with with_isolated_model(lambda: _load_sd_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder = comps
        device, dtype = get_device_and_dtype()
        vae_scale: float = float(vae.config.scaling_factor)

        with torch.inference_mode():
            # 1. Encode source image → z_0
            src_pixels = pil_to_tensor(source_image).unsqueeze(0)   # (1, 3, H, W)
            z_0 = vae.encode(
                safe_to(src_pixels, device, dtype) * 2.0 - 1.0
            ).latent_dist.sample() * vae_scale

            # 2. Encode prompts
            ref_embeds = _encode_prompt(tokenizer, text_encoder, ref_prompt, device)
            gen_embeds = _encode_prompt(tokenizer, text_encoder, gen_prompt, device)

            # 3. DDIM inversion: z_0 → z_T using dry_run_steps for speed
            scheduler.set_timesteps(int(mask_cfg.dry_run_steps))
            z_T = _ddim_inversion(unet, scheduler, z_0, ref_embeds)

            # 4. Register AttentionStore on attn2 layers (attn1 untouched)
            store = AttentionStore(max_spatial_side=int(mask_cfg.max_spatial_side))
            cleanup = register_attention_control(unet, store)

            # 5. Denoising loop — conditional pass accumulates cross-attention maps
            gen_latents = z_T.clone()
            for t in scheduler.timesteps:
                noise_pred = unet(
                    gen_latents, t,
                    encoder_hidden_states=gen_embeds,
                    return_dict=True,
                ).sample
                gen_latents = scheduler.step(noise_pred, t, gen_latents).prev_sample
                store.between_steps()

            # 6. Restore original processors before memory cleanup
            cleanup()

        # 7. Resolve token indices for mask_word (tokenizer still live in comps)
        word_inds = get_word_inds(tokenizer, gen_prompt, mask_word)
        if not word_inds:
            logger.error(
                "Word '{}' not found as a CLIP token in gen-prompt '{}'."
                " Check spelling — the word must appear verbatim in the prompt.",
                mask_word, gen_prompt,
            )
            sys.exit(1)
        logger.info("Token indices for '{}': {}", mask_word, word_inds)

        # 8. Aggregate cross-attention → binary mask at image resolution
        avg = store.get_average_attention()
        agg = aggregate_attention(
            avg,
            res=int(mask_cfg.aggregate_res),
            places=tuple(mask_cfg.aggregate_places),
        )
        mask_bool = build_mask(
            agg,
            word_inds=word_inds,
            threshold=float(mask_cfg.threshold),
            target_hw=(H_img, W_img),
        )
        # Convert while model is still in scope (avoids MPS/CUDA lifetime issues)
        mask_arr = mask_bool.cpu().numpy().astype(np.uint8) * 255

    # 9. Save mask and overlay  (I/O only — no model needed)
    mask_pil = Image.fromarray(mask_arr, mode="L")
    mask_pil.save(exp_dir / "mask.png")

    overlay = _make_mask_overlay(source_image, mask_arr)
    overlay.save(exp_dir / "mask_overlay.png")

    coverage = float(mask_arr.astype(bool).mean()) * 100.0
    logger.info(
        "Mask saved: {}  ({:.1f}% of pixels inside mask)",
        exp_dir / "mask.png", coverage,
    )
    logger.info("Overlay saved: {}", exp_dir / "mask_overlay.png")
    logger.info(
        "Inspect mask_overlay.png to verify the mask, then re-run without --dry-run-mask."
    )


# ---------------------------------------------------------------------------
# Full render  (CLAUDE.md §5.3 — requires prior dry-run)
# ---------------------------------------------------------------------------


def _run_full_render(
    cfg,
    source_image: Image.Image,
    ref_prompt: str,
    gen_prompt: str,
    exp_dir: Path,
) -> None:
    """Load pre-computed mask, run PnP pipeline, composite inside mask region.

    CLAUDE.md §5.3: no code path may reach this function without a valid
    mask.png being present — the CLI hard-stops if it is absent.

    Saves:
      outputs/result.png  — composited result (gen inside mask, src outside).
      comparison.png      — side-by-side source | result panel.

    Args:
        cfg:          Full OmegaConf config (phase2_local.yaml).
        source_image: RGB PIL Image already resized to cfg.image_size.
        ref_prompt:   Source description prompt.
        gen_prompt:   Target generation prompt.
        exp_dir:      Experiment directory; mask.png must exist here.
    """
    # §5.3 gate — hard stop if mask has not been verified
    mask_path = exp_dir / "mask.png"
    if not mask_path.exists():
        logger.error(
            "mask.png not found at '{}'.  Run with --dry-run-mask first to "
            "extract and verify the mask before a full render (CLAUDE.md §5.3).",
            mask_path,
        )
        sys.exit(1)

    # Load saved binary mask → float array (H, W) in [0, 1]
    mask_pil = Image.open(mask_path).convert("L")
    mask_np = np.array(mask_pil, dtype=np.float32) / 255.0   # (H, W)
    coverage = float((mask_np > 0.5).mean()) * 100.0
    logger.info("Loaded mask from '{}'  ({:.1f}% coverage)", mask_path, coverage)

    # Run TwoPassPipeline (PnP + FFT blend) on the full image
    logger.info("─── Running: TwoPassPipeline (PnP + FFT blend) ───")
    with with_isolated_model(lambda: _load_sd_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder = comps
        pipeline = TwoPassPipeline(
            unet, vae, scheduler, tokenizer, text_encoder, cfg.pnp
        )
        img_gen = pipeline.run(
            source_image=source_image,
            ref_prompt=ref_prompt,
            gen_prompt=gen_prompt,
            num_inference_steps=int(cfg.pnp.num_inference_steps),
        )
    logger.info("TwoPassPipeline: done.")

    # Pixel-level composite: inside mask → generated, outside → source
    src_arr = np.array(source_image, dtype=np.float32) / 255.0   # (H, W, 3)
    gen_arr = np.array(img_gen,      dtype=np.float32) / 255.0   # (H, W, 3)
    mask_3  = mask_np[:, :, np.newaxis]                           # (H, W, 1) broadcast
    result_arr = mask_3 * gen_arr + (1.0 - mask_3) * src_arr
    result_pil = Image.fromarray(
        np.clip(result_arr * 255.0, 0, 255).astype(np.uint8), mode="RGB"
    )

    # Save outputs
    out_dir = exp_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_pil.save(out_dir / "result.png")
    logger.info("Result saved: {}", out_dir / "result.png")

    # Side-by-side comparison panel: source | result
    W, H = source_image.size
    panel = Image.new("RGB", (W * 2, H))
    panel.paste(source_image, (0, 0))
    panel.paste(result_pil,   (W, 0))
    panel.save(exp_dir / "comparison.png")
    logger.info("Comparison panel saved: {}", exp_dir / "comparison.png")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 region-aware texture transfer.  "
            "Run with --dry-run-mask first to verify the garment mask, "
            "then without the flag for the full render."
        )
    )
    parser.add_argument(
        "--source", required=True, type=Path,
        help="Path to the source image (person wearing garment).",
    )
    parser.add_argument(
        "--ref-prompt", required=True, type=str,
        help="Text prompt describing the source garment.",
    )
    parser.add_argument(
        "--gen-prompt", required=True, type=str,
        help="Text prompt describing the desired output garment.",
    )
    parser.add_argument(
        "--mask-word", required=True, type=str,
        help="Single word in gen-prompt that identifies the garment "
             "(e.g. 'shirt').  Must appear verbatim in --gen-prompt.",
    )
    parser.add_argument(
        "--exp-name", required=True, type=str,
        help="Experiment name; artefacts saved under experiments_root/<exp-name>/.",
    )
    parser.add_argument(
        "--dry-run-mask", action="store_true",
        help=(
            "Extract and save the cross-attention mask only.  "
            "No generation is performed.  Run this first and inspect "
            "mask_overlay.png before a full render (CLAUDE.md §5.3)."
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=_DEFAULT_CONFIG,
        help=f"OmegaConf YAML config (default: {_DEFAULT_CONFIG}).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = _parse_args(argv)

    # 1. Validate inputs eagerly — fail fast before any model loading
    if not args.source.exists():
        logger.error("Source image not found: {}", args.source)
        sys.exit(1)
    if not args.config.exists():
        logger.error("Config file not found: {}", args.config)
        sys.exit(1)

    mode = "dry-run (mask only)" if args.dry_run_mask else "full render"
    logger.info("Phase 2 {} — exp: '{}'", mode, args.exp_name)
    logger.info("  source      : {}", args.source)
    logger.info("  ref-prompt  : {}", args.ref_prompt)
    logger.info("  gen-prompt  : {}", args.gen_prompt)
    logger.info("  mask-word   : {}", args.mask_word)
    logger.info("  config      : {}", args.config)

    # 2. Load config
    cfg = OmegaConf.load(args.config)
    logger.debug("Config: {}", OmegaConf.to_yaml(cfg, resolve=True))

    # 3. Create experiment directory and snapshot config (CLAUDE.md §8)
    exp_dir = Path(cfg.experiments_root) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, exp_dir / "config.yaml")
    logger.info("Config snapshot → {}", exp_dir / "config.yaml")

    # 4. Seed everything
    seed_everything(cfg.seed)
    logger.info("Seed set to {}", cfg.seed)

    # 5. Load and resize source image
    image_size = tuple(cfg.image_size)  # (W, H) per phase1 convention
    source_image = load_image(args.source, size=image_size)
    logger.info("Source image loaded and resized to {}×{}", *image_size)

    # 6. Dispatch — dry-run or full render
    if args.dry_run_mask:
        _run_dry_run_mask(
            cfg=cfg,
            source_image=source_image,
            ref_prompt=args.ref_prompt,
            gen_prompt=args.gen_prompt,
            mask_word=args.mask_word,
            exp_dir=exp_dir,
        )
    else:
        _run_full_render(
            cfg=cfg,
            source_image=source_image,
            ref_prompt=args.ref_prompt,
            gen_prompt=args.gen_prompt,
            exp_dir=exp_dir,
        )

    logger.info("All artefacts saved to {}", exp_dir)


if __name__ == "__main__":
    main()
