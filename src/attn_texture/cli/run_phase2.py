"""CLI entry point for Phase 2: region-aware garment texture transfer.

Workflow (CLAUDE.md §5.3 — mask before crop)
---------------------------------------------
**Step 1 — inspect the mask (required):**

    python -m attn_texture.cli.run_phase2 \\
        --source     path/to/person.jpg \\
        --ref-prompt "a person wearing a plain white shirt" \\
        --gen-prompt "a person wearing a striped silk shirt" \\
        --mask-word  shirt \\
        --exp-name   2026-05-25_person1_silk_stripe \\
        --dry-run-mask

    Saves mask.png (grayscale) and mask_overlay.png (red tint for inspection).
    No generation happens.

**Step 2 — crop the garment region:**

    python -m attn_texture.cli.run_phase2 \\
        --source     path/to/person.jpg ... --exp-name ...

    Requires mask.png from Step 1.
    Saves outputs/garment_crop.png — tight bounding-box crop of the masked region.
    Face and background are never touched.

**Step 3 — optionally apply Phase 1 texture transfer on the crop:**

    python -m attn_texture.cli.run_phase2 \\
        --source     path/to/person.jpg ... --exp-name ... --apply-texture

    Resizes the crop to image_size, runs TwoPassPipeline (PnP + FFT blend) on it,
    saves outputs/garment_generated.png and comparison.png.

What it does
------------
Dry-run (--dry-run-mask):
  1. Load SD 1.5; encode source → z_0.
  2. DDIM inversion (mask.dry_run_steps): z_0 → z_T.
  3. Register AttentionStore on attn2 layers; denoising loop accumulates maps.
  4. Aggregate at 16×16; build binary mask; upsample to image resolution.
  5. Save mask.png + mask_overlay.png.

Crop (no flags):
  1. Verify mask.png exists (§5.3 hard stop if absent).
  2. Compute bounding box of masked pixels.
  3. Crop source image to that box → outputs/garment_crop.png.

Crop + texture (--apply-texture):
  Same as crop, then:
  4. Resize crop to cfg.image_size, run TwoPassPipeline.
  5. Save outputs/garment_generated.png and comparison.png.
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
from attn_texture.utils.io import load_image, pil_to_tensor
from attn_texture.utils.memory import with_isolated_model
from attn_texture.utils.seed import seed_everything

_DEFAULT_CONFIG = Path(__file__).parents[3] / "configs" / "phase2_local.yaml"


# ---------------------------------------------------------------------------
# Model factory
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
# Shared helpers
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


def _ddim_inversion(unet, scheduler, z_0: torch.Tensor, ref_embeds: torch.Tensor) -> torch.Tensor:
    """Deterministic DDIM forward process: z_0 → z_T.

    scheduler.set_timesteps() must be called before this function.

    Args:
        unet:       UNet2DConditionModel.
        scheduler:  DDIMScheduler with .alphas_cumprod and .timesteps.
        z_0:        VAE-encoded source latent (1, 4, H, W).
        ref_embeds: text embeddings for the source prompt (1, S, D).

    Returns:
        Inverted latent z_T, same shape as z_0.
    """
    timesteps_fwd = scheduler.timesteps.flip(0)
    alphas: torch.Tensor = scheduler.alphas_cumprod.to(z_0.device, z_0.dtype)

    z = z_0.clone()
    for i, t in enumerate(timesteps_fwd):
        alpha_t = alphas[t]
        noise_pred = unet(z, t, encoder_hidden_states=ref_embeds).sample
        z_0_pred = (z - (1.0 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        alpha_next = alphas[timesteps_fwd[i + 1]] if i + 1 < len(timesteps_fwd) else torch.zeros_like(alpha_t)
        z = alpha_next.sqrt() * z_0_pred + (1.0 - alpha_next).sqrt() * noise_pred

    logger.debug("_ddim_inversion: complete, z_T shape={}", z.shape)
    return z


def _make_mask_overlay(source: Image.Image, mask_arr: np.ndarray) -> Image.Image:
    """Overlay a semi-transparent red tint on masked pixels of *source*.

    Args:
        source:   RGB PIL Image.
        mask_arr: uint8 array (H, W); 255 = inside mask, 0 = outside.

    Returns:
        RGBA PIL Image with red overlay inside the mask.
    """
    overlay = source.copy().convert("RGBA")
    red = Image.new("RGBA", source.size, (220, 30, 30, 130))
    overlay.paste(red, mask=Image.fromarray(mask_arr, mode="L"))
    return overlay


def _bbox_from_mask(mask_arr: np.ndarray) -> tuple[int, int, int, int]:
    """Return the tight bounding box of non-zero pixels in *mask_arr*.

    Args:
        mask_arr: uint8 (H, W) mask; non-zero pixels define the region.

    Returns:
        (x0, y0, x1, y1) inclusive pixel coordinates suitable for PIL.crop.

    Raises:
        ValueError: if the mask is entirely zero (nothing to crop).
    """
    ys, xs = np.where(mask_arr > 127)
    if len(ys) == 0:
        raise ValueError(
            "Mask is empty — no pixels above threshold. "
            "Re-run --dry-run-mask with a lower threshold in configs/phase2_local.yaml."
        )
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# ---------------------------------------------------------------------------
# Step 1 — Dry-run: mask extraction  (CLAUDE.md §5.3)
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
      mask.png         — grayscale binary mask (255 = garment, 0 = background).
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
    W_img, H_img = source_image.size

    mask_arr: np.ndarray  # populated inside with_isolated_model

    with with_isolated_model(lambda: _load_sd_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder = comps
        device, dtype = get_device_and_dtype()
        vae_scale: float = float(vae.config.scaling_factor)

        with torch.inference_mode():
            # 1. Encode source image → z_0
            src_pixels = pil_to_tensor(source_image).unsqueeze(0)
            z_0 = vae.encode(
                safe_to(src_pixels, device, dtype) * 2.0 - 1.0
            ).latent_dist.sample() * vae_scale

            # 2. Encode prompts
            ref_embeds = _encode_prompt(tokenizer, text_encoder, ref_prompt, device)
            gen_embeds = _encode_prompt(tokenizer, text_encoder, gen_prompt, device)

            # 3. DDIM inversion: z_0 → z_T
            scheduler.set_timesteps(int(mask_cfg.dry_run_steps))
            z_T = _ddim_inversion(unet, scheduler, z_0, ref_embeds)

            # 4. Register AttentionStore on attn2 layers (attn1 untouched)
            store = AttentionStore(max_spatial_side=int(mask_cfg.max_spatial_side))
            cleanup = register_attention_control(unet, store)

            # 5. Conditional denoising loop — accumulate cross-attention maps
            gen_latents = z_T.clone()
            for t in scheduler.timesteps:
                noise_pred = unet(
                    gen_latents, t,
                    encoder_hidden_states=gen_embeds,
                    return_dict=True,
                ).sample
                gen_latents = scheduler.step(noise_pred, t, gen_latents).prev_sample
                store.between_steps()

            cleanup()

        # 6. Resolve token indices (tokenizer still live)
        word_inds = get_word_inds(tokenizer, gen_prompt, mask_word)
        if not word_inds:
            logger.error(
                "Word '{}' not found as a CLIP token in gen-prompt '{}'. "
                "Check spelling — must appear verbatim.",
                mask_word, gen_prompt,
            )
            sys.exit(1)
        logger.info("Token indices for '{}': {}", mask_word, word_inds)

        # 7. Aggregate cross-attention → binary mask at image resolution
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
        mask_arr = mask_bool.cpu().numpy().astype(np.uint8) * 255

    # 8. Save mask and overlay
    Image.fromarray(mask_arr, mode="L").save(exp_dir / "mask.png")
    _make_mask_overlay(source_image, mask_arr).save(exp_dir / "mask_overlay.png")

    coverage = float(mask_arr.astype(bool).mean()) * 100.0
    logger.info("Mask saved: {}  ({:.1f}% coverage)", exp_dir / "mask.png", coverage)
    logger.info("Overlay saved: {}", exp_dir / "mask_overlay.png")
    logger.info("Inspect mask_overlay.png, then re-run without --dry-run-mask.")


# ---------------------------------------------------------------------------
# Step 1b — Gen+mask: generate source and extract mask in one forward pass
# ---------------------------------------------------------------------------


def _run_gen_and_mask(
    cfg,
    ref_prompt: str,
    mask_word: str,
    exp_dir: Path,
    save_source_path: Path,
    mask_step: int,
) -> None:
    """Generate source image and extract cross-attention mask in one forward pass.

    Implements the P2P (Hertz et al. ICLR 2023) approach: hooks cross-attention
    DURING forward generation (random noise → image), not DDIM inversion.
    Captures the single-step *conditional* attention map at `mask_step` and
    thresholds it into a binary mask.

    This avoids the 100% mask saturation that occurs when inverting SD-generated
    images: near-perfect inversion → uniform attention → no discriminative signal.

    With CFG, each UNet call processes batch=[uncond, cond].  Only the conditional
    half of the attention map (second half of the B*heads dimension) is used for
    the mask so that the unconditional stream does not dilute the semantic signal.

    Saves:
        <save_source_path>  — 512×512 RGB generated source image.
        mask.png            — grayscale binary mask from step `mask_step`.
        mask_overlay.png    — red tint overlay for inspection.

    Args:
        cfg:              Full OmegaConf config (phase2_local.yaml).
        ref_prompt:       Source prompt; `mask_word` must appear in it.
        mask_word:        Garment word to localise (e.g. "shirt").
        exp_dir:          Experiment directory for mask and overlay outputs.
        save_source_path: Where to write the generated source image.
        mask_step:        0-based denoising step index at which the conditional
                          attention is captured (P2P: early-to-mid steps carry
                          the best semantic signal).
    """
    logger.info(
        "─── Gen+mask: forward generation with attention hook at step {} ───",
        mask_step,
    )
    mask_cfg = cfg.mask
    image_size = tuple(cfg.image_size)  # (W, H)
    W_img, H_img = image_size
    guidance_scale = float(cfg.pnp.guidance_scale)
    num_steps = int(cfg.pnp.num_inference_steps)

    with with_isolated_model(lambda: _load_sd_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder = comps
        device, dtype = get_device_and_dtype()
        vae_scale: float = float(vae.config.scaling_factor)

        # Encode prompts — conditional (ref) + unconditional (empty) for CFG
        ref_embeds = _encode_prompt(tokenizer, text_encoder, ref_prompt, device)
        uncond_embeds = _encode_prompt(tokenizer, text_encoder, "", device)

        # Resolve token indices for mask_word inside ref_prompt
        word_inds = get_word_inds(tokenizer, ref_prompt, mask_word)
        if not word_inds:
            logger.error(
                "Word '{}' not found as a CLIP token in ref-prompt '{}'. "
                "Check spelling — must appear verbatim.",
                mask_word, ref_prompt,
            )
            sys.exit(1)
        logger.info("Token indices for '{}': {}", mask_word, word_inds)

        # Start from random noise — no DDIM inversion
        scheduler.set_timesteps(num_steps)
        z = torch.randn(1, 4, H_img // 8, W_img // 8, device=device, dtype=dtype)
        z = z * scheduler.init_noise_sigma

        # Hook AttentionStore on attn2 layers
        store = AttentionStore(max_spatial_side=int(mask_cfg.max_spatial_side))
        cleanup = register_attention_control(unet, store)

        # CFG denoising loop — capture conditional attention at mask_step
        step_attention: dict[str, list[torch.Tensor]] = {}
        with torch.inference_mode():
            for i, t in enumerate(scheduler.timesteps):
                # Duplicate latent for CFG batch [uncond, cond]
                latent_input = torch.cat([z, z])
                emb_input = torch.cat([uncond_embeds, ref_embeds])
                noise_pred_both = unet(
                    latent_input, t, encoder_hidden_states=emb_input
                ).sample
                noise_pred_uncond, noise_pred_cond = noise_pred_both.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                z = scheduler.step(noise_pred, t, z).prev_sample

                if i == mask_step:
                    # _step_maps: each tensor shape (2*heads, N_spatial, N_tokens)
                    # Take conditional half (second half of B*heads dim)
                    step_attention = {
                        p: [m[m.shape[0] // 2 :].clone() for m in store._step_maps[p]]
                        for p in AttentionStore._PLACES
                    }
                    logger.info(
                        "Captured conditional attention at step {}/{} (scheduler t={})",
                        i, num_steps - 1, int(t.item()),
                    )

                store.between_steps()

        cleanup()

        # Decode latent → PIL image
        with torch.inference_mode():
            image_tensor = vae.decode(z / vae_scale).sample
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_np = (
            image_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255
        ).astype(np.uint8)
        generated_image = Image.fromarray(image_np)

    # Save generated source image
    save_source_path.parent.mkdir(parents=True, exist_ok=True)
    generated_image.save(save_source_path)
    logger.info("Generated source image saved: {}", save_source_path)

    # Build mask from single-step conditional attention
    if not step_attention or all(len(v) == 0 for v in step_attention.values()):
        logger.error(
            "No attention maps captured — mask_step ({}) may exceed "
            "num_inference_steps ({}).",
            mask_step, num_steps,
        )
        sys.exit(1)

    agg = aggregate_attention(
        step_attention,
        res=int(mask_cfg.aggregate_res),
        places=tuple(mask_cfg.aggregate_places),
    )
    mask_bool = build_mask(
        agg,
        word_inds=word_inds,
        threshold=float(mask_cfg.threshold),
        target_hw=(H_img, W_img),
    )
    mask_arr = mask_bool.cpu().numpy().astype(np.uint8) * 255

    Image.fromarray(mask_arr, mode="L").save(exp_dir / "mask.png")
    _make_mask_overlay(generated_image, mask_arr).save(exp_dir / "mask_overlay.png")

    coverage = float(mask_arr.astype(bool).mean()) * 100.0
    logger.info("Mask saved: {}  ({:.1f}% coverage)", exp_dir / "mask.png", coverage)
    logger.info("Overlay saved: {}", exp_dir / "mask_overlay.png")
    logger.info("Inspect mask_overlay.png, then re-run without --dry-run-mask.")


# ---------------------------------------------------------------------------
# Step 2 — Crop garment from bounding box  (CLAUDE.md §5.3)
# ---------------------------------------------------------------------------


def _run_crop(
    cfg,
    source_image: Image.Image,
    ref_prompt: str,
    gen_prompt: str,
    exp_dir: Path,
    apply_texture: bool,
) -> None:
    """Crop the masked garment region and optionally apply Phase 1 texture.

    CLAUDE.md §5.3: hard-stops if mask.png is absent — dry-run must come first.
    The source image, face, and background are never modified.

    Step 2 outputs (always):
      outputs/garment_crop.png  — tight bounding-box crop of the source garment.

    Step 3 outputs (--apply-texture only):
      outputs/garment_generated.png — Phase 1 result on the resized crop.
      comparison.png                — side-by-side source crop | generated.

    Args:
        cfg:           Full OmegaConf config (phase2_local.yaml).
        source_image:  RGB PIL Image already resized to cfg.image_size.
        ref_prompt:    Source description prompt for Phase 1.
        gen_prompt:    Target description prompt for Phase 1.
        exp_dir:       Experiment directory; mask.png must exist here.
        apply_texture: If True, run TwoPassPipeline on the crop.
    """
    # §5.3 gate
    mask_path = exp_dir / "mask.png"
    if not mask_path.exists():
        logger.error(
            "mask.png not found at '{}'. Run --dry-run-mask first (CLAUDE.md §5.3).",
            mask_path,
        )
        sys.exit(1)

    mask_arr = np.array(Image.open(mask_path).convert("L"))   # uint8 (H, W)
    coverage = float((mask_arr > 127).mean()) * 100.0
    logger.info("Loaded mask from '{}'  ({:.1f}% coverage)", mask_path, coverage)

    # Bounding box crop — face and background untouched
    try:
        x0, y0, x1, y1 = _bbox_from_mask(mask_arr)
    except ValueError as exc:
        logger.error("{}", exc)
        sys.exit(1)

    logger.info("Garment bounding box: ({}, {}) → ({}, {})  [{}×{}px]",
                x0, y0, x1, y1, x1 - x0, y1 - y0)

    crop = source_image.crop((x0, y0, x1, y1))

    out_dir = exp_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    crop.save(out_dir / "garment_crop.png")
    logger.info("Garment crop saved: {}", out_dir / "garment_crop.png")

    if not apply_texture:
        return

    # ── Phase 1 texture transfer on the crop ────────────────────────────────
    logger.info("─── Applying Phase 1 texture transfer on crop ───")

    image_size = tuple(cfg.image_size)  # (W, H)
    crop_resized = crop.resize(image_size, Image.LANCZOS)
    logger.info("Crop resized to {}×{} for Phase 1", *image_size)

    with with_isolated_model(lambda: _load_sd_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder = comps
        pipeline = TwoPassPipeline(unet, vae, scheduler, tokenizer, text_encoder, cfg.pnp)
        img_gen = pipeline.run(
            source_image=crop_resized,
            ref_prompt=ref_prompt,
            gen_prompt=gen_prompt,
            num_inference_steps=int(cfg.pnp.num_inference_steps),
        )
    logger.info("Phase 1 done.")

    img_gen.save(out_dir / "garment_generated.png")
    logger.info("Generated garment saved: {}", out_dir / "garment_generated.png")

    # Side-by-side comparison: source crop (resized) | generated
    W, H = image_size
    panel = Image.new("RGB", (W * 2, H))
    panel.paste(crop_resized, (0, 0))
    panel.paste(img_gen,      (W, 0))
    panel.save(exp_dir / "comparison.png")
    logger.info("Comparison panel saved: {}", exp_dir / "comparison.png")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 2 garment texture transfer. "
            "Run --dry-run-mask first to verify the mask, then crop (and optionally "
            "texture-transfer) without the flag."
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
        help="Word in gen-prompt identifying the garment (e.g. 'shirt').",
    )
    parser.add_argument(
        "--exp-name", required=True, type=str,
        help="Experiment name; artefacts saved under experiments_root/<exp-name>/.",
    )
    parser.add_argument(
        "--dry-run-mask", action="store_true",
        help=(
            "Extract and save the cross-attention mask only. "
            "Run this first and inspect mask_overlay.png (CLAUDE.md §5.3)."
        ),
    )
    parser.add_argument(
        "--apply-texture", action="store_true",
        help=(
            "After cropping, run Phase 1 TwoPassPipeline on the garment crop. "
            "Ignored when --dry-run-mask is set."
        ),
    )
    parser.add_argument(
        "--gen-source", action="store_true",
        help=(
            "Generate source image from ref-prompt during --dry-run-mask, hooking "
            "cross-attention DURING forward generation (P2P approach) rather than "
            "DDIM inversion. Saves the generated image to --source path. "
            "Requires --dry-run-mask."
        ),
    )
    parser.add_argument(
        "--mask-step", type=int, default=20,
        help=(
            "0-based denoising step index at which to capture the conditional "
            "attention map for the mask. Used with --gen-source. Default: 20."
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

    if args.gen_source and not args.dry_run_mask:
        logger.error("--gen-source requires --dry-run-mask.")
        sys.exit(1)
    if not args.gen_source and not args.source.exists():
        logger.error("Source image not found: {}", args.source)
        sys.exit(1)
    if not args.config.exists():
        logger.error("Config file not found: {}", args.config)
        sys.exit(1)

    if args.gen_source:
        mode = f"gen+mask (step {args.mask_step})"
    elif args.dry_run_mask:
        mode = "dry-run (mask only)"
    elif args.apply_texture:
        mode = "crop + texture transfer"
    else:
        mode = "crop only"

    logger.info("Phase 2 {} — exp: '{}'", mode, args.exp_name)
    logger.info("  source      : {}", args.source)
    logger.info("  ref-prompt  : {}", args.ref_prompt)
    logger.info("  gen-prompt  : {}", args.gen_prompt)
    logger.info("  mask-word   : {}", args.mask_word)
    logger.info("  config      : {}", args.config)

    cfg = OmegaConf.load(args.config)

    exp_dir = Path(cfg.experiments_root) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, exp_dir / "config.yaml")

    seed_everything(cfg.seed)

    image_size = tuple(cfg.image_size)

    if args.gen_source:
        # Generate source image and extract mask in one forward pass (P2P approach)
        _run_gen_and_mask(
            cfg=cfg,
            ref_prompt=args.ref_prompt,
            mask_word=args.mask_word,
            exp_dir=exp_dir,
            save_source_path=args.source,
            mask_step=args.mask_step,
        )
    elif args.dry_run_mask:
        source_image = load_image(args.source, size=image_size)
        logger.info("Source image loaded and resized to {}×{}", *image_size)
        _run_dry_run_mask(
            cfg=cfg,
            source_image=source_image,
            ref_prompt=args.ref_prompt,
            gen_prompt=args.gen_prompt,
            mask_word=args.mask_word,
            exp_dir=exp_dir,
        )
    else:
        source_image = load_image(args.source, size=image_size)
        logger.info("Source image loaded and resized to {}×{}", *image_size)
        _run_crop(
            cfg=cfg,
            source_image=source_image,
            ref_prompt=args.ref_prompt,
            gen_prompt=args.gen_prompt,
            exp_dir=exp_dir,
            apply_texture=args.apply_texture,
        )

    logger.info("All artefacts saved to {}", exp_dir)


if __name__ == "__main__":
    main()
