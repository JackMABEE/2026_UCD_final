"""CLI entry point for Phase 1: full-frame texture transfer shootout.

Usage
-----
python -m attn_texture.cli.run_phase1 \\
    --source  path/to/source.jpg \\
    --ref-prompt   "a plain silk fabric" \\
    --gen-prompt   "a floral silk fabric" \\
    --exp-name     "2026-05-19_silk_floral_seed42" \\
    [--config      configs/phase1_global.yaml]

What it does
------------
1. Load configs/phase1_global.yaml (or --config path).
2. Seed everything via utils/seed.py.
3. Load source image, resize to cfg.image_size.
4. Run our PnP method (TwoPassPipeline) inside with_isolated_model.
5. Run SDEdit baseline inside with_isolated_model.
6. Run ControlNet baseline inside with_isolated_model.
   (Models are never co-resident in memory — CLAUDE.md §5.2.)
7. Call run_shootout() to save shootout.png and metrics.json under
   cfg.experiments_root / exp_name.

Note: MasaCtrl (_run_masactrl) is available in this file but excluded from
the default shootout pending per-task parameter tuning.

Model weights are loaded from HuggingFace hub via model_id / controlnet_id
in the config.  On MPS, fp16 is avoided automatically by utils/device.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Internal imports (never hardcode device / dtype — go through utils)
# ---------------------------------------------------------------------------
from attn_texture.utils.device import get_device_and_dtype
from attn_texture.utils.io import load_image
from attn_texture.utils.memory import with_isolated_model
from attn_texture.utils.seed import seed_everything

from attn_texture.core.two_pass_pipeline import TwoPassPipeline
from attn_texture.baselines.sdedit_runner import SDEditRunner
from attn_texture.baselines.controlnet_runner import ControlNetRunner
from attn_texture.baselines.masactrl_runner import MasaCtrlRunner
from attn_texture.eval.shootout import run_shootout

_DEFAULT_CONFIG = Path(__file__).parents[3] / "configs" / "phase1_global.yaml"


# ---------------------------------------------------------------------------
# Model factories — called inside with_isolated_model, so each block
# loads, infers, then frees GPU/MPS memory before the next model loads.
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

    # Replace the default scheduler with DDIM (MPS-compatible; UniPC is not).
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return pipe.unet, pipe.vae, scheduler, pipe.tokenizer, pipe.text_encoder


def _load_controlnet_components(cfg):
    """Load SD 1.5 + ControlNet (Canny) as a single bundle."""
    from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionPipeline

    logger.info(
        "Loading SD 1.5 + ControlNet from '{}' / '{}'…",
        cfg.model_id,
        cfg.controlnet_id,
    )
    device, dtype = get_device_and_dtype()

    controlnet = ControlNetModel.from_pretrained(cfg.controlnet_id, torch_dtype=dtype).to(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.model_id,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    return pipe.unet, pipe.vae, scheduler, pipe.tokenizer, pipe.text_encoder, controlnet


# ---------------------------------------------------------------------------
# Per-method runners
# ---------------------------------------------------------------------------


def _run_pnp_baseline(cfg, source_image, ref_prompt: str, gen_prompt: str):
    """PnP attention injection only — no FFT blend (fft_cutoff_ratio forced to 0)."""
    logger.info("─── Running: pnp_baseline (TwoPassPipeline, no FFT) ───")
    cfg_no_fft = OmegaConf.merge(cfg.ours, OmegaConf.create({"fft_cutoff_ratio": 0.0}))
    with with_isolated_model(lambda: _load_sd_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder = comps
        pipeline = TwoPassPipeline(unet, vae, scheduler, tokenizer, text_encoder, cfg_no_fft)
        result = pipeline.run(
            source_image=source_image,
            ref_prompt=ref_prompt,
            gen_prompt=gen_prompt,
            num_inference_steps=cfg.ours.num_inference_steps,
        )
    logger.info("pnp_baseline: done.")
    return result


def _run_ours(cfg, source_image, ref_prompt: str, gen_prompt: str):
    logger.info("─── Running: ours (TwoPassPipeline) ───")
    with with_isolated_model(lambda: _load_sd_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder = comps
        pipeline = TwoPassPipeline(unet, vae, scheduler, tokenizer, text_encoder, cfg.ours)
        result = pipeline.run(
            source_image=source_image,
            ref_prompt=ref_prompt,
            gen_prompt=gen_prompt,
            num_inference_steps=cfg.ours.num_inference_steps,
        )
    logger.info("ours: done.")
    return result


def _run_sdedit(cfg, source_image, gen_prompt: str):
    logger.info("─── Running: SDEdit baseline ───")
    with with_isolated_model(lambda: _load_sd_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder = comps
        runner = SDEditRunner(unet, vae, scheduler, tokenizer, text_encoder, cfg.sdedit)
        result = runner.run(
            source_image=source_image,
            prompt=gen_prompt,
            strength=cfg.sdedit.strength,
            num_inference_steps=cfg.sdedit.num_inference_steps,
        )
    logger.info("sdedit: done.")
    return result


def _run_controlnet(cfg, source_image, gen_prompt: str):
    logger.info("─── Running: ControlNet baseline ───")
    with with_isolated_model(lambda: _load_controlnet_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder, controlnet = comps
        runner = ControlNetRunner(
            unet, vae, scheduler, tokenizer, text_encoder, controlnet, cfg.controlnet
        )
        result = runner.run(
            source_image=source_image,
            prompt=gen_prompt,
            num_inference_steps=cfg.controlnet.num_inference_steps,
        )
    logger.info("controlnet: done.")
    return result


def _run_masactrl(cfg, source_image, ref_prompt: str, gen_prompt: str):
    logger.info("─── Running: MasaCtrl baseline ───")
    with with_isolated_model(lambda: _load_sd_components(cfg)) as comps:
        unet, vae, scheduler, tokenizer, text_encoder = comps
        runner = MasaCtrlRunner(unet, vae, scheduler, tokenizer, text_encoder, cfg.masactrl)
        result = runner.run(
            source_image=source_image,
            ref_prompt=ref_prompt,
            gen_prompt=gen_prompt,
            num_inference_steps=cfg.masactrl.num_inference_steps,
        )
    logger.info("masactrl: done.")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 1 texture-transfer shootout (PnP vs SDEdit vs ControlNet)."
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to the source texture image.",
    )
    parser.add_argument(
        "--ref-prompt",
        required=True,
        type=str,
        help="Text prompt describing the source image (used for the ref pass).",
    )
    parser.add_argument(
        "--gen-prompt",
        required=True,
        type=str,
        help="Text prompt describing the desired output texture.",
    )
    parser.add_argument(
        "--exp-name",
        required=True,
        type=str,
        help="Experiment name; artefacts are saved under experiments_root/<exp-name>/.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"Path to OmegaConf YAML config (default: {_DEFAULT_CONFIG}).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    # 1. Validate inputs eagerly — fail fast before any model loading
    if not args.source.exists():
        logger.error("Source image not found: {}", args.source)
        sys.exit(1)
    if not args.config.exists():
        logger.error("Config file not found: {}", args.config)
        sys.exit(1)

    logger.info("Phase 1 shootout — exp: '{}'", args.exp_name)
    logger.info("  source     : {}", args.source)
    logger.info("  ref-prompt : {}", args.ref_prompt)
    logger.info("  gen-prompt : {}", args.gen_prompt)
    logger.info("  config     : {}", args.config)

    # 2. Load config
    cfg = OmegaConf.load(args.config)
    logger.debug("Config loaded: {}", OmegaConf.to_yaml(cfg, resolve=True))

    # 2a. Snapshot config for reproducibility (CLAUDE.md §8)
    exp_dir = Path(cfg.experiments_root) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = exp_dir / "config.yaml"
    OmegaConf.save(cfg, config_snapshot)
    logger.info("Config snapshot → {}", config_snapshot)

    # 3. Seed everything
    seed_everything(cfg.seed)
    logger.info("Seed set to {}", cfg.seed)

    # 4. Load and resize source image
    image_size = tuple(cfg.image_size)  # [W, H]
    source_image = load_image(args.source, size=image_size)
    logger.info("Source image loaded and resized to {}×{}", *image_size)

    # 5. Run each method serially — memory freed between each (CLAUDE.md §5.2)
    img_ours = _run_ours(cfg, source_image, args.ref_prompt, args.gen_prompt)
    img_pnp_baseline = _run_pnp_baseline(cfg, source_image, args.ref_prompt, args.gen_prompt)
    img_sdedit = _run_sdedit(cfg, source_image, args.gen_prompt)
    img_controlnet = _run_controlnet(cfg, source_image, args.gen_prompt)

    # 6. Assemble shootout panel and save metrics
    logger.info("─── Running: shootout panel + metrics ───")
    metrics = run_shootout(
        exp_name=args.exp_name,
        original=source_image,
        sdedit=img_sdedit,
        controlnet=img_controlnet,
        pnp_baseline=img_pnp_baseline,
        ours=img_ours,
        gen_prompt=args.gen_prompt,
        experiments_root=Path(cfg.experiments_root),
    )

    # 7. Print summary to stdout
    logger.info("Metrics summary:")
    for method, scores in metrics.items():
        logger.info(
            "  {:12s}  SSIM={:.4f}  PSNR={:.2f}dB  LPIPS={:.4f}  CLIP={:.4f}  DINO={:.4f}",
            method,
            scores["ssim"],
            scores["psnr"],
            scores["lpips"],
            scores["clip"],
            scores["dino"],
        )

    exp_dir = Path(cfg.experiments_root) / args.exp_name
    logger.info("All artefacts saved to {}", exp_dir)


if __name__ == "__main__":
    main()
