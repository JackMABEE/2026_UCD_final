"""5-way comparison panel and metrics aggregation for Phase 1 shootout.

Accepts five PIL images (original, sdedit, controlnet, pnp_baseline, ours),
stitches them into a side-by-side panel with burned-in labels, computes
SSIM / PSNR / LPIPS / CLIP / DINO for each method vs. the original, and writes
two artefacts to the experiment directory:

  experiments/<exp_name>/shootout.png   — 1×5 RGB panel
  experiments/<exp_name>/metrics.json   — nested dict, one entry per method

Domain naming follows CLAUDE.md §4:
  method keys: "sdedit", "controlnet", "pnp_baseline", "ours"
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

from attn_texture.eval.metrics import clip_score, dino_distance, lpips_score, psnr, ssim

_METHODS: tuple[str, ...] = ("original", "sdedit", "controlnet", "pnp_baseline", "ours")
_LABEL_HEIGHT: int = 24  # pixel rows reserved for the text banner above each image
_LABEL_FILL: tuple[int, int, int] = (255, 255, 255)
_LABEL_BG: tuple[int, int, int] = (30, 30, 30)
_TEXT_COLOUR: tuple[int, int, int] = (255, 255, 255)


def run_shootout(
    exp_name: str,
    original: Image.Image,
    sdedit: Image.Image,
    controlnet: Image.Image,
    pnp_baseline: Image.Image,
    ours: Image.Image,
    gen_prompt: str,
    experiments_root: Path = Path("experiments"),
) -> dict[str, dict[str, float]]:
    """Run the 5-way shootout: build panel, compute metrics, save artefacts.

    Args:
        exp_name:          Experiment identifier, used as the subdirectory name
                           under *experiments_root* (may contain path separators
                           for nested layouts, e.g. "2026-05-19_silk_floral").
        original:          Source / reference image.
        sdedit:            SDEdit baseline output.
        controlnet:        ControlNet baseline output.
        pnp_baseline:      PnP attention-injection only (fft_cutoff_ratio=0).
        ours:              PnP + LAB FFT blend (full method).
        gen_prompt:        Target text prompt used during generation; passed to
                           clip_score to measure image-text alignment.
        experiments_root:  Root directory for all experiment artefacts.
                           Defaults to ``experiments/`` in the working directory.

    Returns:
        Nested dict of the form::

            {"sdedit":        {"ssim": …, "psnr": …, "lpips": …, "clip": …, "dino": …},
             "controlnet":    {…},
             "pnp_baseline":  {…},
             "ours":          {…}}

    Side effects:
        Writes ``shootout.png`` and ``metrics.json`` to
        ``experiments_root / exp_name``.
    """
    images = [original, sdedit, controlnet, pnp_baseline, ours]

    # 1. Build and save the panel
    panel = _build_panel(images, list(_METHODS))
    exp_dir = Path(experiments_root) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    panel_path = exp_dir / "shootout.png"
    panel.save(panel_path)
    logger.info("Saved panel → {}", panel_path)

    # 2. Compute metrics for each non-original method vs. original
    method_images = {
        "sdedit": sdedit,
        "controlnet": controlnet,
        "pnp_baseline": pnp_baseline,
        "ours": ours,
    }
    metrics: dict[str, dict[str, float]] = {}
    for method, img in method_images.items():
        s = ssim(original, img)
        p = psnr(original, img)
        lp = lpips_score(original, img)
        cl = clip_score(img, gen_prompt)
        di = dino_distance(original, img)
        metrics[method] = {
            "ssim": float(s),
            "psnr": _finite_or_cap(p),
            "lpips": float(lp),
            "clip": float(cl),
            "dino": float(di),
        }
        logger.debug(
            "{}: ssim={:.4f} psnr={:.2f} lpips={:.4f} clip={:.4f} dino={:.4f}",
            method, s, p, lp, cl, di,
        )

    # 3. Save metrics JSON
    json_path = exp_dir / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved metrics → {}", json_path)

    return metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _finite_or_cap(value: float, cap: float = 100.0) -> float:
    """Replace +inf PSNR (identical images) with a capped sentinel for JSON."""
    return cap if math.isinf(value) else float(value)


def _build_panel(images: list[Image.Image], labels: list[str]) -> Image.Image:
    """Stitch *images* side-by-side with *labels* burned into a top banner.

    Args:
        images: list of N RGB PIL images, all the same size.
        labels: list of N label strings, one per image.

    Returns:
        RGB PIL Image of width = N × W, height = _LABEL_HEIGHT + H.
    """
    n = len(images)
    W, H = images[0].size
    panel_w = n * W
    panel_h = _LABEL_HEIGHT + H

    panel = Image.new("RGB", (panel_w, panel_h), color=_LABEL_BG)
    draw = ImageDraw.Draw(panel)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        x_off = i * W
        # Paste image content below the label banner
        panel.paste(img.convert("RGB"), (x_off, _LABEL_HEIGHT))
        # Draw label centred in the banner cell
        draw.text((x_off + 4, 4), label, fill=_TEXT_COLOUR, font=font)

    return panel
