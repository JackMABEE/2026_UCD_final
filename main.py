"""
main.py — Entry point for the attention-guided img2img research pipeline.

Usage
-----
    python main.py --config configs/experiment_01.yaml
    python main.py --config configs/experiment_01.yaml --source assets/cat.png
    python main.py --config configs/experiment_01.yaml --batch assets/img1.png assets/img2.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from PIL import Image


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_inputs(cfg: Dict[str, Any], cli_source: str | None, cli_batch: List[str]) -> List[Path]:
    """Determine the list of source images from CLI args and config."""
    if cli_batch:
        return [Path(p) for p in cli_batch]
    if cli_source:
        return [Path(cli_source)]
    if cfg.get("batch_inputs"):
        return [Path(p) for p in cfg["batch_inputs"]]
    return [Path(cfg["source_image_path"])]


def setup_pipeline(cfg: Dict[str, Any]):
    """Import here to keep startup fast when just inspecting flags."""
    from pipeline_custom import CustomImg2ImgPipeline

    dtype = torch.float16 if cfg.get("torch_dtype", "float16") == "float16" else torch.float32
    device = cfg.get("device", "cuda")

    pipe = CustomImg2ImgPipeline.from_pretrained(
        cfg["model_id"],
        torch_dtype=dtype,
        device=device,
    )
    pipe.setup_control(
        attn_cfg=cfg.get("attention_control", {"enabled": False}),
        freq_cfg=cfg.get("frequency_filter", {"enabled": False}),
    )
    return pipe


def run_single(
    pipe,
    source_path: Path,
    cfg: Dict[str, Any],
    out_dir: Path,
    metrics_evaluator,
) -> Dict[str, Any]:
    """Run the pipeline on a single source image and return a metrics dict."""
    source_img = Image.open(source_path).convert("RGB")

    t0 = time.perf_counter()
    result = pipe.run(
        source_image=source_img,
        prompt=cfg["prompt"],
        negative_prompt=cfg.get("negative_prompt", ""),
        num_inference_steps=cfg["num_inference_steps"],
        strength=cfg.get("strength", 0.75),
        guidance_scale=cfg.get("guidance_scale", 7.5),
        seed=cfg.get("seed", None),
    )
    elapsed = time.perf_counter() - t0

    gen_image: Image.Image = result["image"]
    ref_image: Image.Image = result["reference_image"]

    # Save outputs.
    stem = source_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_path = out_dir / f"{stem}_generated.png"
    ref_path = out_dir / f"{stem}_reference.png"
    gen_image.save(gen_path)
    ref_image.save(ref_path)
    print(f"  Saved: {gen_path}")

    entry: Dict[str, Any] = {
        "source": str(source_path),
        "generated": str(gen_path),
        "reference": str(ref_path),
        "elapsed_s": round(elapsed, 2),
    }

    # Compute metrics.
    if metrics_evaluator is not None:
        try:
            scores = metrics_evaluator.compute_all(source_img, gen_image)
            entry["metrics"] = scores
            print(
                f"  SSIM={scores['ssim']:.4f}  "
                f"PSNR={scores['psnr']:.2f}dB  "
                f"LPIPS={scores['lpips']:.4f}"
            )
        except Exception as exc:
            entry["metrics_error"] = str(exc)
            print(f"  [WARNING] Metrics failed: {exc}", file=sys.stderr)

    return entry


def main():
    parser = argparse.ArgumentParser(description="Attention-guided img2img pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--source", default=None, help="Override source_image_path from config")
    parser.add_argument("--batch", nargs="*", default=[], help="Override batch_inputs from config")
    parser.add_argument("--output_dir", default=None, help="Override output_dir from config")
    parser.add_argument("--no_metrics", action="store_true", help="Skip metric computation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    out_dir = Path(cfg["output_dir"])
    source_paths = resolve_inputs(cfg, args.source, args.batch)

    print(f"Experiment : {cfg.get('experiment_name', 'unnamed')}")
    print(f"Output dir : {out_dir}")
    print(f"Sources    : {[str(p) for p in source_paths]}")
    print()

    # Load pipeline.
    print("Loading pipeline…")
    pipe = setup_pipeline(cfg)

    # Load metrics evaluator.
    metrics_evaluator = None
    if cfg.get("metrics", {}).get("enabled", True) and not args.no_metrics:
        from utils.metrics import ImageMetrics
        device = cfg.get("device", "cpu")
        # LPIPS on CPU is fine for evaluation; move to device if GPU is free.
        metrics_evaluator = ImageMetrics(
            lpips_net=cfg.get("metrics", {}).get("lpips_net", "alex"),
            device=device,
        )

    # Process each source.
    report: List[Dict[str, Any]] = []
    for i, src in enumerate(source_paths, 1):
        print(f"[{i}/{len(source_paths)}] Processing {src}…")
        if not src.exists():
            print(f"  [SKIP] File not found: {src}", file=sys.stderr)
            continue
        entry = run_single(pipe, src, cfg, out_dir, metrics_evaluator)
        report.append(entry)
        print()

    # Save JSON report.
    if cfg.get("metrics", {}).get("save_report", True) and not args.no_metrics:
        report_name = cfg.get("metrics", {}).get("report_filename", "metrics_report.json")
        report_path = out_dir / report_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump({"experiment": cfg.get("experiment_name"), "results": report}, f, indent=2)
        print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
