# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis research framework for **attention-guided image-to-image generation** using Stable Diffusion. The core idea: run two denoising passes — a reference pass that *captures* self-attention K/Q tensors from the U-Net, then a target pass that *injects* those tensors while optionally filtering latents in the frequency domain.

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Run an experiment
```bash
python main.py --config configs/experiment_01.yaml
# Override source image
python main.py --config configs/experiment_01.yaml --source assets/photo.png
# Batch mode
python main.py --config configs/experiment_01.yaml --batch assets/img1.png assets/img2.png
# Skip metrics (faster iteration)
python main.py --config configs/experiment_01.yaml --no_metrics
```

## Architecture

```
main.py                   Entry point — parses YAML, loops over inputs, saves JSON report
pipeline_custom.py        CustomImg2ImgPipeline — orchestrates the two-pass denoising loop
configs/experiment_01.yaml  All hyperparameters (model, prompt, attention, freq filter, metrics)

models/
  attention_control.py   AttentionStore (capture K/Q) + AttentionInjector (replace/blend K/Q)

utils/
  frequency_filter.py    FrequencyFilter — differentiable FFT low/high/hybrid pass on latents
  metrics.py             ImageMetrics — SSIM, PSNR (skimage), LPIPS (lpips package)
```

### Two-pass pipeline flow (`pipeline_custom.py`)

1. **Pass A (reference):** `AttentionStore` hooks capture K/Q from every `Attention` block in the U-Net at each denoising step. Output is saved as `*_reference.png`.
2. **Pass B (target):** `AttentionInjector` pre-hooks replace the live K/Q tensors with stored ones for steps in `injection_steps`. `FrequencyFilter` is applied to the latents in the step callback. Output is saved as `*_generated.png`.

### Attention hook mechanics (`models/attention_control.py`)

- Hooks attach to `diffusers.models.attention_processor.Attention` modules.
- `AttentionStore` uses a **forward hook** to re-project `hidden_states` through `to_k`/`to_q` and stores the result (shape `(B, heads, seq, head_dim)`).
- `AttentionInjector` uses a **forward pre-hook** that monkey-patches `module.to_k` / `module.to_q` for a single forward call, then restores originals via a self-removing post-hook.
- `target_layers` config accepts substrings matched against `unet.named_modules()` (e.g. `["up_blocks"]`).

### Frequency filter (`utils/frequency_filter.py`)

- Uses `torch.fft.rfft2` / `irfft2` with `norm="ortho"`.
- The low-pass mask is a circular boolean mask in the DC-centered frequency plane, with radius = `low_pass_radius * 0.5` (where 0.5 is the Nyquist limit).
- Mask is cached per spatial shape to avoid recomputation.
- `decompose(x)` returns `(low, high)` tensors that sum to `x` — useful for ablations.

### Metrics (`utils/metrics.py`)

- `ImageMetrics.compute_all(source, generated)` returns `{"ssim": float, "psnr": float, "lpips": float}`.
- Accepts PIL images, CHW float tensors, or HWC numpy arrays.
- LPIPS failure (missing package) degrades gracefully to `NaN` with a warning.

## Key Config Fields

| Field | Effect |
|---|---|
| `attention_control.injection_threshold` | Fraction of steps below which injection is active (informational — use `injection_steps` list directly) |
| `attention_control.injection_mode` | `"replace"` or `"blend"` (blend uses `blend_alpha`) |
| `frequency_filter.mode` | `"low"` / `"high"` / `"hybrid"` |
| `frequency_filter.low_pass_radius` | Fraction of spatial dim for LP mask radius (0–1) |
| `strength` | img2img noise strength — lower = closer to source |
| `seed` | Set for reproducibility; `null` for random |

## diffusers Version Compatibility

`pipeline_custom.py` tries the new `callback_on_step_end` API (diffusers ≥ 0.21) and falls back to the old `callback` API. If latent modification via the frequency filter stops working, check which API version is active.
