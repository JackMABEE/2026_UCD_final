# CLAUDE.md

Guidance for AI collaborators (Claude / Copilot) working in this repository.
**Reject any suggestion that conflicts with the rules below.**

---

## 1. Project Context

Master's thesis codebase: **Attention-Guided Image-to-Image Texture Transfer**.

Two algorithmic contributions:
1. **Self-Attention Injection** — Two-pass inference. Pass A extracts self-attention K/V from the source; Pass B injects them into the generation stream so new textures follow the original garment's folds.
2. **Dual-Domain FFT Blending** — In the Fourier domain, fuse the source's low frequencies (lighting) with the generated image's high frequencies (texture).

Two-phase research plan:
- **Phase 1 (current focus)**: full-frame texture images, no masking. Prove superiority against SDEdit / ControlNet via a 4-way shootout.
- **Phase 2**: real-person try-on. Extract a **cross-attention mask** from a text prompt (e.g. "shirt") and apply Phase 1 only inside that region.

---

## 2. Tech Stack

| Category | Choice | Notes |
|---|---|---|
| Language | Python ≥ 3.10 | type hints required |
| DL framework | PyTorch ≥ 2.1 | MPS + CUDA dual-backend mandatory |
| Diffusion | `diffusers` (HuggingFace) | SD 1.5 / SDXL — hookable attention layers |
| Text | `transformers` | CLIP tokenizer + text encoder |
| Baselines | `diffusers` SDEdit pipeline; `controlnet-aux` + ControlNet (Canny/HED) | comparison only |
| Math | `torch.fft` preferred; `numpy` / `scipy` only for offline eval |
| Imaging | `Pillow`, `opencv-python` |
| Metrics | `scikit-image` (SSIM/PSNR), `lpips`, `torchmetrics` |
| Testing | `pytest`, `pytest-cov` | TDD enforced |
| Config | `OmegaConf` or `pydantic` | no hardcoded hyperparams |
| Logging | `loguru` | no `print` |
| Quality | `ruff` + `black` + `mypy` | CI must pass |

**No new heavy deps** (`accelerate`, `bitsandbytes`, `xformers`, …) without discussion.

---

## 3. Directory Layout

```
.
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── configs/                      # all tunable params — YAML
│   ├── base.yaml                 # device, dtype, seed
│   ├── phase1_global.yaml
│   ├── phase2_local.yaml
│   └── shootout.yaml
├── src/attn_texture/             # main package
│   ├── core/                     # paper contributions live here
│   │   ├── attention_injection.py
│   │   ├── fft_blend.py
│   │   ├── two_pass_pipeline.py
│   │   └── mask_extraction.py    # Phase 2
│   ├── baselines/                # SDEdit / ControlNet runners
│   ├── eval/                     # metrics.py, shootout.py
│   ├── utils/                    # device, memory, tokenizer, io, seed
│   └── cli/                      # run_phase1.py, run_phase2.py, run_shootout.py
├── tests/
│   ├── unit/
│   │   ├── test_tokenizer_edge.py
│   │   ├── test_mask_extraction.py
│   │   ├── test_fft_blend.py
│   │   └── test_attn_injection_gate.py
│   ├── integration/test_two_pass_smoke.py
│   └── fixtures/                 # tiny images (≤ 64×64)
├── experiments/YYYY-MM-DD_<name>/
│   ├── config.yaml               # snapshot
│   ├── inputs/  outputs/
│   ├── shootout.png
│   └── metrics.json
├── assets/  docs/  scripts/
```

**Hard rule:** `core/` and `baselines/` must not import each other. Baselines are competitors, not collaborators.

---

## 4. Naming Conventions

**Python**: `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE` constants.

**Domain terms (paper ↔ code, must match exactly):**

| Paper | Code |
|---|---|
| Reference pass / Pass A | `ref_pass`, `ref_*` |
| Generation pass / Pass B | `gen_pass`, `gen_*` |
| Self-attention injection | `attn_injection`, `inject_kv` (never `attn_swap` / `replace`) |
| Low / high frequency | `low_freq`, `high_freq` (no `lf`/`hf` abbreviations) |
| Cross-attention mask | `cross_attn_mask` |
| 4-way comparison | `shootout` |

**Experiments**: `experiments/2026-05-19_phase1_silk_floral/`
**Outputs**: `{source_stem}__{method}__{seed}.png` where `method ∈ {original, sdedit, controlnet, ours}`.

---

## 5. Workflow Rules

**5.1 Test before Compare.** Every new algorithmic function ships with a pytest. The four unit suites (tokenizer / mask / FFT / injection gate) must pass on every PR. Tests use `tests/fixtures/` small images and mock UNets — **never download real models in tests**.

**5.2 Serial model loading.** Two diffusion models must **never** live in memory simultaneously. Load → infer → `del; gc.collect(); empty_cache()` → next. This is centralized in `utils/memory.py::with_isolated_model()`. Hand-rolled `del` in business code is forbidden.

**5.3 Mask before render (Phase 2).** `run_phase2.py` must support `--dry-run-mask` to visualize the mask alone. No code path may skip mask verification before a full render.

---

## 6. DO NOT

1. ❌ Hardcode token indices (e.g. `prompt_ids[5]` for "shirt"). Use `utils/tokenizer.py` with sub-word merging.
2. ❌ Hardcode device or dtype. `.cuda()` / `torch.float16` literals are banned. Go through `utils/device.py::get_device_and_dtype()`.
3. ❌ Ignore MPS quirks. `fp16` on Mac causes NaN / black images for some ops — `device.py` must auto-downgrade to `fp32` / `bf16`.
4. ❌ Cross-import between `core/` and `baselines/`.
5. ❌ Skip tests and jump to full-res renders.
6. ❌ Commit `experiments/*/outputs/` large files (gitignored; curate into `assets/` manually).
7. ❌ Use `print` for debugging. Use `loguru`.
8. ❌ Use `np.fft` on the hot path — it breaks GPU pipelining. `torch.fft` only.
9. ❌ Mutate a `configs/*.yaml` that backs a published experiment. Create a new file instead.
10. ❌ Inline random seeds. Seeds come from config via `utils/seed.py::seed_everything()`.
11. ❌ Swallow exceptions (`except Exception: pass`). Black-image debugging dies here.
12. ❌ Add new dependencies silently — especially hardware-sensitive ones.

---

## 7. Code Style

- Line width 100; `ruff` + `black` enforced.
- Type hints mandatory on public APIs; Google-style docstrings.
- Document tensor shapes:
  ```python
  def inject_kv(ref_kv: torch.Tensor, gen_kv: torch.Tensor) -> torch.Tensor:
      """Inject reference K/V into generation stream.

      Args:
          ref_kv: reference K/V, shape (B, H, N, D)
          gen_kv: generation K/V, shape (B, H, N, D)
      Returns:
          injected K/V, shape (B, H, N, D)
      """
  ```
- Comments explain **why**, not what. Paper-equation lines need a reference, e.g. `# § 3.2 Eq.(4)`.
- One function = one responsibility. > 50 lines → split.

---

## 8. Git

- Branches: `main` (paper-final) / `dev` / `feat/<topic>` / `exp/<exp-id>`.
- Conventional Commits:
  - `feat(core): add FFT blend with adaptive cutoff`
  - `fix(mps): force fp32 to avoid black image on Mac`
  - `test(mask): cover empty cross-attention case`
  - `exp: phase1 silk floral seed42`
- **Every paper figure/table must trace to a commit hash + config yaml.** Reproducibility is non-negotiable.

---

## 9. Cross-Platform

- Dev: MacBook (MPS) for fast small-res iteration.
- Final: CUDA GPU for paper-resolution renders.
- `utils/device.py` must:
  1. Auto-detect `cuda` / `mps` / `cpu`.
  2. Pick dtype: CUDA → `fp16`, MPS → `fp32` (or `bf16` on macOS 14+, verified), CPU → `fp32`.
  3. Provide `safe_to(tensor, device, dtype)` skipping unsupported MPS dtype casts.
- Anything that runs on CUDA must also pass a small-res MPS smoke test before merging.

---

## 10. When in Doubt, Ask

Claude must stop and ask before:
- Modifying `core/` code that backs a published experiment.
- Adding a dependency not in §2.
- Touching edge cases the test suite does not cover (new samplers, new architectures).
- Renaming domain terms tied to the paper.

> **Plan first, render later.** One extra question is cheaper than a silent bug in the main algorithm.

---

## 11. Key References

```
papers/
├── 2208_01626v1.pdf                          # Prompt-to-Prompt (P2P) — basis for cross_attn_mask (Phase 2)
└── Tumanyan_Plug-and-Play_CVPR_2023.pdf      # PnP Diffusion Features — basis for attention_injection.py (Phase 1)
```

Rules:
- Read PnP before implementing `core/attention_injection.py`. Summarize §4 and get approval before coding.
- Read P2P before implementing `core/mask_extraction.py`. Summarize §3.1–3.2 and get approval before coding.
- ❌ Do not start either file without paper review and explicit approval.

---

*Last updated: 2026-05-19*
