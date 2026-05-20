"""Smoke tests for core/two_pass_pipeline.py — TwoPassPipeline orchestration.

All tests run on CPU with mocked model components — no real weights downloaded.
The pipeline under test must:
  1. Encode the source image with VAE once (ref pass only)
  2. Run a denoising loop of N steps, at each step:
       a. Ref pass: begin_ref_pass → UNet(z_ref, ...) → end_ref_pass
       b. Gen pass: begin_gen_pass(t, total) → UNet(z_gen, ...) → end_gen_pass
  3. Decode the final gen latent with VAE once and return a PIL Image.

Key constraints (CLAUDE.md):
  • device / dtype from get_device_and_dtype() — never hardcoded.
  • τ_f / τ_A from OmegaConf config, forwarded to AttentionInjector.
  • DDIM scheduler only.
  • No model loading — components injected as constructor args.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

# ---------------------------------------------------------------------------
# Mock UNet — same decoder topology as test_attn_injection_gate.py
# ---------------------------------------------------------------------------


class _MockAttn1(nn.Module):
    _DIM: int = 8

    def __init__(self) -> None:
        super().__init__()
        d = self._DIM
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(d, d, bias=False)])

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        scale = self._DIM**-0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        return self.to_out[0](attn @ v)


class _MockTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = _MockAttn1()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn1(x) + x


class _MockTransformer2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_MockTransformerBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.transformer_blocks:
            x = blk(x)
        return x


class _MockResNet(nn.Module):
    _CH: int = 4

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(self._CH, self._CH, 1)

    def forward(self, x: torch.Tensor, temb=None) -> torch.Tensor:
        return self.conv(x)


class _MockUpBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([_MockResNet(), _MockResNet()])
        self.attentions = nn.ModuleList([_MockTransformer2D()])


@dataclass
class _UNetOutput:
    sample: torch.Tensor


class _MockUNet(nn.Module):
    """Diffusers-compatible UNet stub that fires forward hooks via real up_blocks."""

    _B, _C, _H, _W = 1, 4, 4, 4
    _N, _D_SEQ = 16, 8

    def __init__(self) -> None:
        super().__init__()
        self.up_blocks = nn.ModuleList([_MockUpBlock() for _ in range(3)])

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | int,
        encoder_hidden_states: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> _UNetOutput:
        x_sp = sample
        # Flatten spatial dims for the sequence dimension expected by attention
        B, C, H, W = x_sp.shape
        x_seq = x_sp.reshape(B, C, H * W).transpose(1, 2)  # (B, N, C) → but _DIM=8, so pad
        # Build a compatible sequence tensor from scratch to keep attn dims clean
        x_seq = torch.zeros(B, self._N, self._D_SEQ, device=sample.device, dtype=sample.dtype)

        for blk in self.up_blocks:
            for res in blk.resnets:
                x_sp = res(x_sp)
            for attn in blk.attentions:
                x_seq = attn(x_seq)

        return _UNetOutput(sample=torch.zeros_like(sample))


# ---------------------------------------------------------------------------
# Mock VAE
# ---------------------------------------------------------------------------


class _MockVAE:
    """VAE stub: encode → latent, decode → (B, 3, H*8, W*8) image tensor."""

    class config:
        scaling_factor: float = 0.18215

    def __init__(self) -> None:
        self.encode = MagicMock(side_effect=self._encode)
        self.decode = MagicMock(side_effect=self._decode)

    def _encode(self, pixel_values: torch.Tensor):
        B, _, H, W = pixel_values.shape
        latent = torch.zeros(B, 4, H // 8, W // 8, device=pixel_values.device, dtype=pixel_values.dtype)
        result = types.SimpleNamespace()
        sample_fn = MagicMock(return_value=latent)
        result.latent_dist = types.SimpleNamespace(sample=sample_fn)
        return result

    def _decode(self, latents: torch.Tensor):
        B, _, H, W = latents.shape
        img = torch.zeros(B, 3, H * 8, W * 8, device=latents.device, dtype=latents.dtype)
        return types.SimpleNamespace(sample=img)


# ---------------------------------------------------------------------------
# Mock Scheduler (DDIM-like)
# ---------------------------------------------------------------------------


@dataclass
class _StepResult:
    prev_sample: torch.Tensor


class _MockScheduler:
    """Minimal DDIM scheduler stub."""

    class config:
        num_train_timesteps: int = 1000

    def __init__(self) -> None:
        self.timesteps: torch.Tensor = torch.tensor([])
        self.set_timesteps = MagicMock(side_effect=self._set_timesteps)
        self.step = MagicMock(side_effect=self._step)
        self.add_noise = MagicMock(side_effect=self._add_noise)

    def _set_timesteps(self, num_inference_steps: int) -> None:
        self.timesteps = torch.linspace(999, 0, num_inference_steps).long()

    def _step(self, noise_pred: torch.Tensor, t, latents: torch.Tensor) -> _StepResult:
        return _StepResult(prev_sample=latents.clone())

    def _add_noise(self, original: torch.Tensor, noise: torch.Tensor, timesteps) -> torch.Tensor:
        return noise.clone()


# ---------------------------------------------------------------------------
# Mock Tokenizer / TextEncoder
# ---------------------------------------------------------------------------


class _MockTokenizer:
    model_max_length: int = 77

    def __call__(self, text, return_tensors="pt", padding=None, max_length=None, truncation=False):
        ids = torch.zeros(1, self.model_max_length, dtype=torch.long)
        return types.SimpleNamespace(input_ids=ids)


class _MockTextEncoder(nn.Module):
    def forward(self, input_ids: torch.Tensor, **kwargs):
        B, S = input_ids.shape
        hidden = torch.zeros(B, S, 768, device=input_ids.device, dtype=torch.float32)
        return types.SimpleNamespace(last_hidden_state=hidden)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NUM_STEPS = 3


@pytest.fixture()
def unet() -> _MockUNet:
    return _MockUNet()


@pytest.fixture()
def vae() -> _MockVAE:
    return _MockVAE()


@pytest.fixture()
def scheduler() -> _MockScheduler:
    return _MockScheduler()


@pytest.fixture()
def tokenizer() -> _MockTokenizer:
    return _MockTokenizer()


@pytest.fixture()
def text_encoder() -> _MockTextEncoder:
    return _MockTextEncoder()


@pytest.fixture()
def cfg():
    return OmegaConf.create({"tau_f": 0.8, "tau_A": 0.5, "guidance_scale": 7.5})


@pytest.fixture()
def source_image() -> Image.Image:
    """Tiny 32×32 white source image (no disk I/O needed)."""
    return Image.new("RGB", (32, 32), color=(200, 200, 200))


@pytest.fixture(autouse=True)
def force_cpu():
    """Patch get_device_and_dtype so all tests run on CPU regardless of hardware."""
    with patch(
        "attn_texture.core.two_pass_pipeline.get_device_and_dtype",
        return_value=("cpu", torch.float32),
    ):
        yield


@pytest.fixture()
def pipeline(unet, vae, scheduler, tokenizer, text_encoder, cfg):
    from attn_texture.core.two_pass_pipeline import TwoPassPipeline

    return TwoPassPipeline(unet, vae, scheduler, tokenizer, text_encoder, cfg)


def _run(pipeline, source_image) -> Image.Image:
    return pipeline.run(
        source_image=source_image,
        ref_prompt="a plain fabric texture",
        gen_prompt="a floral silk texture",
        num_inference_steps=_NUM_STEPS,
    )


# ---------------------------------------------------------------------------
# TestPipelineConstruction
# ---------------------------------------------------------------------------


class TestPipelineConstruction:
    """Pipeline reads τ from config and registers hooks at construction time."""

    def test_tau_f_from_config(self, pipeline):
        assert pipeline.injector.tau_f == pytest.approx(0.8)

    def test_tau_A_from_config(self, pipeline):
        assert pipeline.injector.tau_A == pytest.approx(0.5)

    def test_hooks_registered_at_construction(self, pipeline):
        # 1 feature hook + 3 attn1 hooks (3 up_blocks × 1 attn each)
        assert len(pipeline.injector._hook_handles) == 4

    def test_custom_tau_respected(self, unet, vae, scheduler, tokenizer, text_encoder):
        from attn_texture.core.two_pass_pipeline import TwoPassPipeline

        cfg = OmegaConf.create({"tau_f": 0.3, "tau_A": 0.6, "guidance_scale": 7.5})
        p = TwoPassPipeline(unet, vae, scheduler, tokenizer, text_encoder, cfg)
        assert p.injector.tau_f == pytest.approx(0.3)
        assert p.injector.tau_A == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# TestRunOutput
# ---------------------------------------------------------------------------


class TestRunOutput:
    """run() must return a valid PIL RGB image."""

    def test_returns_pil_image(self, pipeline, source_image):
        result = _run(pipeline, source_image)
        assert isinstance(result, Image.Image)

    def test_output_is_rgb(self, pipeline, source_image):
        result = _run(pipeline, source_image)
        assert result.mode == "RGB"

    def test_output_has_positive_dimensions(self, pipeline, source_image):
        result = _run(pipeline, source_image)
        w, h = result.size
        assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# TestVAEInteraction
# ---------------------------------------------------------------------------


class TestVAEInteraction:
    """VAE encode called exactly once; decode called exactly once."""

    def test_encode_called_once(self, pipeline, vae, source_image):
        _run(pipeline, source_image)
        assert vae.encode.call_count == 1

    def test_decode_called_once(self, pipeline, vae, source_image):
        _run(pipeline, source_image)
        assert vae.decode.call_count == 1

    def test_encode_receives_4d_float_tensor(self, pipeline, vae, source_image):
        _run(pipeline, source_image)
        args, _ = vae.encode.call_args
        pixel_tensor = args[0]
        assert pixel_tensor.ndim == 4
        assert pixel_tensor.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_encode_receives_normalized_pixels(self, pipeline, vae, source_image):
        """Pixels must be in [-1, 1] (standard VAE convention)."""
        _run(pipeline, source_image)
        args, _ = vae.encode.call_args
        pixel_tensor = args[0]
        assert pixel_tensor.min() >= -1.0 - 1e-4
        assert pixel_tensor.max() <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# TestSchedulerInteraction
# ---------------------------------------------------------------------------


class TestSchedulerInteraction:
    """Scheduler is driven correctly by the denoising loop."""

    def test_set_timesteps_called_once(self, pipeline, scheduler, source_image):
        _run(pipeline, source_image)
        scheduler.set_timesteps.assert_called_once_with(_NUM_STEPS)

    def test_step_called_twice_per_denoising_step(self, pipeline, scheduler, source_image):
        """Two UNet forwards (ref + gen) per denoising step → 2 × N scheduler.step calls."""
        _run(pipeline, source_image)
        assert scheduler.step.call_count == 2 * _NUM_STEPS

    def test_add_noise_called(self, pipeline, scheduler, source_image):
        """Latents must be initialised from noise via the scheduler."""
        _run(pipeline, source_image)
        assert scheduler.add_noise.call_count >= 1


# ---------------------------------------------------------------------------
# TestInjectorOrchestration
# ---------------------------------------------------------------------------


class TestInjectorOrchestration:
    """AttentionInjector lifecycle is invoked once per denoising step in the right order."""

    @pytest.fixture()
    def spy_pipeline(self, pipeline):
        """Wrap injector methods to record call order."""
        pipeline._call_log: list[str] = []
        orig_begin_ref = pipeline.injector.begin_ref_pass
        orig_end_ref = pipeline.injector.end_ref_pass
        orig_begin_gen = pipeline.injector.begin_gen_pass
        orig_end_gen = pipeline.injector.end_gen_pass

        def _begin_ref():
            pipeline._call_log.append("begin_ref")
            orig_begin_ref()

        def _end_ref():
            pipeline._call_log.append("end_ref")
            return orig_end_ref()

        def _begin_gen(step, total):
            pipeline._call_log.append(f"begin_gen:{step}:{total}")
            orig_begin_gen(step, total)

        def _end_gen():
            pipeline._call_log.append("end_gen")
            orig_end_gen()

        pipeline.injector.begin_ref_pass = _begin_ref
        pipeline.injector.end_ref_pass = _end_ref
        pipeline.injector.begin_gen_pass = _begin_gen
        pipeline.injector.end_gen_pass = _end_gen
        return pipeline

    def test_begin_ref_called_n_times(self, spy_pipeline, source_image):
        _run(spy_pipeline, source_image)
        count = spy_pipeline._call_log.count("begin_ref")
        assert count == _NUM_STEPS

    def test_end_ref_called_n_times(self, spy_pipeline, source_image):
        _run(spy_pipeline, source_image)
        count = spy_pipeline._call_log.count("end_ref")
        assert count == _NUM_STEPS

    def test_begin_gen_called_n_times(self, spy_pipeline, source_image):
        _run(spy_pipeline, source_image)
        gen_calls = [e for e in spy_pipeline._call_log if e.startswith("begin_gen")]
        assert len(gen_calls) == _NUM_STEPS

    def test_end_gen_called_n_times(self, spy_pipeline, source_image):
        _run(spy_pipeline, source_image)
        count = spy_pipeline._call_log.count("end_gen")
        assert count == _NUM_STEPS

    def test_ref_before_gen_within_each_step(self, spy_pipeline, source_image):
        """Within every denoising step: begin_ref → end_ref → begin_gen → end_gen."""
        _run(spy_pipeline, source_image)
        log = spy_pipeline._call_log
        # Split into groups of 4 events per step
        assert len(log) == 4 * _NUM_STEPS
        for i in range(_NUM_STEPS):
            chunk = log[i * 4 : (i + 1) * 4]
            assert chunk[0] == "begin_ref", f"step {i}: expected begin_ref, got {chunk[0]}"
            assert chunk[1] == "end_ref", f"step {i}: expected end_ref, got {chunk[1]}"
            assert chunk[2].startswith("begin_gen"), f"step {i}: expected begin_gen, got {chunk[2]}"
            assert chunk[3] == "end_gen", f"step {i}: expected end_gen, got {chunk[3]}"

    def test_begin_gen_receives_total_train_timesteps(self, spy_pipeline, source_image):
        """total_steps passed to begin_gen_pass must equal scheduler.config.num_train_timesteps."""
        _run(spy_pipeline, source_image)
        gen_calls = [e for e in spy_pipeline._call_log if e.startswith("begin_gen")]
        for event in gen_calls:
            _, _, total_str = event.split(":")
            assert int(total_str) == 1000  # _MockScheduler.config.num_train_timesteps

    def test_begin_gen_receives_integer_timestep(self, spy_pipeline, source_image):
        """The step value passed to begin_gen_pass must be a Python int."""
        _run(spy_pipeline, source_image)
        gen_calls = [e for e in spy_pipeline._call_log if e.startswith("begin_gen")]
        for event in gen_calls:
            _, step_str, _ = event.split(":")
            assert step_str.lstrip("-").isdigit()


# ---------------------------------------------------------------------------
# TestRefPassCapture
# ---------------------------------------------------------------------------


class TestRefPassCapture:
    """After a complete run, the injector holds valid captured features."""

    def test_spatial_features_not_none_after_run(self, pipeline, source_image):
        _run(pipeline, source_image)
        assert pipeline.injector.captured.spatial_features is not None

    def test_self_attentions_populated_after_run(self, pipeline, source_image):
        _run(pipeline, source_image)
        assert len(pipeline.injector.captured.self_attentions) > 0

    def test_spatial_features_is_4d_tensor(self, pipeline, source_image):
        _run(pipeline, source_image)
        sf = pipeline.injector.captured.spatial_features
        assert sf is not None
        assert sf.ndim == 4  # (B, C, H, W)
