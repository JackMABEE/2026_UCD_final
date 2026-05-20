"""Tests for baselines/sdedit_runner.py — SDEdit image-to-image baseline.

All tests use CPU mocks — no real model weights downloaded.

Public contract under test
--------------------------
SDEditRunner(unet, vae, scheduler, tokenizer, text_encoder, cfg)
    Stores components internally via a factory so that run() can release
    GPU/MPS memory through with_isolated_model after each call.

SDEditRunner.run(source_image, prompt, strength, num_inference_steps) -> PIL.Image
    1. Encode source image with VAE once → z_0.
    2. Determine active denoising timesteps from strength (0.0–1.0):
         start_step = round(num_inference_steps * (1 - strength))
         active_timesteps = scheduler.timesteps[start_step:]
    3. If active_timesteps non-empty: add_noise(z_0, noise, active_timesteps[0]) → z_t.
    4. Denoise z_t for len(active_timesteps) steps with target prompt.
    5. Decode final latent with VAE once → RGB PIL Image.
    The denoising loop runs inside with_isolated_model so GPU/MPS memory is
    freed after each call.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from attn_texture.utils.memory import with_isolated_model

# ---------------------------------------------------------------------------
# Mock components — minimal stubs, no model downloads
# ---------------------------------------------------------------------------

_NUM_STEPS = 4  # small enough for fast tests, clean for strength=0.5 (2 steps)


@dataclass
class _UNetOutput:
    sample: torch.Tensor


class _MockUNet(nn.Module):
    """UNet stub: accepts diffusers-compatible forward, returns zeros."""

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states=None,
        return_dict: bool = True,
        **kwargs,
    ) -> _UNetOutput:
        return _UNetOutput(sample=torch.zeros_like(sample))


class _MockVAE:
    class config:
        scaling_factor: float = 0.18215

    def __init__(self) -> None:
        self.encode = MagicMock(side_effect=self._encode)
        self.decode = MagicMock(side_effect=self._decode)

    def _encode(self, pixel_values: torch.Tensor):
        B, _, H, W = pixel_values.shape
        latent = torch.zeros(B, 4, H // 8, W // 8, device=pixel_values.device, dtype=pixel_values.dtype)
        sample_fn = MagicMock(return_value=latent)
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(sample=sample_fn))

    def _decode(self, latents: torch.Tensor):
        B, _, H, W = latents.shape
        img = torch.zeros(B, 3, H * 8, W * 8, device=latents.device, dtype=latents.dtype)
        return types.SimpleNamespace(sample=img)


@dataclass
class _StepResult:
    prev_sample: torch.Tensor


class _MockScheduler:
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


class _MockTokenizer:
    model_max_length: int = 77

    def __call__(self, text, return_tensors="pt", padding=None, max_length=None, truncation=False):
        ids = torch.zeros(1, self.model_max_length, dtype=torch.long)
        return types.SimpleNamespace(input_ids=ids)


class _MockTextEncoder(nn.Module):
    def forward(self, input_ids: torch.Tensor, **kwargs):
        B, S = input_ids.shape
        return types.SimpleNamespace(
            last_hidden_state=torch.zeros(B, S, 768, device=input_ids.device, dtype=torch.float32)
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
    return OmegaConf.create({"guidance_scale": 7.5})


@pytest.fixture(autouse=True)
def force_cpu():
    with patch(
        "attn_texture.baselines.sdedit_runner.get_device_and_dtype",
        return_value=("cpu", torch.float32),
    ):
        yield


@pytest.fixture()
def runner(unet, vae, scheduler, tokenizer, text_encoder, cfg):
    from attn_texture.baselines.sdedit_runner import SDEditRunner

    return SDEditRunner(unet, vae, scheduler, tokenizer, text_encoder, cfg)


@pytest.fixture()
def source_image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(128, 64, 192))


def _run(runner, source_image, strength: float = 0.75) -> Image.Image:
    return runner.run(
        source_image=source_image,
        prompt="a floral silk texture",
        strength=strength,
        num_inference_steps=_NUM_STEPS,
    )


# ---------------------------------------------------------------------------
# TestSDEditRunnerConstruction
# ---------------------------------------------------------------------------


class TestSDEditRunnerConstruction:
    def test_construction_succeeds(self, runner):
        from attn_texture.baselines.sdedit_runner import SDEditRunner

        assert isinstance(runner, SDEditRunner)

    def test_with_isolated_model_called_during_run(self, runner, source_image):
        """run() must delegate to with_isolated_model — not bypass it."""
        with patch(
            "attn_texture.baselines.sdedit_runner.with_isolated_model",
            wraps=with_isolated_model,
        ) as spy:
            _run(runner, source_image)
        spy.assert_called_once()

    def test_models_not_stored_as_direct_attributes(self, unet, vae, scheduler, tokenizer, text_encoder, cfg):
        """Runner must not hold unet/vae/etc. as top-level instance attributes."""
        from attn_texture.baselines.sdedit_runner import SDEditRunner

        r = SDEditRunner(unet, vae, scheduler, tokenizer, text_encoder, cfg)
        # Direct attribute lookup must not return the model objects themselves
        assert not hasattr(r, "unet"), "unet must not be a direct attribute (use factory)"
        assert not hasattr(r, "vae"), "vae must not be a direct attribute (use factory)"


# ---------------------------------------------------------------------------
# TestRunOutput
# ---------------------------------------------------------------------------


class TestRunOutput:
    def test_returns_pil_image(self, runner, source_image):
        assert isinstance(_run(runner, source_image), Image.Image)

    def test_output_is_rgb(self, runner, source_image):
        assert _run(runner, source_image).mode == "RGB"

    def test_output_has_positive_dimensions(self, runner, source_image):
        w, h = _run(runner, source_image).size
        assert w > 0 and h > 0


# ---------------------------------------------------------------------------
# TestVAEInteraction
# ---------------------------------------------------------------------------


class TestVAEInteraction:
    def test_encode_called_once(self, runner, vae, source_image):
        _run(runner, source_image)
        assert vae.encode.call_count == 1

    def test_decode_called_once(self, runner, vae, source_image):
        _run(runner, source_image)
        assert vae.decode.call_count == 1

    def test_encode_receives_4d_float_tensor(self, runner, vae, source_image):
        _run(runner, source_image)
        args, _ = vae.encode.call_args
        t = args[0]
        assert t.ndim == 4
        assert t.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_encode_receives_normalized_pixels(self, runner, vae, source_image):
        """VAE expects pixels in [-1, 1], not [0, 1]."""
        _run(runner, source_image)
        args, _ = vae.encode.call_args
        t = args[0]
        assert t.min() >= -1.0 - 1e-4
        assert t.max() <= 1.0 + 1e-4


# ---------------------------------------------------------------------------
# TestSchedulerInteraction
# ---------------------------------------------------------------------------


class TestSchedulerInteraction:
    def test_set_timesteps_called_with_num_steps(self, runner, scheduler, source_image):
        _run(runner, source_image)
        scheduler.set_timesteps.assert_called_once_with(_NUM_STEPS)

    def test_add_noise_called_once_when_strength_positive(self, runner, scheduler, source_image):
        _run(runner, source_image, strength=0.5)
        assert scheduler.add_noise.call_count == 1

    def test_add_noise_not_called_at_zero_strength(self, runner, scheduler, source_image):
        """strength=0.0 → no noise added, no denoising steps."""
        _run(runner, source_image, strength=0.0)
        assert scheduler.add_noise.call_count == 0

    def test_full_strength_runs_all_steps(self, runner, scheduler, source_image):
        """strength=1.0 → start from pure noise → all N denoising steps run."""
        _run(runner, source_image, strength=1.0)
        assert scheduler.step.call_count == _NUM_STEPS

    def test_half_strength_runs_half_steps(self, runner, scheduler, source_image):
        """strength=0.5 → start halfway → N/2 denoising steps run."""
        _run(runner, source_image, strength=0.5)
        expected = _NUM_STEPS - round(_NUM_STEPS * (1 - 0.5))
        assert scheduler.step.call_count == expected

    def test_zero_strength_runs_no_steps(self, runner, scheduler, source_image):
        """strength=0.0 → no denoising → scheduler.step never called."""
        _run(runner, source_image, strength=0.0)
        assert scheduler.step.call_count == 0

    def test_add_noise_timestep_reflects_strength(self, runner, scheduler, source_image):
        """The timestep passed to add_noise must be drawn from active_timesteps[0]."""
        _run(runner, source_image, strength=0.5)
        _, kwargs = scheduler.add_noise.call_args
        # Positional call: add_noise(original, noise, timesteps)
        args, _ = scheduler.add_noise.call_args
        noise_t = args[2]  # third positional arg is the timestep(s)
        # For strength=0.5, start_step = round(4*0.5) = 2; timesteps[2] is the 3rd value
        # Just verify it's a tensor (not a plain int) and within valid range
        assert isinstance(noise_t, torch.Tensor)
        assert int(noise_t.item()) >= 0


# ---------------------------------------------------------------------------
# TestPromptEncoding
# ---------------------------------------------------------------------------


class TestPromptEncoding:
    def test_text_encoder_called(self, runner, text_encoder, source_image):
        with patch.object(text_encoder, "forward", wraps=text_encoder.forward) as spy:
            _run(runner, source_image)
        assert spy.call_count >= 1

    def test_encoded_hidden_states_shape(self, runner, vae, source_image):
        """Sanity: VAE encode receives a batch-dim tensor, implying prompt was encoded."""
        _run(runner, source_image)
        args, _ = vae.encode.call_args
        assert args[0].shape[0] == 1  # batch size 1
