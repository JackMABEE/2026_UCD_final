"""Tests for baselines/controlnet_runner.py — ControlNet-conditioned baseline.

All tests run on CPU with mocked model components — no real weights downloaded.

Public contract under test
--------------------------
ControlNetRunner(unet, vae, scheduler, tokenizer, text_encoder, controlnet, cfg)
    cfg must contain canny_low and canny_high (Canny edge thresholds from config,
    never hardcoded).  Models are held in a factory closure; run() uses
    with_isolated_model so GPU/MPS cache is freed after each call.

ControlNetRunner.run(source_image, prompt, num_inference_steps) -> PIL.Image
    1. Canny-preprocess source_image → condition tensor (B, 3, H, W) in [0, 1].
    2. Initialise latents from torch.randn (txt2img — no VAE encode of source).
    3. For each DDIM step t:
         down_res, mid_res = controlnet(latents, t, text_embeds, controlnet_cond=cond)
         noise_pred = unet(latents, t, text_embeds,
                          down_block_additional_residuals=down_res,
                          mid_block_additional_residual=mid_res)
         latents = scheduler.step(noise_pred, t, latents).prev_sample
    4. Decode latents with VAE once → RGB PIL Image.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from attn_texture.utils.memory import with_isolated_model

# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------

_NUM_STEPS = 3
_H, _W = 32, 32  # source image size; latent = 4×4


@dataclass
class _UNetOutput:
    sample: torch.Tensor


class _MockUNet(nn.Module):
    """UNet stub that records ControlNet residual kwargs."""

    def __init__(self) -> None:
        super().__init__()
        self.received_down_residuals: list | None = None
        self.received_mid_residual: torch.Tensor | None = None

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states=None,
        down_block_additional_residuals=None,
        mid_block_additional_residual=None,
        return_dict: bool = True,
        **kwargs,
    ) -> _UNetOutput:
        self.received_down_residuals = down_block_additional_residuals
        self.received_mid_residual = mid_block_additional_residual
        return _UNetOutput(sample=torch.zeros_like(sample))


class _MockControlNet(nn.Module):
    """ControlNet stub that records its cond inputs and call count."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count: int = 0
        self.last_controlnet_cond: torch.Tensor | None = None

    def forward(
        self,
        sample: torch.Tensor,
        timestep,
        encoder_hidden_states=None,
        controlnet_cond: torch.Tensor | None = None,
        return_dict: bool = False,
        **kwargs,
    ):
        self.call_count += 1
        self.last_controlnet_cond = controlnet_cond
        B, C, H, W = sample.shape
        # Return (list[down_res], mid_res) — matches diffusers ControlNetOutput unpacking
        down = [torch.zeros(B, C, H, W, device=sample.device, dtype=sample.dtype)]
        mid = torch.zeros(B, C, H, W, device=sample.device, dtype=sample.dtype)
        return down, mid


class _MockVAE:
    class config:
        scaling_factor: float = 0.18215

    def __init__(self) -> None:
        self.encode = MagicMock()
        self.decode = MagicMock(side_effect=self._decode)

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

    def _set_timesteps(self, num_inference_steps: int) -> None:
        self.timesteps = torch.linspace(999, 0, num_inference_steps).long()

    def _step(self, noise_pred: torch.Tensor, t, latents: torch.Tensor) -> _StepResult:
        return _StepResult(prev_sample=latents.clone())


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
def controlnet() -> _MockControlNet:
    return _MockControlNet()


@pytest.fixture()
def cfg():
    return OmegaConf.create({"canny_low": 100, "canny_high": 200, "guidance_scale": 7.5})


@pytest.fixture(autouse=True)
def force_cpu():
    with patch(
        "attn_texture.baselines.controlnet_runner.get_device_and_dtype",
        return_value=("cpu", torch.float32),
    ):
        yield


@pytest.fixture()
def runner(unet, vae, scheduler, tokenizer, text_encoder, controlnet, cfg):
    from attn_texture.baselines.controlnet_runner import ControlNetRunner

    return ControlNetRunner(unet, vae, scheduler, tokenizer, text_encoder, controlnet, cfg)


@pytest.fixture()
def source_image() -> Image.Image:
    """32×32 RGB image with enough variation to produce Canny edges."""
    arr = np.zeros((_H, _W, 3), dtype=np.uint8)
    arr[8:24, 8:24] = 200  # bright square — creates edges for Canny
    return Image.fromarray(arr, mode="RGB")


def _run(runner, source_image) -> Image.Image:
    return runner.run(
        source_image=source_image,
        prompt="a floral silk texture",
        num_inference_steps=_NUM_STEPS,
    )


# ---------------------------------------------------------------------------
# TestControlNetRunnerConstruction
# ---------------------------------------------------------------------------


class TestControlNetRunnerConstruction:
    def test_construction_succeeds(self, runner):
        from attn_texture.baselines.controlnet_runner import ControlNetRunner

        assert isinstance(runner, ControlNetRunner)

    def test_with_isolated_model_called_during_run(self, runner, source_image):
        with patch(
            "attn_texture.baselines.controlnet_runner.with_isolated_model",
            wraps=with_isolated_model,
        ) as spy:
            _run(runner, source_image)
        spy.assert_called_once()

    def test_models_not_stored_as_direct_attributes(
        self, unet, vae, scheduler, tokenizer, text_encoder, controlnet, cfg
    ):
        from attn_texture.baselines.controlnet_runner import ControlNetRunner

        r = ControlNetRunner(unet, vae, scheduler, tokenizer, text_encoder, controlnet, cfg)
        assert not hasattr(r, "unet"), "unet must not be a direct attribute"
        assert not hasattr(r, "vae"), "vae must not be a direct attribute"
        assert not hasattr(r, "controlnet"), "controlnet must not be a direct attribute"

    def test_missing_canny_low_raises(self, unet, vae, scheduler, tokenizer, text_encoder, controlnet):
        from omegaconf.errors import ConfigAttributeError
        from attn_texture.baselines.controlnet_runner import ControlNetRunner

        bad_cfg = OmegaConf.create({"canny_high": 200})
        with pytest.raises((ConfigAttributeError, KeyError)):
            ControlNetRunner(unet, vae, scheduler, tokenizer, text_encoder, controlnet, bad_cfg)

    def test_missing_canny_high_raises(self, unet, vae, scheduler, tokenizer, text_encoder, controlnet):
        from omegaconf.errors import ConfigAttributeError
        from attn_texture.baselines.controlnet_runner import ControlNetRunner

        bad_cfg = OmegaConf.create({"canny_low": 100})
        with pytest.raises((ConfigAttributeError, KeyError)):
            ControlNetRunner(unet, vae, scheduler, tokenizer, text_encoder, controlnet, bad_cfg)


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
# TestCannyPreprocessing
# ---------------------------------------------------------------------------


class TestCannyPreprocessing:
    def test_canny_called_with_config_thresholds(
        self, unet, vae, scheduler, tokenizer, text_encoder, controlnet, source_image
    ):
        """cv2.Canny must be called with canny_low / canny_high from config."""
        from attn_texture.baselines.controlnet_runner import ControlNetRunner

        cfg = OmegaConf.create({"canny_low": 42, "canny_high": 137, "guidance_scale": 7.5})
        runner = ControlNetRunner(unet, vae, scheduler, tokenizer, text_encoder, controlnet, cfg)

        with patch("attn_texture.baselines.controlnet_runner.cv2") as mock_cv2:
            mock_cv2.cvtColor.return_value = np.zeros((_H, _W), dtype=np.uint8)
            mock_cv2.Canny.return_value = np.zeros((_H, _W), dtype=np.uint8)
            mock_cv2.COLOR_RGB2GRAY = 7  # real constant value
            runner.run(source_image, "prompt", num_inference_steps=1)

        mock_cv2.Canny.assert_called_once()
        args = mock_cv2.Canny.call_args[0]
        assert args[1] == 42, f"Expected canny_low=42, got {args[1]}"
        assert args[2] == 137, f"Expected canny_high=137, got {args[2]}"

    def test_controlnet_cond_is_3_channel(self, runner, controlnet, source_image):
        """ControlNet condition tensor must be (B, 3, H, W)."""
        _run(runner, source_image)
        cond = controlnet.last_controlnet_cond
        assert cond is not None
        assert cond.ndim == 4
        assert cond.shape[1] == 3

    def test_controlnet_cond_in_unit_range(self, runner, controlnet, source_image):
        """Condition tensor pixel values must lie in [0, 1]."""
        _run(runner, source_image)
        cond = controlnet.last_controlnet_cond
        assert cond is not None
        assert float(cond.min()) >= 0.0 - 1e-4
        assert float(cond.max()) <= 1.0 + 1e-4

    def test_controlnet_cond_matches_source_image_spatial_dims(self, runner, controlnet, source_image):
        """Canny output spatial dims must match the source image (no silent resize)."""
        _run(runner, source_image)
        cond = controlnet.last_controlnet_cond
        assert cond is not None
        _, _, h, w = cond.shape
        assert h == _H and w == _W


# ---------------------------------------------------------------------------
# TestVAEInteraction
# ---------------------------------------------------------------------------


class TestVAEInteraction:
    def test_vae_encode_not_called(self, runner, vae, source_image):
        """ControlNet baseline is txt2img: latents start from randn, not encoded source."""
        _run(runner, source_image)
        vae.encode.assert_not_called()

    def test_vae_decode_called_once(self, runner, vae, source_image):
        _run(runner, source_image)
        assert vae.decode.call_count == 1

    def test_vae_decode_receives_4d_tensor(self, runner, vae, source_image):
        _run(runner, source_image)
        args, _ = vae.decode.call_args
        assert args[0].ndim == 4


# ---------------------------------------------------------------------------
# TestControlNetInteraction
# ---------------------------------------------------------------------------


class TestControlNetInteraction:
    def test_controlnet_called_every_step(self, runner, controlnet, source_image):
        _run(runner, source_image)
        assert controlnet.call_count == _NUM_STEPS

    def test_unet_receives_down_block_residuals(self, runner, unet, source_image):
        """UNet must receive down_block_additional_residuals from ControlNet output."""
        _run(runner, source_image)
        assert unet.received_down_residuals is not None

    def test_unet_receives_mid_block_residual(self, runner, unet, source_image):
        """UNet must receive mid_block_additional_residual from ControlNet output."""
        _run(runner, source_image)
        assert unet.received_mid_residual is not None

    def test_down_residuals_is_list(self, runner, unet, source_image):
        _run(runner, source_image)
        assert isinstance(unet.received_down_residuals, list)


# ---------------------------------------------------------------------------
# TestSchedulerInteraction
# ---------------------------------------------------------------------------


class TestSchedulerInteraction:
    def test_set_timesteps_called_with_num_steps(self, runner, scheduler, source_image):
        _run(runner, source_image)
        scheduler.set_timesteps.assert_called_once_with(_NUM_STEPS)

    def test_step_called_n_times(self, runner, scheduler, source_image):
        _run(runner, source_image)
        assert scheduler.step.call_count == _NUM_STEPS

    def test_latents_start_from_randn_not_encoded(self, runner, vae, source_image):
        """VAE encode must not be called — latents are initialised from torch.randn."""
        _run(runner, source_image)
        vae.encode.assert_not_called()
