"""Tests for baselines/masactrl_runner.py — MasaCtrl K/V injection baseline.

All tests use CPU mocks — no real model weights downloaded.

Contract under test
-------------------
MasaCtrlRunner(unet, vae, scheduler, tokenizer, text_encoder, cfg)
    cfg must contain: step_start, layer_start, guidance_scale, num_inference_steps.
    Model objects are captured in a factory closure (not direct attributes).

MasaCtrlRunner.run(source_image, ref_prompt, gen_prompt, num_inference_steps) -> PIL.Image
    1. Encode source image with VAE → z_src.
    2. DDIM inversion: z_src → z_T (ascending timesteps, num_inference_steps calls).
    3. Register _MasaCtrlAttnProcessor on every UNet self-attention layer.
    4. Denoising loop with batch-of-4: [src_unc, tgt_unc, src_cond, tgt_cond].
       Active processors inject K,V from source slots into target slots.
    5. Decode target latent via VAE → RGB PIL Image.

_MasaCtrlAttnProcessor
    Injects K and V from source slots (0, 2) into target slots (1, 3) when:
      - batch size == 4
      - _cur_step >= step_start
      - layer_idx  >= layer_start
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
# Constants
# ---------------------------------------------------------------------------

_NUM_STEPS = 4
# Realistic diffusers-style names: 3 self-attn (attn1) + 3 cross-attn (attn2)
_ATTN_LAYER_NAMES = [
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor",
    "mid_block.attentions.0.transformer_blocks.0.attn1.processor",
    "mid_block.attentions.0.transformer_blocks.0.attn2.processor",
    "up_blocks.0.attentions.0.transformer_blocks.0.attn1.processor",
    "up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor",
]
_SELF_ATTN_NAMES = [n for n in _ATTN_LAYER_NAMES if "attn1" in n]  # 3 names

# ---------------------------------------------------------------------------
# Mock components — minimal stubs, no model downloads
# ---------------------------------------------------------------------------


@dataclass
class _UNetOutput:
    sample: torch.Tensor


class _MockUNet(nn.Module):
    """UNet stub with set_attn_processor / attn_processors support."""

    def __init__(self) -> None:
        super().__init__()
        self._processors: dict = {name: MagicMock() for name in _ATTN_LAYER_NAMES}
        self.set_attn_processor = MagicMock(side_effect=self._do_set_processor)

    @property
    def attn_processors(self) -> dict:
        return dict(self._processors)

    def _do_set_processor(self, processors: dict) -> None:
        self._processors = dict(processors)

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
        latent = torch.zeros(B, 4, H // 8, W // 8, dtype=pixel_values.dtype)
        sample_fn = MagicMock(return_value=latent)
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(sample=sample_fn))

    def _decode(self, latents: torch.Tensor):
        B, _, H, W = latents.shape
        img = torch.zeros(B, 3, H * 8, W * 8, dtype=latents.dtype)
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
            last_hidden_state=torch.zeros(B, S, 768, dtype=torch.float32)
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
    return OmegaConf.create(
        {
            "step_start": 4,
            "layer_start": 10,
            "guidance_scale": 7.5,
            "num_inference_steps": _NUM_STEPS,
        }
    )


@pytest.fixture(autouse=True)
def force_cpu():
    with patch(
        "attn_texture.baselines.masactrl_runner.get_device_and_dtype",
        return_value=("cpu", torch.float32),
    ):
        yield


@pytest.fixture()
def runner(unet, vae, scheduler, tokenizer, text_encoder, cfg):
    from attn_texture.baselines.masactrl_runner import MasaCtrlRunner

    return MasaCtrlRunner(unet, vae, scheduler, tokenizer, text_encoder, cfg)


@pytest.fixture()
def source_image() -> Image.Image:
    return Image.new("RGB", (32, 32), color=(128, 64, 192))


def _run(runner, source_image) -> Image.Image:
    return runner.run(
        source_image=source_image,
        ref_prompt="a plain silk fabric",
        gen_prompt="a floral silk fabric",
        num_inference_steps=_NUM_STEPS,
    )


# ---------------------------------------------------------------------------
# TestMasaCtrlAttnProcessor
# ---------------------------------------------------------------------------


class TestMasaCtrlAttnProcessor:
    """Unit tests for _MasaCtrlAttnProcessor in isolation from the runner."""

    _DIM = 16
    _HEADS = 2
    _SEQ = 8

    @classmethod
    def _make_attn(cls, batch: int = 4) -> MagicMock:
        """Build a minimal attn stub with identity projections and uniform scores."""
        dim, heads, seq = cls._DIM, cls._HEADS, cls._SEQ

        proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            proj.weight.copy_(torch.eye(dim))

        attn = MagicMock()
        attn.to_q = MagicMock(side_effect=lambda x: x)
        attn.to_k = MagicMock(side_effect=lambda x: x)
        attn.to_v = MagicMock(side_effect=lambda x: x)
        attn.head_to_batch_dim = MagicMock(
            side_effect=lambda x: x.reshape(x.shape[0] * heads, x.shape[1], x.shape[2] // heads)
        )
        attn.batch_to_head_dim = MagicMock(
            side_effect=lambda x: x.reshape(x.shape[0] // heads, x.shape[1], x.shape[2] * heads)
        )
        attn.get_attention_scores = MagicMock(
            side_effect=lambda q, k, m=None: torch.ones(q.shape[0], q.shape[1], q.shape[1]) / q.shape[1]
        )
        attn.to_out = [proj, nn.Dropout(0.0)]
        return attn

    def test_returns_tensor_with_correct_shape(self):
        from attn_texture.baselines.masactrl_runner import _MasaCtrlAttnProcessor

        proc = _MasaCtrlAttnProcessor(layer_idx=10, step_start=4, layer_start=10)
        proc._cur_step = 4
        attn = self._make_attn(batch=4)
        hidden = torch.randn(4, self._SEQ, self._DIM)
        result = proc(attn, hidden)
        assert result.shape == hidden.shape

    def test_kv_injection_replaces_target_slots(self):
        """When gating is active, k[1]←k[0] and k[3]←k[2] (tgt borrows src K/V)."""
        from attn_texture.baselines.masactrl_runner import _MasaCtrlAttnProcessor

        proc = _MasaCtrlAttnProcessor(layer_idx=10, step_start=4, layer_start=10)
        proc._cur_step = 4

        # Scale k/v by slot index so source and target are distinguishable
        slot_scale = torch.tensor([1.0, 10.0, 3.0, 30.0]).view(4, 1, 1)

        captured_k: list[torch.Tensor] = []
        h2b_call = [0]
        heads = self._HEADS

        def h2b(x):
            h2b_call[0] += 1
            if h2b_call[0] == 2:  # first=q, second=k, third=v
                captured_k.append(x.clone())
            return x.reshape(x.shape[0] * heads, x.shape[1], x.shape[2] // heads)

        attn = MagicMock()
        attn.to_q = MagicMock(side_effect=lambda x: x)
        attn.to_k = MagicMock(side_effect=lambda x: x * slot_scale)
        attn.to_v = MagicMock(side_effect=lambda x: x * slot_scale)
        attn.head_to_batch_dim = MagicMock(side_effect=h2b)
        attn.batch_to_head_dim = MagicMock(
            side_effect=lambda x: x.reshape(x.shape[0] // heads, x.shape[1], x.shape[2] * heads)
        )
        attn.get_attention_scores = MagicMock(
            side_effect=lambda q, k, m=None: torch.ones(q.shape[0], q.shape[1], q.shape[1]) / q.shape[1]
        )
        proj = nn.Linear(self._DIM, self._DIM, bias=False)
        with torch.no_grad():
            proj.weight.zero_()
        attn.to_out = [proj, nn.Identity()]

        hidden = torch.ones(4, self._SEQ, self._DIM)
        proc(attn, hidden)

        assert len(captured_k) == 1, "head_to_batch_dim not called with k"
        k = captured_k[0]  # (4, SEQ, DIM) — post-injection
        assert torch.allclose(k[0], k[1]), "K: tgt_uncond not injected from src_uncond"
        assert torch.allclose(k[2], k[3]), "K: tgt_cond not injected from src_cond"

    def test_injection_inactive_before_step_start(self):
        """No injection when _cur_step < step_start — output still has correct shape."""
        from attn_texture.baselines.masactrl_runner import _MasaCtrlAttnProcessor

        proc = _MasaCtrlAttnProcessor(layer_idx=10, step_start=4, layer_start=10)
        proc._cur_step = 0  # below threshold
        attn = self._make_attn()
        hidden = torch.randn(4, self._SEQ, self._DIM)
        assert proc(attn, hidden).shape == hidden.shape

    def test_injection_inactive_before_layer_start(self):
        """No injection when layer_idx < layer_start."""
        from attn_texture.baselines.masactrl_runner import _MasaCtrlAttnProcessor

        proc = _MasaCtrlAttnProcessor(layer_idx=5, step_start=4, layer_start=10)
        proc._cur_step = 4
        attn = self._make_attn()
        hidden = torch.randn(4, self._SEQ, self._DIM)
        assert proc(attn, hidden).shape == hidden.shape

    def test_non_4_batch_skips_injection(self):
        """Batch size != 4 means source/target layout is undefined; no injection."""
        from attn_texture.baselines.masactrl_runner import _MasaCtrlAttnProcessor

        proc = _MasaCtrlAttnProcessor(layer_idx=10, step_start=0, layer_start=0)
        proc._cur_step = 0
        attn = self._make_attn(batch=2)
        hidden = torch.randn(2, self._SEQ, self._DIM)
        assert proc(attn, hidden).shape == hidden.shape


# ---------------------------------------------------------------------------
# TestMasaCtrlRunnerConstruction
# ---------------------------------------------------------------------------


class TestMasaCtrlRunnerConstruction:
    def test_construction_succeeds(self, runner):
        from attn_texture.baselines.masactrl_runner import MasaCtrlRunner

        assert isinstance(runner, MasaCtrlRunner)

    def test_with_isolated_model_called_during_run(self, runner, source_image):
        """run() must delegate to with_isolated_model — not bypass it."""
        with patch(
            "attn_texture.baselines.masactrl_runner.with_isolated_model",
            wraps=with_isolated_model,
        ) as spy:
            _run(runner, source_image)
        spy.assert_called_once()

    def test_models_not_stored_as_direct_attributes(
        self, unet, vae, scheduler, tokenizer, text_encoder, cfg
    ):
        """Runner must not hold model objects as top-level instance attributes."""
        from attn_texture.baselines.masactrl_runner import MasaCtrlRunner

        r = MasaCtrlRunner(unet, vae, scheduler, tokenizer, text_encoder, cfg)
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

    def test_encode_receives_normalized_pixels(self, runner, vae, source_image):
        """VAE expects pixels in [-1, 1]."""
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

    def test_denoising_loop_runs_num_steps(self, runner, scheduler, source_image):
        """scheduler.step must be called at least num_inference_steps times for target path."""
        _run(runner, source_image)
        assert scheduler.step.call_count >= _NUM_STEPS


# ---------------------------------------------------------------------------
# TestAttnProcessorRegistration
# ---------------------------------------------------------------------------


class TestAttnProcessorRegistration:
    def test_set_attn_processor_called_once(self, runner, unet, source_image):
        _run(runner, source_image)
        unet.set_attn_processor.assert_called_once()

    def test_full_processor_dict_has_all_layers(self, runner, unet, source_image):
        """set_attn_processor receives the complete dict (attn1 + attn2)."""
        _run(runner, source_image)
        args, _ = unet.set_attn_processor.call_args
        processors = args[0]
        assert len(processors) == len(_ATTN_LAYER_NAMES)

    def test_self_attn_layers_are_masactrl_type(self, runner, unet, source_image):
        """Only attn1 (self-attention) layers get _MasaCtrlAttnProcessor."""
        from attn_texture.baselines.masactrl_runner import _MasaCtrlAttnProcessor

        _run(runner, source_image)
        args, _ = unet.set_attn_processor.call_args
        processors = args[0]
        for name in _SELF_ATTN_NAMES:
            assert isinstance(processors[name], _MasaCtrlAttnProcessor), (
                f"Self-attn {name!r} should be _MasaCtrlAttnProcessor"
            )

    def test_cross_attn_layers_keep_original_processor(self, runner, unet, source_image):
        """attn2 (cross-attention) layers keep the original MagicMock processor."""
        from attn_texture.baselines.masactrl_runner import _MasaCtrlAttnProcessor

        _run(runner, source_image)
        args, _ = unet.set_attn_processor.call_args
        processors = args[0]
        for name in _ATTN_LAYER_NAMES:
            if "attn2" in name:
                assert not isinstance(processors[name], _MasaCtrlAttnProcessor), (
                    f"Cross-attn {name!r} must NOT be _MasaCtrlAttnProcessor"
                )

    def test_layer_indices_are_sequential(self, runner, unet, source_image):
        """Self-attention processors have sequential layer_idx values 0, 1, 2, …"""
        from attn_texture.baselines.masactrl_runner import _MasaCtrlAttnProcessor

        _run(runner, source_image)
        args, _ = unet.set_attn_processor.call_args
        processors = args[0]
        masa_procs = [p for p in processors.values() if isinstance(p, _MasaCtrlAttnProcessor)]
        layer_indices = sorted(p.layer_idx for p in masa_procs)
        assert layer_indices == list(range(len(_SELF_ATTN_NAMES)))
