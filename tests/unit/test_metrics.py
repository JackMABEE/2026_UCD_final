"""Tests for eval/metrics.py — image quality metrics over PIL images.

All metrics accept a pair of RGB PIL images and return a float.
Tests use tiny (32×32) synthetic images for speed — no disk I/O.

Contract under test
-------------------
ssim(img_a, img_b)       → float in [-1, 1]; identical images → 1.0
psnr(img_a, img_b)       → float ≥ 0 or +inf; identical images → inf
lpips_score(img_a, img_b) → float ≥ 0; identical images ≈ 0.0

All three raise ValueError when image spatial dimensions differ.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from attn_texture.eval.metrics import clip_score, dino_distance, lpips_score, psnr, ssim

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_W, _H = 32, 32


def _solid(r: int, g: int, b: int) -> Image.Image:
    return Image.new("RGB", (_W, _H), color=(r, g, b))


def _checkerboard() -> Image.Image:
    """High-frequency pattern with strong edges — gives non-trivial SSIM/PSNR."""
    arr = np.zeros((_H, _W, 3), dtype=np.uint8)
    arr[::2, ::2] = 255
    arr[1::2, 1::2] = 255
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# TestSsim
# ---------------------------------------------------------------------------


class TestSsim:
    def test_returns_float(self):
        img = _solid(128, 64, 32)
        assert isinstance(ssim(img, img), float)

    def test_identical_images_return_one(self):
        img = _checkerboard()
        assert ssim(img, img) == pytest.approx(1.0, abs=1e-5)

    def test_identical_solid_returns_one(self):
        img = _solid(200, 100, 50)
        assert ssim(img, img) == pytest.approx(1.0, abs=1e-5)

    def test_different_images_below_one(self):
        a = _solid(0, 0, 0)
        b = _solid(255, 255, 255)
        assert ssim(a, b) < 1.0

    def test_score_in_valid_range(self):
        a = _solid(100, 150, 200)
        b = _solid(50, 100, 150)
        score = ssim(a, b)
        assert -1.0 <= score <= 1.0

    def test_more_similar_pair_has_higher_ssim(self):
        ref = _solid(128, 128, 128)
        close = _solid(130, 130, 130)
        far = _solid(200, 200, 200)
        assert ssim(ref, close) > ssim(ref, far)

    def test_size_mismatch_raises_value_error(self):
        a = Image.new("RGB", (_W, _H))
        b = Image.new("RGB", (_W * 2, _H * 2))
        with pytest.raises(ValueError):
            ssim(a, b)


# ---------------------------------------------------------------------------
# TestPsnr
# ---------------------------------------------------------------------------


class TestPsnr:
    def test_returns_float(self):
        img = _solid(128, 64, 32)
        result = psnr(img, img)
        assert isinstance(result, float)

    def test_identical_images_return_inf(self):
        img = _checkerboard()
        assert math.isinf(psnr(img, img))

    def test_different_images_finite_non_negative(self):
        # Black vs white: MSE = 255² fills the data range → PSNR = 0.0 dB (valid)
        a = _solid(0, 0, 0)
        b = _solid(255, 255, 255)
        score = psnr(a, b)
        assert math.isfinite(score)
        assert score >= 0.0

    def test_more_similar_images_higher_psnr(self):
        ref = _solid(128, 128, 128)
        close = _solid(130, 130, 130)
        far = _solid(200, 200, 200)
        assert psnr(ref, close) > psnr(ref, far)

    def test_size_mismatch_raises_value_error(self):
        a = Image.new("RGB", (_W, _H))
        b = Image.new("RGB", (_W * 2, _H * 2))
        with pytest.raises(ValueError):
            psnr(a, b)


# ---------------------------------------------------------------------------
# TestLpipsScore
# ---------------------------------------------------------------------------


class TestLpipsScore:
    def test_returns_float(self):
        img = _solid(100, 150, 200)
        assert isinstance(lpips_score(img, img), float)

    def test_identical_images_near_zero(self):
        img = _checkerboard()
        assert lpips_score(img, img) == pytest.approx(0.0, abs=0.05)

    def test_non_negative(self):
        a = _solid(0, 0, 0)
        b = _solid(255, 255, 255)
        assert lpips_score(a, b) >= 0.0

    def test_different_images_greater_than_identical(self):
        ref = _checkerboard()
        other = _solid(200, 50, 100)
        assert lpips_score(ref, other) > lpips_score(ref, ref)

    def test_size_mismatch_raises_value_error(self):
        a = Image.new("RGB", (_W, _H))
        b = Image.new("RGB", (_W * 2, _H * 2))
        with pytest.raises(ValueError):
            lpips_score(a, b)


# ---------------------------------------------------------------------------
# TestClipScore
# ---------------------------------------------------------------------------


class TestClipScore:
    """clip_score(image, prompt) — CLIP cosine similarity, mocked for speed."""

    _EMBED_DIM = 4

    @pytest.fixture(autouse=True)
    def inject_clip(self):
        """Inject stub CLIP model/processor into module globals; restore after."""
        import attn_texture.eval.metrics as _m

        mock_proc = MagicMock()
        mock_proc.return_value = {
            "pixel_values": torch.zeros(1, 3, 224, 224),
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }

        # CLIPOutput returned by model(**inputs, return_dict=True)
        _aligned = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        mock_output = MagicMock()
        mock_output.image_embeds = _aligned.clone()
        mock_output.text_embeds = _aligned.clone()

        mock_model = MagicMock()
        mock_model.return_value = mock_output

        old_model, old_proc = _m._clip_model, _m._clip_processor
        _m._clip_model = mock_model
        _m._clip_processor = mock_proc
        yield mock_model, mock_proc
        _m._clip_model = old_model
        _m._clip_processor = old_proc

    def test_returns_float(self):
        assert isinstance(clip_score(_solid(128, 64, 32), "a photo"), float)

    def test_in_valid_range(self):
        result = clip_score(_solid(100, 100, 100), "a fabric texture")
        assert -1.0 <= result <= 1.0 + 1e-6

    def test_aligned_embeddings_return_one(self):
        """Identical unit-vector image and text embeddings → cosine = 1.0."""
        result = clip_score(_solid(200, 200, 200), "any prompt")
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_embeddings_return_zero(self, inject_clip):
        mock_model, _ = inject_clip
        mock_output = MagicMock()
        mock_output.image_embeds = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        mock_output.text_embeds = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        mock_model.return_value = mock_output
        result = clip_score(_solid(0, 0, 0), "orthogonal prompt")
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_opposite_embeddings_return_minus_one(self, inject_clip):
        mock_model, _ = inject_clip
        mock_output = MagicMock()
        mock_output.image_embeds = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        mock_output.text_embeds = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
        mock_model.return_value = mock_output
        result = clip_score(_solid(0, 0, 0), "opposite")
        assert result == pytest.approx(-1.0, abs=1e-4)

    def test_model_called_once_with_all_inputs(self, inject_clip):
        """Single combined forward pass — no separate image/text feature calls."""
        mock_model, mock_proc = inject_clip
        clip_score(_solid(100, 100, 100), "silk fabric")
        mock_proc.assert_called_once()
        mock_model.assert_called_once()


# ---------------------------------------------------------------------------
# TestDinoDistance
# ---------------------------------------------------------------------------


class TestDinoDistance:
    """dino_distance(img_a, img_b) — DINO CLS-token distance, mocked for speed."""

    _EMBED_DIM = 4
    _SEQ_LEN = 5  # CLS + 4 patches

    @staticmethod
    def _cls_hidden(vec: list[float]) -> torch.Tensor:
        """Build a (1, _SEQ_LEN, _EMBED_DIM) hidden-state tensor with CLS = vec."""
        h = torch.zeros(1, 5, 4)
        h[0, 0] = torch.tensor(vec)
        return h

    @pytest.fixture(autouse=True)
    def inject_dino(self):
        """Inject stub DINO model/extractor; restore after."""
        import attn_texture.eval.metrics as _m

        mock_ext = MagicMock()
        mock_ext.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        mock_model = MagicMock()
        # Default: both calls return the same CLS embedding → distance = 0
        mock_model.return_value.last_hidden_state = self._cls_hidden([1.0, 0.0, 0.0, 0.0])

        old_model, old_ext = _m._dino_model, _m._dino_extractor
        _m._dino_model  = mock_model
        _m._dino_extractor = mock_ext
        yield mock_model, mock_ext
        _m._dino_model = old_model
        _m._dino_extractor = old_ext

    def test_returns_float(self):
        assert isinstance(dino_distance(_solid(0, 0, 0), _solid(255, 255, 255)), float)

    def test_non_negative(self):
        result = dino_distance(_solid(100, 100, 100), _solid(200, 200, 200))
        assert result >= 0.0

    def test_identical_cls_tokens_return_zero(self):
        """Same embedding from both calls → cosine = 1 → distance = 0."""
        result = dino_distance(_solid(128, 128, 128), _solid(128, 128, 128))
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_orthogonal_cls_tokens_return_one(self, inject_dino):
        """Orthogonal embeddings → cosine = 0 → distance = 1."""
        mock_model, _ = inject_dino
        h_a = self._cls_hidden([1.0, 0.0, 0.0, 0.0])
        h_b = self._cls_hidden([0.0, 1.0, 0.0, 0.0])
        mock_model.side_effect = [
            MagicMock(last_hidden_state=h_a),
            MagicMock(last_hidden_state=h_b),
        ]
        result = dino_distance(_solid(0, 0, 0), _solid(255, 255, 255))
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_opposite_cls_tokens_return_two(self, inject_dino):
        """Anti-parallel embeddings → cosine = -1 → distance = 2."""
        mock_model, _ = inject_dino
        h_a = self._cls_hidden([1.0, 0.0, 0.0, 0.0])
        h_b = self._cls_hidden([-1.0, 0.0, 0.0, 0.0])
        mock_model.side_effect = [
            MagicMock(last_hidden_state=h_a),
            MagicMock(last_hidden_state=h_b),
        ]
        result = dino_distance(_solid(0, 0, 0), _solid(255, 255, 255))
        assert result == pytest.approx(2.0, abs=1e-4)

    def test_model_called_twice_once_per_image(self, inject_dino):
        mock_model, _ = inject_dino
        dino_distance(_solid(100, 100, 100), _solid(200, 200, 200))
        assert mock_model.call_count == 2
