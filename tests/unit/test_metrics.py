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

import numpy as np
import pytest
from PIL import Image

from attn_texture.eval.metrics import lpips_score, psnr, ssim

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
