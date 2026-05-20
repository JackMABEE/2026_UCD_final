"""Tests for utils/io.py — image/tensor read-write helpers."""

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from attn_texture.utils.io import load_image, save_image, tensor_to_pil, pil_to_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_png(tmp_path: Path) -> Path:
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    p = tmp_path / "test.png"
    img.save(p)
    return p


@pytest.fixture()
def rgb_pil() -> Image.Image:
    arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_returns_pil_image(self, tmp_png):
        img = load_image(tmp_png)
        assert isinstance(img, Image.Image)

    def test_returns_rgb(self, tmp_png):
        img = load_image(tmp_png)
        assert img.mode == "RGB"

    def test_accepts_str_path(self, tmp_png):
        img = load_image(str(tmp_png))
        assert isinstance(img, Image.Image)

    def test_resize_kwarg(self, tmp_png):
        img = load_image(tmp_png, size=(32, 32))
        assert img.size == (32, 32)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "nonexistent.png")


# ---------------------------------------------------------------------------
# save_image
# ---------------------------------------------------------------------------


class TestSaveImage:
    def test_saves_pil_image(self, tmp_path, rgb_pil):
        dest = tmp_path / "out.png"
        save_image(rgb_pil, dest)
        assert dest.exists()

    def test_saved_file_is_readable(self, tmp_path, rgb_pil):
        dest = tmp_path / "out.png"
        save_image(rgb_pil, dest)
        loaded = Image.open(dest)
        assert loaded.size == rgb_pil.size

    def test_creates_parent_dirs(self, tmp_path, rgb_pil):
        dest = tmp_path / "nested" / "dir" / "out.png"
        save_image(rgb_pil, dest)
        assert dest.exists()

    def test_saves_tensor_chw_float(self, tmp_path):
        # Float tensor in [0, 1], shape (3, H, W)
        t = torch.rand(3, 64, 64)
        dest = tmp_path / "tensor.png"
        save_image(t, dest)
        assert dest.exists()


# ---------------------------------------------------------------------------
# tensor_to_pil / pil_to_tensor
# ---------------------------------------------------------------------------


class TestTensorPilConversions:
    def test_pil_to_tensor_shape(self, rgb_pil):
        t = pil_to_tensor(rgb_pil)
        assert t.shape == (3, 64, 64)

    def test_pil_to_tensor_dtype_float(self, rgb_pil):
        t = pil_to_tensor(rgb_pil)
        assert t.dtype == torch.float32

    def test_pil_to_tensor_range(self, rgb_pil):
        t = pil_to_tensor(rgb_pil)
        assert t.min() >= 0.0
        assert t.max() <= 1.0

    def test_tensor_to_pil_returns_pil(self):
        t = torch.rand(3, 64, 64)
        img = tensor_to_pil(t)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"

    def test_tensor_to_pil_size(self):
        t = torch.rand(3, 48, 32)
        img = tensor_to_pil(t)
        assert img.size == (32, 48)  # PIL size is (W, H)

    def test_roundtrip_is_close(self, rgb_pil):
        t = pil_to_tensor(rgb_pil)
        recovered = tensor_to_pil(t)
        # Pixel values should survive uint8 round-trip within rounding error
        orig = np.array(rgb_pil).astype(np.float32)
        rec = np.array(recovered).astype(np.float32)
        assert np.abs(orig - rec).max() <= 1.0  # at most 1 gray level off
