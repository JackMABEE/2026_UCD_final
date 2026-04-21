"""
Perceptual and structural image quality metrics.

Dependencies
------------
  pip install scikit-image lpips torch torchvision

All methods accept either:
  - PIL.Image objects, or
  - torch.Tensor of shape (C, H, W) in [0, 1], or
  - numpy.ndarray of shape (H, W, C) in [0, 1]

Results are returned as plain Python floats for easy JSON serialisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


ImageLike = Union[Image.Image, torch.Tensor, np.ndarray]


def _to_numpy_hwc_uint8(img: ImageLike) -> np.ndarray:
    """Normalise any supported image type to HWC uint8 numpy array."""
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu().float()
        if t.ndim == 3:          # C, H, W  → H, W, C
            t = t.permute(1, 2, 0)
        t = t.clamp(0.0, 1.0)
        return (t.numpy() * 255).astype(np.uint8)
    if isinstance(img, np.ndarray):
        if img.dtype != np.uint8:
            img = (img.clip(0.0, 1.0) * 255).astype(np.uint8)
        return img
    raise TypeError(f"Unsupported image type: {type(img)}")


def _to_torch_chw_float(img: ImageLike) -> torch.Tensor:
    """Return (C, H, W) float32 tensor in [0, 1]."""
    if isinstance(img, Image.Image):
        return TF.to_tensor(img.convert("RGB"))
    if isinstance(img, torch.Tensor):
        t = img.detach().cpu().float()
        if t.ndim == 3 and t.shape[0] in (1, 3, 4):
            return t[:3].clamp(0.0, 1.0)
        if t.ndim == 3:  # HWC
            return t.permute(2, 0, 1)[:3].clamp(0.0, 1.0)
        raise ValueError(f"Cannot convert tensor of shape {t.shape} to CHW.")
    if isinstance(img, np.ndarray):
        arr = img.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)[:3]
    raise TypeError(f"Unsupported image type: {type(img)}")


class ImageMetrics:
    """
    Computes SSIM, PSNR, and LPIPS between a source and generated image.

    Parameters
    ----------
    lpips_net : str
        Backbone for LPIPS: ``"alex"`` (fastest), ``"vgg"``, ``"squeeze"``.
    device : str | torch.device
        Device for LPIPS inference.
    """

    def __init__(self, lpips_net: str = "alex", device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        self._lpips_model = self._load_lpips(lpips_net)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_all(
        self, source: ImageLike, generated: ImageLike
    ) -> Dict[str, float]:
        """
        Compute all metrics between *source* and *generated*.

        Returns
        -------
        dict with keys: ``ssim``, ``psnr``, ``lpips``
        """
        return {
            "ssim": self.ssim(source, generated),
            "psnr": self.psnr(source, generated),
            "lpips": self.lpips(source, generated),
        }

    def ssim(self, source: ImageLike, generated: ImageLike) -> float:
        """Structural Similarity Index (higher is better, max 1.0)."""
        src = _to_numpy_hwc_uint8(source)
        gen = _to_numpy_hwc_uint8(generated)
        # Match spatial size if needed.
        gen = self._resize_numpy(gen, src.shape[:2])
        score = _ssim(src, gen, channel_axis=2, data_range=255)
        return float(score)

    def psnr(self, source: ImageLike, generated: ImageLike) -> float:
        """Peak Signal-to-Noise Ratio in dB (higher is better)."""
        src = _to_numpy_hwc_uint8(source)
        gen = _to_numpy_hwc_uint8(generated)
        gen = self._resize_numpy(gen, src.shape[:2])
        score = _psnr(src, gen, data_range=255)
        return float(score)

    def lpips(self, source: ImageLike, generated: ImageLike) -> float:
        """
        Learned Perceptual Image Patch Similarity (lower is better, min 0.0).
        """
        if self._lpips_model is None:
            return float("nan")

        src_t = _to_torch_chw_float(source).unsqueeze(0).to(self.device)
        gen_t = _to_torch_chw_float(generated).unsqueeze(0).to(self.device)

        # LPIPS expects inputs in [-1, 1].
        src_t = src_t * 2.0 - 1.0
        gen_t = gen_t * 2.0 - 1.0

        # Resize generated to match source spatial dims if necessary.
        if src_t.shape[-2:] != gen_t.shape[-2:]:
            import torch.nn.functional as F
            gen_t = F.interpolate(gen_t, size=src_t.shape[-2:], mode="bilinear", align_corners=False)

        with torch.no_grad():
            score = self._lpips_model(src_t, gen_t)
        return float(score.item())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_numpy(arr: np.ndarray, hw: tuple) -> np.ndarray:
        """Resize HWC uint8 array to target (H, W) if sizes differ."""
        if arr.shape[:2] == tuple(hw):
            return arr
        pil = Image.fromarray(arr).resize((hw[1], hw[0]), Image.LANCZOS)
        return np.array(pil)

    @staticmethod
    def _load_lpips(net: str):
        try:
            import lpips
            model = lpips.LPIPS(net=net)
            model.eval()
            return model
        except ImportError:
            import warnings
            warnings.warn(
                "lpips package not found. LPIPS metric will return NaN. "
                "Install with: pip install lpips",
                stacklevel=3,
            )
            return None
