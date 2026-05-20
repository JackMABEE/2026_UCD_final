"""Image quality metrics over PIL images.

All three functions accept an (original, generated) pair of RGB PIL images
and return a scalar float.  They are used by eval/shootout.py to compare
each baseline/ours output against the source reference.

Implementations
---------------
ssim        — scikit-image structural_similarity, channel_axis=2, data_range=255
psnr        — scikit-image peak_signal_noise_ratio, data_range=255
lpips_score — lpips.LPIPS(net="alex"); singleton loaded once per process
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import lpips as _lpips_lib

# Module-level singleton so the AlexNet weights are loaded only once.
_lpips_model: "_lpips_lib.LPIPS | None" = None


def _get_lpips_model() -> "_lpips_lib.LPIPS":
    global _lpips_model
    if _lpips_model is None:
        logger.debug("Loading LPIPS AlexNet weights (first call only)…")
        _lpips_model = _lpips_lib.LPIPS(net="alex", verbose=False)
        _lpips_model.eval()
    return _lpips_model


def _to_numpy_rgb(img: Image.Image) -> np.ndarray:
    """Convert PIL image to uint8 HWC numpy array."""
    return np.array(img.convert("RGB"))


def _check_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"Images must have identical spatial dimensions, "
            f"got {a.shape[:2]} vs {b.shape[:2]}"
        )


def ssim(img_a: Image.Image, img_b: Image.Image) -> float:
    """Structural Similarity Index between two RGB PIL images.

    Args:
        img_a: reference image.
        img_b: comparison image; must have the same spatial dimensions as img_a.

    Returns:
        SSIM score in [-1, 1]; identical images return 1.0.

    Raises:
        ValueError: if the images differ in spatial dimensions.
    """
    a, b = _to_numpy_rgb(img_a), _to_numpy_rgb(img_b)
    _check_shape(a, b)
    score, _ = structural_similarity(a, b, channel_axis=2, data_range=255, full=True)
    return float(score)


def psnr(img_a: Image.Image, img_b: Image.Image) -> float:
    """Peak Signal-to-Noise Ratio between two RGB PIL images (dB).

    Args:
        img_a: reference image.
        img_b: comparison image; must have the same spatial dimensions as img_a.

    Returns:
        PSNR in dB; returns float("inf") for identical images.

    Raises:
        ValueError: if the images differ in spatial dimensions.
    """
    a, b = _to_numpy_rgb(img_a), _to_numpy_rgb(img_b)
    _check_shape(a, b)
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def lpips_score(img_a: Image.Image, img_b: Image.Image) -> float:
    """Learned Perceptual Image Patch Similarity (AlexNet backbone).

    Args:
        img_a: reference image.
        img_b: comparison image; must have the same spatial dimensions as img_a.

    Returns:
        LPIPS distance ≥ 0; near 0 for identical images.

    Raises:
        ValueError: if the images differ in spatial dimensions.
    """
    a_np, b_np = _to_numpy_rgb(img_a), _to_numpy_rgb(img_b)
    _check_shape(a_np, b_np)

    def _to_lpips_tensor(arr: np.ndarray) -> torch.Tensor:
        # lpips expects (1, 3, H, W) float32 in [-1, 1]
        t = torch.from_numpy(arr).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
        return t

    a_t = _to_lpips_tensor(a_np)
    b_t = _to_lpips_tensor(b_np)

    model = _get_lpips_model()
    with torch.no_grad():
        dist = model(a_t, b_t)
    return float(dist.item())
