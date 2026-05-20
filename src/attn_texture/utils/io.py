"""Image and tensor I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

PathLike = Union[str, Path]


def load_image(path: PathLike, *, size: tuple[int, int] | None = None) -> Image.Image:
    """Load an image from disk and return it as an RGB PIL Image.

    Args:
        path: file path (str or Path).
        size: optional (width, height) to resize to after loading.

    Returns:
        RGB PIL Image.

    Raises:
        FileNotFoundError: if *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return img


def save_image(image: Union[Image.Image, torch.Tensor], path: PathLike) -> None:
    """Save a PIL Image or float CHW tensor to *path*, creating parent dirs.

    Args:
        image: PIL Image or float32 tensor with shape (3, H, W) in [0, 1].
        path: destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        image = tensor_to_pil(image)
    image.save(path)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL Image to a float32 CHW tensor in [0, 1].

    Args:
        image: RGB PIL Image of any size.

    Returns:
        Tensor of shape (3, H, W), dtype float32, values in [0.0, 1.0].
    """
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a float32 CHW tensor in [0, 1] to an RGB PIL Image.

    Args:
        tensor: shape (3, H, W), float32, values clamped to [0, 1].

    Returns:
        RGB PIL Image.
    """
    arr = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
