"""Device and dtype utilities.

All code must go through get_device_and_dtype() and safe_to() instead of
hardcoding .cuda() / torch.float16 literals. See CLAUDE.md §6 rules 2–3.
"""

from __future__ import annotations

import platform
from typing import Tuple

import torch


def _macos_major_version() -> int:
    """Return the macOS major version (e.g. 15 for Sequoia), or 0 on non-Mac."""
    ver_str, _, _ = platform.mac_ver()
    if not ver_str:
        return 0
    try:
        return int(ver_str.split(".")[0])
    except ValueError:
        return 0


def get_device_and_dtype() -> Tuple[str, torch.dtype]:
    """Detect the best available backend and a safe default dtype.

    Priority: CUDA > MPS > CPU.

    Returns:
        (device_str, dtype): e.g. ("mps", torch.float32)
    """
    if torch.cuda.is_available():
        return "cuda", torch.float16

    if torch.backends.mps.is_available():
        # bfloat16 is supported on MPS from macOS 14 (Sonoma) onward.
        # fp16 is intentionally never returned here — it causes NaN on MPS (CLAUDE.md §7 rule 1).
        dtype = torch.bfloat16 if _macos_major_version() >= 14 else torch.float32
        return "mps", dtype

    return "cpu", torch.float32


def resolve_dtype(device: str, requested: torch.dtype) -> torch.dtype:
    """Return a dtype that is safe on *device*, potentially downgrading the request.

    fp16 on MPS triggers NaN / black-image overflows in the U-Net (CLAUDE.md §7 rule 1).
    All other combinations pass through unchanged.

    Args:
        device: target device string ("cuda", "mps", "cpu", ...)
        requested: caller's desired dtype

    Returns:
        A dtype that is safe to use on the given device.
    """
    if device == "mps" and requested == torch.float16:
        return torch.float32
    return requested


def safe_to(tensor: torch.Tensor, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Move *tensor* to *device* with a device-safe dtype.

    Calls resolve_dtype first so callers never accidentally put fp16 on MPS.

    Args:
        tensor: source tensor
        device: target device string
        dtype: requested dtype (will be silently downgraded if unsafe)

    Returns:
        Tensor on *device* with the resolved dtype.
    """
    safe_dtype = resolve_dtype(device, dtype)
    return tensor.to(device=device, dtype=safe_dtype)
