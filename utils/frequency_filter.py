"""
2D-FFT based spatial frequency filter for latent tensors.

Supports three modes:
  "low"    — keep only low-frequency components (smooth structure).
  "high"   — keep only high-frequency components (edges / detail).
  "hybrid" — weighted sum of both: alpha * high + (1-alpha) * low.

The filter operates on tensors of shape (B, C, H, W) and is differentiable
(torch.fft is used throughout), so it can be inserted into a training loop.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn


FilterMode = Literal["low", "high", "hybrid"]


class FrequencyFilter(nn.Module):
    """
    Differentiable 2D FFT-based low/high/hybrid pass filter.

    Parameters
    ----------
    low_pass_radius : float
        Fraction of the spatial dimension used as the radius of the
        circular low-pass mask  (0.0 < r ≤ 1.0).
    mode : str
        ``"low"`` | ``"high"`` | ``"hybrid"``.
    high_freq_weight : float
        Weight of the high-freq component when ``mode == "hybrid"``.
        The low-freq weight is ``1 - high_freq_weight``.
    """

    def __init__(
        self,
        low_pass_radius: float = 0.25,
        mode: FilterMode = "hybrid",
        high_freq_weight: float = 0.5,
    ):
        super().__init__()
        if not (0.0 < low_pass_radius <= 1.0):
            raise ValueError("low_pass_radius must be in (0, 1].")
        self.low_pass_radius = low_pass_radius
        self.mode = mode
        self.high_freq_weight = high_freq_weight

        # Cache the mask to avoid recomputing for each call with same shape.
        self._cached_shape: Optional[tuple] = None
        self._cached_mask: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency filter to a latent tensor.

        Parameters
        ----------
        x : Tensor of shape (B, C, H, W)

        Returns
        -------
        Tensor of the same shape, dtype, and device as *x*.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4-D input (B,C,H,W), got shape {x.shape}.")

        B, C, H, W = x.shape
        low = self._low_pass(x, H, W)

        if self.mode == "low":
            return low
        elif self.mode == "high":
            return x - low
        else:  # "hybrid"
            high = x - low
            return self.high_freq_weight * high + (1.0 - self.high_freq_weight) * low

    # ------------------------------------------------------------------
    # Low-pass implementation
    # ------------------------------------------------------------------

    def _low_pass(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        mask = self._get_mask(H, W, x.device, x.dtype)

        # Transform to frequency domain (2D FFT over spatial dims).
        x_f = torch.fft.rfft2(x, norm="ortho")          # (B, C, H, W//2+1) complex

        # The mask was built for rfft2 output shape.
        x_f_filtered = x_f * mask.unsqueeze(0).unsqueeze(0)

        # Transform back.
        x_low = torch.fft.irfft2(x_f_filtered, s=(H, W), norm="ortho")
        return x_low.to(x.dtype)

    # ------------------------------------------------------------------
    # Mask construction (cached per spatial shape)
    # ------------------------------------------------------------------

    def _get_mask(
        self, H: int, W: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        shape = (H, W, str(device))
        if self._cached_shape == shape and self._cached_mask is not None:
            return self._cached_mask.to(device=device, dtype=dtype)

        # Build a 2D boolean circular mask in the DC-centered frequency plane,
        # then rearrange to match rfft2 layout (DC at corner, not center).
        W_rfft = W // 2 + 1  # rfft2 output width

        # Frequency coordinates in the full (H x W) DFT grid.
        fy = torch.fft.fftfreq(H)   # shape (H,)
        fx = torch.fft.rfftfreq(W)  # shape (W_rfft,) — only non-negative freqs

        # Meshgrid: rows=fy, cols=fx
        grid_fy, grid_fx = torch.meshgrid(fy, fx, indexing="ij")  # (H, W_rfft)
        dist = torch.sqrt(grid_fy ** 2 + grid_fx ** 2)            # Euclidean freq radius

        # Radius threshold: low_pass_radius is a fraction of 0.5 (Nyquist = 0.5).
        r_thresh = self.low_pass_radius * 0.5
        mask = (dist <= r_thresh).to(dtype=dtype, device=device)

        self._cached_shape = shape
        self._cached_mask = mask
        return mask

    # ------------------------------------------------------------------
    # Convenience: decompose a tensor into (low, high) components
    # ------------------------------------------------------------------

    def decompose(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (low_freq, high_freq) tensors that sum to *x*."""
        B, C, H, W = x.shape
        low = self._low_pass(x, H, W)
        return low, x - low

    # ------------------------------------------------------------------
    # Extra: combine pre-computed low/high with configurable weights
    # ------------------------------------------------------------------

    @staticmethod
    def combine(
        low: torch.Tensor,
        high: torch.Tensor,
        high_weight: float = 0.5,
    ) -> torch.Tensor:
        """Merge low and high frequency tensors with given high_weight."""
        return high_weight * high + (1.0 - high_weight) * low
