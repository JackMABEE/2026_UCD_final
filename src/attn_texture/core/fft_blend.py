"""Dual-domain FFT frequency blending.

Fuses the source latent's low frequencies (global lighting / structure) with
the generated latent's high frequencies (new texture detail). All computation
stays on GPU/MPS via torch.fft — no NumPy on the hot path (CLAUDE.md §6 rule 8).

Public API
----------
blend_latents       — global blend, whole spatial extent
blend_latents_local — blend only inside a spatial mask (Phase 2)
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_low_freq_mask(
    H: int,
    W: int,
    cutoff_ratio: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a (1, 1, H, W) float mask that is 1 inside the low-freq circle.

    The mask is defined in *shifted* frequency space (DC at centre), so it can
    be applied directly after fftshift. The cutoff circle is parameterised as a
    fraction of the shortest half-dimension, giving a normalised radius in [0, 1].

    Args:
        H, W: spatial dimensions of the latent.
        cutoff_ratio: fraction of the frequency radius marked as low-freq [0, 1].
        device: target device for the mask tensor.
        dtype: float dtype matching the latent.

    Returns:
        Boolean-valued float tensor, shape (1, 1, H, W).
    """
    # Normalised frequency coordinates — each axis spans [-1, 1] after shift.
    fy = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype)
    fx = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")  # (H, W)
    radius = (grid_y**2 + grid_x**2).sqrt()                 # Euclidean norm
    mask = (radius <= cutoff_ratio).to(dtype)                # 1 inside, 0 outside
    return mask.unsqueeze(0).unsqueeze(0)                    # (1, 1, H, W)


def _validate_inputs(src: torch.Tensor, gen: torch.Tensor, cutoff_ratio: float) -> None:
    if src.shape != gen.shape:
        raise ValueError(
            f"src and gen must have the same shape, got {src.shape} vs {gen.shape}"
        )
    if not (0.0 <= cutoff_ratio <= 1.0):
        raise ValueError(f"cutoff_ratio must be in [0, 1], got {cutoff_ratio}")


def _normalize_spatial_mask(
    spatial_mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Broadcast *spatial_mask* to (B, 1, H, W) matching *reference*."""
    m = spatial_mask
    if m.dim() == 2:                    # (H, W)  → (1, 1, H, W)
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:                  # (1, H, W) → (1, 1, H, W)
        m = m.unsqueeze(0)
    # Now m is (B, 1, H, W) or broadcastable; move to same device/dtype as ref
    return m.to(device=reference.device, dtype=reference.dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def blend_latents(
    src: torch.Tensor,
    gen: torch.Tensor,
    cutoff_ratio: float = 0.5,
) -> torch.Tensor:
    """Blend *src* and *gen* in the frequency domain across the full spatial extent.

    Keeps *src*'s low-frequency components (global lighting, coarse structure)
    and *gen*'s high-frequency components (fine texture detail). § 3.2 Eq.(1).

    Args:
        src: source latent, shape (B, C, H, W), float32.
        gen: generated latent, same shape as *src*.
        cutoff_ratio: radius of the low-freq circle as a fraction of max freq [0, 1].
                      0 → take everything from gen; 1 → take everything from src.

    Returns:
        Blended latent, same shape and dtype as *src*.
    """
    _validate_inputs(src, gen, cutoff_ratio)

    # Short-circuit at the hard boundaries to avoid circular-mask corner gaps.
    # The unit-circle mask has radius √2 at the corners, so cutoff_ratio=1.0
    # would leave the four high-freq corners from gen rather than taking all of src.
    if cutoff_ratio <= 0.0:
        return gen.clone()
    if cutoff_ratio >= 1.0:
        return src.clone()

    H, W = src.shape[-2:]

    # Transform both latents to frequency domain and shift DC to centre.
    src_fft = torch.fft.fftshift(torch.fft.fftn(src, dim=(-2, -1)), dim=(-2, -1))
    gen_fft = torch.fft.fftshift(torch.fft.fftn(gen, dim=(-2, -1)), dim=(-2, -1))

    low_mask = _make_low_freq_mask(H, W, cutoff_ratio, src.device, src.dtype)

    # Low-freq region from src; high-freq region from gen.
    blended_fft = src_fft * low_mask + gen_fft * (1.0 - low_mask)

    # Inverse shift then IFFT; take real part (imaginary residual is numerical noise).
    blended_fft = torch.fft.ifftshift(blended_fft, dim=(-2, -1))
    return torch.fft.ifftn(blended_fft, dim=(-2, -1)).real


def blend_latents_local(
    src: torch.Tensor,
    gen: torch.Tensor,
    spatial_mask: torch.Tensor,
    cutoff_ratio: float = 0.5,
) -> torch.Tensor:
    """Frequency-blend only inside *spatial_mask*; leave the rest as *src*.

    Used in Phase 2 when the texture transfer should stay within the garment
    region extracted by cross-attention. Outside the mask the source latent is
    returned unchanged, so background and skin tones are not contaminated.

    Args:
        src: source latent, shape (B, C, H, W), float32.
        gen: generated latent, same shape as *src*.
        spatial_mask: float tensor in [0, 1] selecting the blend region.
                      Accepted shapes: (H, W), (1, H, W), or (B, 1, H, W).
                      A value of 1 means "fully blend here"; 0 means "keep src".
        cutoff_ratio: passed through to blend_latents.

    Returns:
        Latent with frequency blend applied inside the mask, src outside.
    """
    _validate_inputs(src, gen, cutoff_ratio)

    freq_blended = blend_latents(src, gen, cutoff_ratio)
    mask = _normalize_spatial_mask(spatial_mask, src)

    # Spatial composite: inside mask use freq_blended, outside use src.
    return mask * freq_blended + (1.0 - mask) * src
