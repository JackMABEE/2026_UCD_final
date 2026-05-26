"""Dual-domain FFT frequency blending.

Fuses the source latent's low frequencies (global lighting / structure) with
the generated latent's high frequencies (new texture detail). All computation
stays on GPU/MPS via torch.fft — no NumPy on the hot path (CLAUDE.md §6 rule 8).

Public API
----------
blend_latents           — global blend in latent space, whole spatial extent
blend_latents_local     — blend only inside a spatial mask (Phase 2)
blend_images_lab        — pixel-space blend: L channel only, gen chroma preserved
match_ab_histogram      — histogram-match AB chroma of blended to a reference image
global_brightness_match — scalar L-channel brightness correction (no FFT)
_rgb_to_lab             — sRGB → CIE LAB (exported for testing)
_lab_to_rgb             — CIE LAB → sRGB (exported for testing)
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


# ---------------------------------------------------------------------------
# CIE LAB colour-space helpers
# ---------------------------------------------------------------------------

# D65 illuminant reference white (IEC 61966-2-1)
_D65 = (0.9504559, 1.0000000, 1.0890578)

# sRGB ↔ CIE XYZ matrices (IEC 61966-2-1 primaries, D65)
_RGB_TO_XYZ = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
]
_XYZ_TO_RGB = [
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
]


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Convert sRGB images to CIE LAB.

    Args:
        rgb: (B, 3, H, W) float in [0, 1], sRGB gamma-encoded.

    Returns:
        lab: (B, 3, H, W) — L in [0, 100], a/b in ~[-128, 127].
    """
    # sRGB gamma decode (IEC 61966-2-1)
    lin = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )

    # Linear RGB → CIE XYZ (D65)
    mat = torch.tensor(_RGB_TO_XYZ, device=rgb.device, dtype=rgb.dtype)
    xyz = torch.einsum("ij,bjhw->bihw", mat, lin)  # (B, 3, H, W)

    # Normalise by D65 illuminant
    d65 = torch.tensor(_D65, device=rgb.device, dtype=rgb.dtype).view(1, 3, 1, 1)
    xyz = xyz / d65

    # CIE f function  §4.2 CIE 015:2004
    delta = 6.0 / 29.0
    t = xyz.clamp(min=0.0)
    f = torch.where(t > delta ** 3, t.pow(1.0 / 3.0), t / (3.0 * delta ** 2) + 4.0 / 29.0)

    L = 116.0 * f[:, 1:2] - 16.0
    a = 500.0 * (f[:, 0:1] - f[:, 1:2])
    b = 200.0 * (f[:, 1:2] - f[:, 2:3])
    return torch.cat([L, a, b], dim=1)


def _lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """Convert CIE LAB to sRGB images.

    Args:
        lab: (B, 3, H, W) — L in [0, 100], a/b in ~[-128, 127].

    Returns:
        rgb: (B, 3, H, W) float in [0, 1], clamped to valid range.
    """
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    delta = 6.0 / 29.0

    def _f_inv(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta, t.pow(3.0), 3.0 * delta ** 2 * (t - 4.0 / 29.0))

    d65 = torch.tensor(_D65, device=lab.device, dtype=lab.dtype).view(1, 3, 1, 1)
    xyz = torch.cat([_f_inv(fx), _f_inv(fy), _f_inv(fz)], dim=1) * d65  # (B, 3, H, W)

    # CIE XYZ → linear sRGB
    mat = torch.tensor(_XYZ_TO_RGB, device=lab.device, dtype=lab.dtype)
    lin = torch.einsum("ij,bjhw->bihw", mat, xyz).clamp(0.0, 1.0)

    # sRGB gamma encode
    rgb = torch.where(
        lin <= 0.0031308,
        lin * 12.92,
        1.055 * lin.pow(1.0 / 2.4) - 0.055,
    )
    return rgb.clamp(0.0, 1.0)


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


def blend_images_lab(
    src: torch.Tensor,
    gen: torch.Tensor,
    cutoff_ratio: float = 0.5,
    blend_weight: float = 0.5,
) -> torch.Tensor:
    """FFT-blend decoded RGB images in LAB space, touching only the L channel.

    For low-frequency L bins: *blend_weight* * src_L + (1 - *blend_weight*) * gen_L.
    For high-frequency L bins: gen_L (unchanged).
    A/B chroma channels: always taken from *gen* (colour palette preserved).

    Args:
        src: source RGB image, shape (B, 3, H, W), float in [0, 1].
        gen: generated RGB image, same shape as *src*.
        cutoff_ratio: low-freq radius fraction in [0, 1]. 0 → all L from gen;
                      1 → blend_weight controls entire L channel.
        blend_weight: low-freq mixing weight in [0, 1].
                      1.0 → 100 % src low-freq (original lighting fully preserved).
                      0.0 → 100 % gen low-freq (source lighting discarded).
                      0.5 → equal mix.

    Returns:
        Blended RGB image, shape (B, 3, H, W), float in [0, 1].
    """
    _validate_inputs(src, gen, cutoff_ratio)
    if src.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB input, got {src.shape[1]} channels")
    if not (0.0 <= blend_weight <= 1.0):
        raise ValueError(f"blend_weight must be in [0, 1], got {blend_weight}")

    # FFT and LAB gamma ops require float32 — ComplexHalf is not reliably supported.
    in_dtype = src.dtype
    src32 = src.to(torch.float32)
    gen32 = gen.to(torch.float32)

    src_lab = _rgb_to_lab(src32)
    gen_lab = _rgb_to_lab(gen32)

    src_L = src_lab[:, 0:1]   # (B, 1, H, W)
    gen_L = gen_lab[:, 0:1]
    gen_AB = gen_lab[:, 1:3]  # chroma stays from gen

    # cutoff_ratio=0 → low-freq mask covers nothing → all L from gen.
    if cutoff_ratio <= 0.0:
        blended_L = gen_L
    else:
        H, W = src_L.shape[-2:]
        src_fft = torch.fft.fftshift(torch.fft.fftn(src_L, dim=(-2, -1)), dim=(-2, -1))
        gen_fft = torch.fft.fftshift(torch.fft.fftn(gen_L, dim=(-2, -1)), dim=(-2, -1))

        low_mask = _make_low_freq_mask(H, W, cutoff_ratio, src_L.device, torch.float32)

        # Low-freq: weighted mix; high-freq: gen only.  # § 3.2 Eq.(1)
        mixed_low = blend_weight * src_fft + (1.0 - blend_weight) * gen_fft
        blended_fft = mixed_low * low_mask + gen_fft * (1.0 - low_mask)

        blended_fft = torch.fft.ifftshift(blended_fft, dim=(-2, -1))
        blended_L = torch.fft.ifftn(blended_fft, dim=(-2, -1)).real

    blended_lab = torch.cat([blended_L, gen_AB], dim=1)    # (B, 3, H, W)
    return _lab_to_rgb(blended_lab).to(in_dtype)


def match_ab_histogram(
    blended: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Match *blended*'s AB chroma channels to *reference*'s distribution.

    Applies per-channel histogram matching in CIE LAB colour space. The L
    channel is taken from *blended* unchanged; only the A and B channels are
    redistributed to match *reference*'s cumulative distribution.

    This corrects chroma drift introduced by FFT low-freq blending — e.g. a
    colour cast from the source image's lighting leaking into the chroma —
    without altering the luminance structure.

    Uses scikit-image's ``match_histograms`` on CPU numpy arrays per batch
    element (called once after the denoising loop, not on the hot path).

    Args:
        blended:   FFT-blended RGB image, shape (B, 3, H, W), float in [0, 1].
        reference: Target AB distribution (typically gen_raw before blending),
                   same shape as *blended*.

    Returns:
        RGB image with *blended*'s L channel and histogram-matched AB channels,
        shape (B, 3, H, W), float in [0, 1].
    """
    import numpy as np
    from skimage.exposure import match_histograms

    if blended.shape != reference.shape:
        raise ValueError(
            f"blended and reference must have the same shape, "
            f"got {blended.shape} vs {reference.shape}"
        )
    if blended.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB input, got {blended.shape[1]} channels")

    in_dtype = blended.dtype
    blended_lab = _rgb_to_lab(blended.to(torch.float32))    # (B, 3, H, W)
    ref_lab = _rgb_to_lab(reference.to(torch.float32))

    B = blended.shape[0]
    matched_ab_list: list[torch.Tensor] = []

    for b in range(B):
        # (H, W, 2) float32 arrays for skimage
        blend_ab = blended_lab[b, 1:3].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        ref_ab = ref_lab[b, 1:3].permute(1, 2, 0).cpu().numpy().astype(np.float32)

        # Match A and B channels independently  (channel_axis=-1)
        matched = match_histograms(blend_ab, ref_ab, channel_axis=-1).astype(np.float32)
        matched_ab_list.append(
            torch.from_numpy(matched).permute(2, 0, 1)   # (2, H, W)
        )

    matched_ab = torch.stack(matched_ab_list).to(device=blended.device)  # (B, 2, H, W)

    # Rebuild LAB: keep blended's L, replace AB with matched version
    result_lab = torch.cat([blended_lab[:, 0:1], matched_ab], dim=1)  # (B, 3, H, W)
    return _lab_to_rgb(result_lab).to(in_dtype)


def global_brightness_match(
    src: torch.Tensor,
    gen: torch.Tensor,
) -> torch.Tensor:
    """Scale *gen*'s luminance to match *src*'s mean L, keeping gen's chroma.

    A single scalar per image adjusts exposure without frequency-domain
    artifacts or local structure transfer. Simpler than blend_images_lab when
    only global brightness correction is needed.

    Args:
        src: source RGB image, shape (B, 3, H, W), float in [0, 1].
        gen: generated RGB image, same shape as *src*.

    Returns:
        Gen image with L channel globally scaled to match src mean L.
        Shape (B, 3, H, W), float in [0, 1], same dtype as src.
    """
    if src.shape != gen.shape:
        raise ValueError(
            f"src and gen must have the same shape, got {src.shape} vs {gen.shape}"
        )
    if src.shape[1] != 3:
        raise ValueError(f"Expected 3-channel RGB input, got {src.shape[1]} channels")

    in_dtype = src.dtype
    src32 = src.to(torch.float32)
    gen32 = gen.to(torch.float32)

    src_lab = _rgb_to_lab(src32)
    gen_lab = _rgb_to_lab(gen32)

    # Scalar ratio: src mean brightness / gen mean brightness, per batch element.
    src_mean_L = src_lab[:, 0:1].mean(dim=(-2, -1), keepdim=True)  # (B, 1, 1, 1)
    gen_mean_L = gen_lab[:, 0:1].mean(dim=(-2, -1), keepdim=True)
    scale = src_mean_L / gen_mean_L.clamp(min=1e-6)

    adjusted_L = gen_lab[:, 0:1] * scale           # scale gen L toward src exposure
    adjusted_lab = torch.cat([adjusted_L, gen_lab[:, 1:3]], dim=1)  # keep gen AB
    return _lab_to_rgb(adjusted_lab).to(in_dtype)
