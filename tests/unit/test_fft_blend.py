"""Tests for core/fft_blend.py.

Verification strategy: rather than testing final pixel values (which depend on
many interacting frequencies), we inspect the blended result directly in the
frequency domain. This makes the "low-freq from src, high-freq from gen"
invariant explicit and robust.
"""

import pytest
import torch

from attn_texture.core.fft_blend import (
    _lab_to_rgb,
    _rgb_to_lab,
    blend_images_lab,
    blend_latents,
    blend_latents_local,
    global_brightness_match,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fftshift2(t: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fftn(t, dim=(-2, -1)), dim=(-2, -1))


def _low_freq_region(H: int, W: int, ratio: float):
    """Return (row_slice, col_slice) indexing the center low-freq block."""
    cy, cx = H // 2, W // 2
    r = max(1, int(min(H, W) * ratio * 0.5))
    return slice(cy - r, cy + r + 1), slice(cx - r, cx + r + 1)


def _high_freq_region(H: int, W: int):
    """Return a boolean mask True at the four corners (max high-freq)."""
    mask = torch.zeros(H, W, dtype=torch.bool)
    q = max(1, H // 4), max(1, W // 4)
    mask[: q[0], : q[1]] = True
    mask[-q[0] :, : q[1]] = True
    mask[: q[0], -q[1] :] = True
    mask[-q[0] :, -q[1] :] = True
    return mask


# ---------------------------------------------------------------------------
# blend_latents — global blend
# ---------------------------------------------------------------------------


class TestBlendLatents:
    B, C, H, W = 1, 4, 16, 16

    @pytest.fixture()
    def pair(self):
        torch.manual_seed(0)
        src = torch.rand(self.B, self.C, self.H, self.W)
        gen = torch.rand(self.B, self.C, self.H, self.W)
        return src, gen

    def test_output_shape(self, pair):
        src, gen = pair
        assert blend_latents(src, gen).shape == src.shape

    def test_output_dtype_float32(self, pair):
        src, gen = pair
        assert blend_latents(src, gen).dtype == torch.float32

    def test_output_is_real(self, pair):
        src, gen = pair
        result = blend_latents(src, gen)
        assert not result.is_complex()

    def test_no_nan_or_inf(self, pair):
        src, gen = pair
        result = blend_latents(src, gen)
        assert torch.isfinite(result).all()

    def test_identity_when_src_equals_gen(self):
        torch.manual_seed(1)
        t = torch.rand(1, 4, 16, 16)
        result = blend_latents(t, t)
        # FFT → blend → IFFT round-trip; tolerance from measured 1e-7 max error.
        assert torch.allclose(result, t, atol=1e-5)

    def test_batched_input(self):
        torch.manual_seed(2)
        src = torch.rand(3, 4, 16, 16)
        gen = torch.rand(3, 4, 16, 16)
        assert blend_latents(src, gen).shape == (3, 4, 16, 16)

    # --- Frequency-domain correctness ----------------------------------------

    def test_low_freq_bins_come_from_src(self, pair):
        """Center of fftshift spectrum must match src, not gen."""
        src, gen = pair
        cutoff = 0.3
        result = blend_latents(src, gen, cutoff_ratio=cutoff)

        rs, ss, gs = _fftshift2(result), _fftshift2(src), _fftshift2(gen)
        ry, rx = _low_freq_region(self.H, self.W, cutoff)

        err_from_src = (rs[..., ry, rx] - ss[..., ry, rx]).abs().mean().item()
        err_from_gen = (rs[..., ry, rx] - gs[..., ry, rx]).abs().mean().item()
        assert err_from_src < err_from_gen, (
            f"Low-freq bins should be closer to src (Δsrc={err_from_src:.4f}, Δgen={err_from_gen:.4f})"
        )

    def test_high_freq_bins_come_from_gen(self, pair):
        """Corners of fftshift spectrum (max freq) must match gen, not src."""
        src, gen = pair
        cutoff = 0.3
        result = blend_latents(src, gen, cutoff_ratio=cutoff)

        rs, ss, gs = _fftshift2(result), _fftshift2(src), _fftshift2(gen)
        hf = _high_freq_region(self.H, self.W)  # (H, W) bool mask

        err_from_gen = (rs[..., hf] - gs[..., hf]).abs().mean().item()
        err_from_src = (rs[..., hf] - ss[..., hf]).abs().mean().item()
        assert err_from_gen < err_from_src, (
            f"High-freq bins should be closer to gen (Δgen={err_from_gen:.4f}, Δsrc={err_from_src:.4f})"
        )

    def test_cutoff_ratio_zero_returns_gen(self):
        """cutoff_ratio=0 → low-freq mask covers nothing → result equals gen."""
        torch.manual_seed(3)
        src = torch.rand(1, 4, 16, 16)
        gen = torch.rand(1, 4, 16, 16)
        result = blend_latents(src, gen, cutoff_ratio=0.0)
        assert torch.allclose(result, gen, atol=1e-5)

    def test_cutoff_ratio_one_returns_src(self):
        """cutoff_ratio=1 → low-freq mask covers everything → result equals src."""
        torch.manual_seed(4)
        src = torch.rand(1, 4, 16, 16)
        gen = torch.rand(1, 4, 16, 16)
        result = blend_latents(src, gen, cutoff_ratio=1.0)
        assert torch.allclose(result, src, atol=1e-5)

    def test_different_cutoffs_give_different_results(self):
        torch.manual_seed(5)
        src = torch.rand(1, 4, 16, 16)
        gen = torch.rand(1, 4, 16, 16)
        r1 = blend_latents(src, gen, cutoff_ratio=0.2)
        r2 = blend_latents(src, gen, cutoff_ratio=0.8)
        assert not torch.allclose(r1, r2, atol=1e-4)

    def test_rejects_mismatched_shapes(self):
        src = torch.rand(1, 4, 16, 16)
        gen = torch.rand(1, 4, 8, 8)
        with pytest.raises(ValueError, match="shape"):
            blend_latents(src, gen)

    def test_rejects_invalid_cutoff(self):
        t = torch.rand(1, 4, 16, 16)
        with pytest.raises(ValueError, match="cutoff_ratio"):
            blend_latents(t, t, cutoff_ratio=1.5)


# ---------------------------------------------------------------------------
# blend_latents_local — spatially-masked blend
# ---------------------------------------------------------------------------


class TestBlendLatentsLocal:
    B, C, H, W = 1, 4, 16, 16

    @pytest.fixture()
    def pair(self):
        torch.manual_seed(10)
        src = torch.rand(self.B, self.C, self.H, self.W)
        gen = torch.rand(self.B, self.C, self.H, self.W)
        return src, gen

    def test_output_shape(self, pair):
        src, gen = pair
        mask = torch.ones(self.B, 1, self.H, self.W)
        assert blend_latents_local(src, gen, mask).shape == src.shape

    def test_zeros_mask_returns_src(self, pair):
        """No pixels selected → output == src everywhere."""
        src, gen = pair
        mask = torch.zeros(self.B, 1, self.H, self.W)
        result = blend_latents_local(src, gen, mask)
        assert torch.allclose(result, src, atol=1e-6)

    def test_ones_mask_matches_global_blend(self, pair):
        """Full mask → local blend is identical to global blend."""
        src, gen = pair
        mask = torch.ones(self.B, 1, self.H, self.W)
        local = blend_latents_local(src, gen, mask)
        global_ = blend_latents(src, gen)
        assert torch.allclose(local, global_, atol=1e-6)

    def test_outside_mask_pixels_unchanged(self, pair):
        """Pixels where mask==0 must be bitwise identical to src."""
        src, gen = pair
        # Top half masked, bottom half not
        mask = torch.zeros(self.B, 1, self.H, self.W)
        mask[..., : self.H // 2, :] = 1.0

        result = blend_latents_local(src, gen, mask)
        unmasked = result[..., self.H // 2 :, :]
        src_unmasked = src[..., self.H // 2 :, :]
        assert torch.allclose(unmasked, src_unmasked, atol=1e-6)

    def test_inside_mask_pixels_differ_from_src(self, pair):
        """Pixels where mask==1 must be modified (src ≠ gen ensures this)."""
        src, gen = pair
        mask = torch.zeros(self.B, 1, self.H, self.W)
        mask[..., : self.H // 2, :] = 1.0

        result = blend_latents_local(src, gen, mask)
        masked_result = result[..., : self.H // 2, :]
        masked_src = src[..., : self.H // 2, :]
        assert not torch.allclose(masked_result, masked_src, atol=1e-4)

    def test_soft_mask_interpolates(self, pair):
        """mask=0.5 → result between src and global-blend at each pixel."""
        src, gen = pair
        mask = torch.full((self.B, 1, self.H, self.W), 0.5)
        global_ = blend_latents(src, gen)
        result = blend_latents_local(src, gen, mask)
        expected = 0.5 * global_ + 0.5 * src
        assert torch.allclose(result, expected, atol=1e-5)

    def test_2d_mask_broadcast(self, pair):
        """(H, W) mask should broadcast correctly to (B, C, H, W)."""
        src, gen = pair
        mask = torch.ones(self.H, self.W)
        result = blend_latents_local(src, gen, mask)
        assert result.shape == src.shape

    def test_no_nan_or_inf(self, pair):
        src, gen = pair
        mask = torch.rand(self.B, 1, self.H, self.W)
        assert torch.isfinite(blend_latents_local(src, gen, mask)).all()


# ---------------------------------------------------------------------------
# _rgb_to_lab / _lab_to_rgb — colour-space helpers
# ---------------------------------------------------------------------------


class TestRgbLabRoundTrip:
    B, H, W = 2, 8, 8

    @pytest.fixture()
    def rgb(self):
        torch.manual_seed(42)
        return torch.rand(self.B, 3, self.H, self.W)

    def test_rgb_to_lab_shape(self, rgb):
        assert _rgb_to_lab(rgb).shape == rgb.shape

    def test_lab_to_rgb_shape(self, rgb):
        assert _lab_to_rgb(_rgb_to_lab(rgb)).shape == rgb.shape

    def test_round_trip(self, rgb):
        recovered = _lab_to_rgb(_rgb_to_lab(rgb))
        assert torch.allclose(recovered, rgb, atol=1e-4)

    def test_white_l_is_100(self):
        white = torch.ones(1, 3, 1, 1)
        lab = _rgb_to_lab(white)
        assert abs(lab[0, 0, 0, 0].item() - 100.0) < 0.5

    def test_black_l_is_zero(self):
        black = torch.zeros(1, 3, 1, 1)
        lab = _rgb_to_lab(black)
        assert abs(lab[0, 0, 0, 0].item()) < 0.5

    def test_lab_to_rgb_output_clamped(self, rgb):
        out = _lab_to_rgb(_rgb_to_lab(rgb))
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0 + 1e-6

    def test_dtype_preserved(self, rgb):
        assert _rgb_to_lab(rgb).dtype == rgb.dtype
        assert _lab_to_rgb(_rgb_to_lab(rgb)).dtype == rgb.dtype


# ---------------------------------------------------------------------------
# blend_images_lab — pixel-space LAB luminance blend
# ---------------------------------------------------------------------------


class TestBlendImagesLab:
    B, H, W = 1, 16, 16

    @pytest.fixture()
    def pair(self):
        torch.manual_seed(20)
        src = torch.rand(self.B, 3, self.H, self.W)
        gen = torch.rand(self.B, 3, self.H, self.W)
        return src, gen

    def test_output_shape(self, pair):
        src, gen = pair
        assert blend_images_lab(src, gen).shape == (self.B, 3, self.H, self.W)

    def test_no_nan_or_inf(self, pair):
        src, gen = pair
        assert torch.isfinite(blend_images_lab(src, gen)).all()

    def test_output_in_unit_range(self, pair):
        src, gen = pair
        out = blend_images_lab(src, gen)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0 + 1e-6

    def test_ab_channels_come_from_gen(self, pair):
        """Output must equal the LAB-assembled image using gen's chroma.

        Checking round-trip LAB→RGB→LAB is fragile for out-of-gamut colours
        (src-L + gen-AB can go out of sRGB range, clamping distorts the AB
        when going back). Instead we reconstruct the expected output with the
        same logic and compare directly.
        """
        src, gen = pair
        out = blend_images_lab(src, gen)
        gen_lab = _rgb_to_lab(gen)
        src_L = _rgb_to_lab(src)[:, 0:1]
        blended_L = blend_latents(src_L, gen_lab[:, 0:1], cutoff_ratio=0.5)
        expected = _lab_to_rgb(torch.cat([blended_L, gen_lab[:, 1:3]], dim=1))
        assert torch.allclose(out, expected, atol=1e-6), (
            "Output should equal LAB blend with gen's AB channels preserved"
        )

    def test_l_low_freq_from_src(self, pair):
        """Low-freq L bins must be closer to src's L than gen's L."""
        src, gen = pair
        cutoff = 0.3
        out = blend_images_lab(src, gen, cutoff_ratio=cutoff)

        src_L = _rgb_to_lab(src)[:, 0:1]
        gen_L = _rgb_to_lab(gen)[:, 0:1]
        out_L = _rgb_to_lab(out)[:, 0:1]

        ry, rx = _low_freq_region(self.H, self.W, cutoff)
        err_src = (_fftshift2(out_L)[..., ry, rx] - _fftshift2(src_L)[..., ry, rx]).abs().mean()
        err_gen = (_fftshift2(out_L)[..., ry, rx] - _fftshift2(gen_L)[..., ry, rx]).abs().mean()
        assert err_src < err_gen, (
            f"Low-freq L should be closer to src (Δsrc={err_src:.4f}, Δgen={err_gen:.4f})"
        )

    def test_identity_when_src_equals_gen(self):
        torch.manual_seed(21)
        t = torch.rand(1, 3, 16, 16)
        out = blend_images_lab(t, t)
        assert torch.allclose(out, t, atol=1e-4)

    def test_cutoff_zero_returns_gen(self, pair):
        """cutoff=0 → entire L from gen → output ≈ gen (same L and AB)."""
        src, gen = pair
        out = blend_images_lab(src, gen, cutoff_ratio=0.0)
        assert torch.allclose(out, gen, atol=1e-4)

    def test_rejects_non_rgb(self):
        t = torch.rand(1, 4, 16, 16)
        with pytest.raises(ValueError, match="3-channel"):
            blend_images_lab(t, t)

    def test_rejects_mismatched_shapes(self):
        src = torch.rand(1, 3, 16, 16)
        gen = torch.rand(1, 3, 8, 8)
        with pytest.raises(ValueError, match="shape"):
            blend_images_lab(src, gen)

    def test_rejects_invalid_cutoff(self):
        t = torch.rand(1, 3, 16, 16)
        with pytest.raises(ValueError, match="cutoff_ratio"):
            blend_images_lab(t, t, cutoff_ratio=1.5)


# ---------------------------------------------------------------------------
# global_brightness_match — scalar L-channel exposure correction
# ---------------------------------------------------------------------------


class TestGlobalBrightnessMatch:
    B, H, W = 1, 16, 16

    @pytest.fixture()
    def pair(self):
        torch.manual_seed(30)
        src = torch.rand(self.B, 3, self.H, self.W)
        gen = torch.rand(self.B, 3, self.H, self.W)
        return src, gen

    def test_output_shape(self, pair):
        src, gen = pair
        assert global_brightness_match(src, gen).shape == (self.B, 3, self.H, self.W)

    def test_no_nan_or_inf(self, pair):
        src, gen = pair
        assert torch.isfinite(global_brightness_match(src, gen)).all()

    def test_output_in_unit_range(self, pair):
        src, gen = pair
        out = global_brightness_match(src, gen)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0 + 1e-6

    def test_dtype_preserved(self, pair):
        src, gen = pair
        assert global_brightness_match(src, gen).dtype == src.dtype

    def test_identity_when_src_equals_gen(self):
        """Same image → scale=1 → output equals input."""
        torch.manual_seed(31)
        t = torch.rand(1, 3, 16, 16)
        out = global_brightness_match(t, t)
        assert torch.allclose(out, t, atol=1e-4)

    def test_ab_channels_from_gen(self, pair):
        """Output must equal the LAB-assembled image using gen's original AB channels."""
        src, gen = pair
        out = global_brightness_match(src, gen)
        gen_lab = _rgb_to_lab(gen)
        src_lab = _rgb_to_lab(src)
        src_mean_L = src_lab[:, 0:1].mean(dim=(-2, -1), keepdim=True)
        gen_mean_L = gen_lab[:, 0:1].mean(dim=(-2, -1), keepdim=True)
        scale = src_mean_L / gen_mean_L.clamp(min=1e-6)
        adjusted_L = gen_lab[:, 0:1] * scale
        expected = _lab_to_rgb(torch.cat([adjusted_L, gen_lab[:, 1:3]], dim=1))
        assert torch.allclose(out, expected, atol=1e-6)

    def test_bright_src_lifts_dark_gen(self):
        """Output mean L must be higher than gen's mean L when src is much brighter."""
        # src: near-white; gen: near-black
        src = torch.full((1, 3, 16, 16), 0.9)
        gen = torch.full((1, 3, 16, 16), 0.1)
        out = global_brightness_match(src, gen)
        gen_L = _rgb_to_lab(gen)[:, 0:1].mean().item()
        out_L = _rgb_to_lab(out)[:, 0:1].mean().item()
        assert out_L > gen_L, f"Expected output L ({out_L:.2f}) > gen L ({gen_L:.2f})"

    def test_dark_src_darkens_bright_gen(self):
        """Output mean L must be lower than gen's mean L when src is much darker."""
        src = torch.full((1, 3, 16, 16), 0.1)
        gen = torch.full((1, 3, 16, 16), 0.9)
        out = global_brightness_match(src, gen)
        gen_L = _rgb_to_lab(gen)[:, 0:1].mean().item()
        out_L = _rgb_to_lab(out)[:, 0:1].mean().item()
        assert out_L < gen_L, f"Expected output L ({out_L:.2f}) < gen L ({gen_L:.2f})"

    def test_rejects_non_rgb(self):
        t = torch.rand(1, 4, 16, 16)
        with pytest.raises(ValueError, match="3-channel"):
            global_brightness_match(t, t)

    def test_rejects_mismatched_shapes(self):
        src = torch.rand(1, 3, 16, 16)
        gen = torch.rand(1, 3, 8, 8)
        with pytest.raises(ValueError, match="shape"):
            global_brightness_match(src, gen)
