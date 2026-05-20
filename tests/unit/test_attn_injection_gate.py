"""Tests for core/attention_injection.py — PnP feature/self-attention injection gate.

All tests use minimal mock UNets (no model weights downloaded, no DDIM inversion).

Public contract under test
--------------------------
inject_kv(ref_kv, gen_kv) -> torch.Tensor
    Replace gen K/V with ref K/V.  Shape (B, H, N, D) is preserved.

AttentionInjector(unet, tau_f, tau_A)
    Register forward hooks on UNet decoder blocks without touching weights.
    Gate: inject features when step / total_steps > tau_f,
          inject self-attention when step / total_steps > tau_A.
    Thresholds tau_f / tau_A come from config (from_config); never hardcoded.

CapturedFeatures
    Populated during begin_ref_pass() / end_ref_pass() lifecycle:
      .spatial_features   — f^4_t, shape (B, C, H, W), from one decoder resnet
      .self_attentions    — {A^l_t}, one tensor per decoder self-attention layer
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from attn_texture.core.attention_injection import (
    AttentionInjector,
    CapturedFeatures,
    inject_kv,
)


# ---------------------------------------------------------------------------
# Minimal mock UNet  (mirrors diffusers UNet2DConditionModel decoder structure)
#
#   up_blocks[i]
#     .resnets[j]            — ResnetBlock2D  (spatial features f^l_t live here)
#     .attentions[k]
#       .transformer_blocks[m]
#         .attn1             — Attention module (self-attention A^l_t)
# ---------------------------------------------------------------------------


class _MockAttn1(nn.Module):
    """Self-attention matching the diffusers Attention forward signature."""

    _DIM: int = 8

    def __init__(self) -> None:
        super().__init__()
        d = self._DIM
        self.to_q = nn.Linear(d, d, bias=False)
        self.to_k = nn.Linear(d, d, bias=False)
        self.to_v = nn.Linear(d, d, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(d, d, bias=False)])

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # (B, N, D) -> (B, N, D)
        src = encoder_hidden_states if encoder_hidden_states is not None else x
        q, k, v = self.to_q(x), self.to_k(src), self.to_v(src)
        scale = self._DIM ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        return self.to_out[0](attn @ v)


class _MockTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn1 = _MockAttn1()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn1(x) + x


class _MockTransformer2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_MockTransformerBlock()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.transformer_blocks:
            x = blk(x)
        return x


class _MockResNet(nn.Module):
    _CH: int = 4

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(self._CH, self._CH, 1)

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(x)


class _MockUpBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([_MockResNet(), _MockResNet()])
        self.attentions = nn.ModuleList([_MockTransformer2D()])


class _MockUNet(nn.Module):
    """Three up-blocks; each has 2 resnets and 1 attention — mirrors diffusers decoder."""

    B, C, H, W = 1, 4, 4, 4  # spatial latent shape
    N, D_SEQ = 16, 8          # sequence dims (N = H*W flattened for attention)

    def __init__(self) -> None:
        super().__init__()
        self.up_blocks = nn.ModuleList([_MockUpBlock() for _ in range(3)])

    def forward(
        self,
        x_sp: torch.Tensor,
        x_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for blk in self.up_blocks:
            for res in blk.resnets:
                x_sp = res(x_sp)
            for attn in blk.attentions:
                x_seq = attn(x_seq)
        return x_sp, x_seq

    def rand_inputs(self, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        return (
            torch.randn(self.B, self.C, self.H, self.W),
            torch.randn(self.B, self.N, self.D_SEQ),
        )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def unet() -> _MockUNet:
    return _MockUNet()


@pytest.fixture()
def cfg():
    return OmegaConf.create({"tau_f": 0.8, "tau_A": 0.5})


@pytest.fixture()
def injector(unet: _MockUNet, cfg) -> AttentionInjector:
    inj = AttentionInjector.from_config(unet, cfg)
    inj.register_hooks()
    return inj


def _do_ref_pass(
    injector: AttentionInjector,
    unet: _MockUNet,
    seed: int = 0,
) -> CapturedFeatures:
    """Run one full ref_pass and return the captured features."""
    x_sp, x_seq = unet.rand_inputs(seed)
    injector.begin_ref_pass()
    unet(x_sp, x_seq)
    return injector.end_ref_pass()


# ---------------------------------------------------------------------------
# inject_kv — pure function
# ---------------------------------------------------------------------------


class TestInjectKv:
    """inject_kv replaces the generation K/V with the reference K/V."""

    _B, _H, _N, _D = 2, 4, 16, 8

    def _pair(self) -> tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(0)
        ref = torch.randn(self._B, self._H, self._N, self._D)
        gen = torch.randn(self._B, self._H, self._N, self._D)
        return ref, gen

    def test_output_shape_preserved(self):
        ref, gen = self._pair()
        assert inject_kv(ref, gen).shape == ref.shape

    def test_output_equals_ref_kv(self):
        """Contract: inject_kv is a pure replacement — output must equal ref_kv."""
        ref, gen = self._pair()
        assert torch.equal(inject_kv(ref, gen), ref)

    def test_output_differs_from_gen_kv(self):
        ref, gen = self._pair()
        assert not torch.equal(inject_kv(ref, gen), gen)

    def test_batched_arbitrary_shape(self):
        ref = torch.randn(4, 8, 32, 16)
        gen = torch.randn(4, 8, 32, 16)
        assert inject_kv(ref, gen).shape == (4, 8, 32, 16)

    def test_raises_on_shape_mismatch(self):
        ref = torch.randn(1, 4, 16, 8)
        gen = torch.randn(1, 4, 32, 8)  # N differs
        with pytest.raises(ValueError, match="shape"):
            inject_kv(ref, gen)


# ---------------------------------------------------------------------------
# CapturedFeatures — dataclass contract
# ---------------------------------------------------------------------------


class TestCapturedFeatures:
    def test_default_spatial_features_is_none(self):
        assert CapturedFeatures().spatial_features is None

    def test_default_self_attentions_is_empty(self):
        assert CapturedFeatures().self_attentions == []

    def test_is_a_dataclass(self):
        from dataclasses import is_dataclass
        assert is_dataclass(CapturedFeatures)


# ---------------------------------------------------------------------------
# Gate logic — pure arithmetic, no UNet forward needed
# ---------------------------------------------------------------------------


class TestGateLogic:
    """Gate is open when step / total_steps > threshold (strictly greater than)."""

    @pytest.fixture()
    def inj(self, unet: _MockUNet, cfg) -> AttentionInjector:
        return AttentionInjector.from_config(unet, cfg)

    # --- feature gate (tau_f = 0.8) ---

    def test_features_gate_open_above_threshold(self, inj):
        assert inj.should_inject_features(step=900, total_steps=1000) is True

    def test_features_gate_closed_below_threshold(self, inj):
        assert inj.should_inject_features(step=700, total_steps=1000) is False

    def test_features_gate_closed_exactly_at_threshold(self, inj):
        # 800/1000 == tau_f=0.8 — strictly greater required, so gate is closed
        assert inj.should_inject_features(step=800, total_steps=1000) is False

    # --- attention gate (tau_A = 0.5) ---

    def test_attention_gate_open_above_threshold(self, inj):
        assert inj.should_inject_attention(step=600, total_steps=1000) is True

    def test_attention_gate_closed_below_threshold(self, inj):
        assert inj.should_inject_attention(step=400, total_steps=1000) is False

    def test_attention_gate_closed_exactly_at_threshold(self, inj):
        assert inj.should_inject_attention(step=500, total_steps=1000) is False

    def test_gate_uses_fraction_not_absolute_step_count(self, inj):
        """Scale invariance: equal fractions must produce the same gate result."""
        # 160/200 == 40/50 == 0.8 == tau_f → both closed
        assert inj.should_inject_features(160, 200) == inj.should_inject_features(40, 50)

    def test_step_zero_always_closes_gate(self, inj):
        assert inj.should_inject_features(step=0, total_steps=1000) is False
        assert inj.should_inject_attention(step=0, total_steps=1000) is False

    def test_step_equals_total_steps_opens_gate(self, inj):
        # fraction = 1.0 > any tau in (0, 1) → open
        assert inj.should_inject_features(step=1000, total_steps=1000) is True
        assert inj.should_inject_attention(step=1000, total_steps=1000) is True


# ---------------------------------------------------------------------------
# Config-driven construction
# ---------------------------------------------------------------------------


class TestConfigDrivenConstruction:
    """tau_f and tau_A must come from config — no hardcoded fallbacks allowed."""

    def test_from_config_reads_tau_f(self, unet: _MockUNet):
        cfg = OmegaConf.create({"tau_f": 0.3, "tau_A": 0.5})
        inj = AttentionInjector.from_config(unet, cfg)
        assert inj.tau_f == pytest.approx(0.3)

    def test_from_config_reads_tau_A(self, unet: _MockUNet):
        cfg = OmegaConf.create({"tau_f": 0.8, "tau_A": 0.2})
        inj = AttentionInjector.from_config(unet, cfg)
        assert inj.tau_A == pytest.approx(0.2)

    def test_different_tau_f_changes_gate_behaviour(self, unet: _MockUNet):
        cfg_strict = OmegaConf.create({"tau_f": 0.9, "tau_A": 0.5})
        cfg_loose = OmegaConf.create({"tau_f": 0.5, "tau_A": 0.5})
        inj_strict = AttentionInjector.from_config(unet, cfg_strict)
        inj_loose = AttentionInjector.from_config(unet, cfg_loose)
        # Fraction 0.7: above loose threshold, below strict
        assert inj_strict.should_inject_features(700, 1000) is False
        assert inj_loose.should_inject_features(700, 1000) is True

    def test_missing_tau_f_raises(self, unet: _MockUNet):
        cfg = OmegaConf.create({"tau_A": 0.5})  # tau_f absent
        with pytest.raises(Exception):
            AttentionInjector.from_config(unet, cfg)

    def test_missing_tau_A_raises(self, unet: _MockUNet):
        cfg = OmegaConf.create({"tau_f": 0.8})  # tau_A absent
        with pytest.raises(Exception):
            AttentionInjector.from_config(unet, cfg)


# ---------------------------------------------------------------------------
# Hook registration — weights must be untouched
# ---------------------------------------------------------------------------


class TestHookRegistration:
    def test_weights_unchanged_after_register(self, unet: _MockUNet, cfg):
        before = {n: p.clone() for n, p in unet.named_parameters()}
        inj = AttentionInjector.from_config(unet, cfg)
        inj.register_hooks()
        for name, param in unet.named_parameters():
            assert torch.equal(param, before[name]), (
                f"register_hooks() mutated weight '{name}'"
            )

    def test_hooks_are_attached_to_unet(self, unet: _MockUNet, cfg):
        inj = AttentionInjector.from_config(unet, cfg)
        inj.register_hooks()
        has_hooks = any(
            len(m._forward_hooks) > 0 or len(m._forward_pre_hooks) > 0
            for m in unet.modules()
        )
        assert has_hooks, "No hooks found on any submodule after register_hooks()"

    def test_remove_hooks_clears_all(self, unet: _MockUNet, cfg):
        inj = AttentionInjector.from_config(unet, cfg)
        inj.register_hooks()
        inj.remove_hooks()
        still_hooked = any(
            len(m._forward_hooks) > 0 or len(m._forward_pre_hooks) > 0
            for m in unet.modules()
        )
        assert not still_hooked, "Hooks still present after remove_hooks()"

    def test_register_twice_does_not_duplicate_hooks(self, unet: _MockUNet, cfg):
        """Double registration must not prevent clean removal."""
        inj = AttentionInjector.from_config(unet, cfg)
        inj.register_hooks()
        inj.register_hooks()
        inj.remove_hooks()
        assert not any(len(m._forward_hooks) > 0 for m in unet.modules())


# ---------------------------------------------------------------------------
# Capture during ref_pass
# ---------------------------------------------------------------------------


class TestRefPassCapture:
    """begin_ref_pass() / end_ref_pass() must populate CapturedFeatures."""

    def test_spatial_features_not_none_after_capture(self, injector, unet):
        caps = _do_ref_pass(injector, unet)
        assert caps.spatial_features is not None

    def test_spatial_features_is_4d_bhwc(self, injector, unet):
        caps = _do_ref_pass(injector, unet)
        f = caps.spatial_features
        assert f is not None and f.dim() == 4, (
            f"Expected (B, C, H, W) spatial features, got shape "
            f"{f.shape if f is not None else 'None'}"
        )

    def test_spatial_features_channel_dim_matches_resnet(self, injector, unet):
        caps = _do_ref_pass(injector, unet)
        assert caps.spatial_features is not None
        assert caps.spatial_features.shape[1] == _MockResNet._CH

    def test_self_attentions_list_is_nonempty(self, injector, unet):
        caps = _do_ref_pass(injector, unet)
        assert len(caps.self_attentions) > 0

    def test_self_attention_count_equals_decoder_attention_layers(self, injector, unet):
        caps = _do_ref_pass(injector, unet)
        # One attention module per up_block, one transformer_block each → one attn1 each
        expected = sum(len(blk.attentions) for blk in unet.up_blocks)
        assert len(caps.self_attentions) == expected

    def test_captured_tensors_are_detached_from_graph(self, injector, unet):
        caps = _do_ref_pass(injector, unet)
        assert caps.spatial_features is not None
        assert not caps.spatial_features.requires_grad
        for a in caps.self_attentions:
            assert not a.requires_grad

    def test_second_capture_overwrites_first(self, injector, unet):
        """A new ref_pass replaces the previous capture — no accumulation."""
        caps1 = _do_ref_pass(injector, unet, seed=0)
        caps2 = _do_ref_pass(injector, unet, seed=99)
        assert caps1.spatial_features is not None and caps2.spatial_features is not None
        assert not torch.equal(caps1.spatial_features, caps2.spatial_features)

    def test_captured_self_attentions_are_finite(self, injector, unet):
        caps = _do_ref_pass(injector, unet)
        for i, a in enumerate(caps.self_attentions):
            assert torch.isfinite(a).all(), f"self_attentions[{i}] contains NaN/Inf"


# ---------------------------------------------------------------------------
# Injection gate during gen_pass
# ---------------------------------------------------------------------------


class TestGenPassInjection:
    """Injection happens iff step / total_steps > threshold for the respective gate."""

    @staticmethod
    def _clean_run(
        unet: _MockUNet,
        x_sp: torch.Tensor,
        x_seq: torch.Tensor,
        injector: AttentionInjector,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Baseline forward with no hooks active."""
        injector.remove_hooks()
        with torch.no_grad():
            out = unet(x_sp.clone(), x_seq.clone())
        injector.register_hooks()
        return out

    def test_gate_open_modifies_output(self, injector, unet):
        """Both gates open (step/T=0.9 > tau_f=0.8 and tau_A=0.5) → output changes."""
        _do_ref_pass(injector, unet, seed=0)

        x_sp, x_seq = unet.rand_inputs(seed=1)
        out_clean_sp, out_clean_seq = self._clean_run(unet, x_sp, x_seq, injector)

        # Re-capture after hook re-registration
        _do_ref_pass(injector, unet, seed=0)

        injector.begin_gen_pass(step=900, total_steps=1000)
        with torch.no_grad():
            out_inj_sp, out_inj_seq = unet(x_sp.clone(), x_seq.clone())
        injector.end_gen_pass()

        sp_changed = not torch.allclose(out_inj_sp, out_clean_sp, atol=1e-5)
        seq_changed = not torch.allclose(out_inj_seq, out_clean_seq, atol=1e-5)
        assert sp_changed or seq_changed, (
            "Gate was open (step/T=0.9 > tau_f=0.8 and tau_A=0.5) but UNet output "
            "is identical to the no-injection baseline — injection did nothing"
        )

    def test_spatial_gate_closed_does_not_modify_spatial_output(self, injector, unet):
        """Spatial gate closed (step/T=0.5 ≤ tau_f=0.8) → x_sp output unchanged."""
        _do_ref_pass(injector, unet, seed=0)

        x_sp, x_seq = unet.rand_inputs(seed=2)
        out_clean_sp, _ = self._clean_run(unet, x_sp, x_seq, injector)

        _do_ref_pass(injector, unet, seed=0)

        injector.begin_gen_pass(step=500, total_steps=1000)
        with torch.no_grad():
            out_inj_sp, _ = unet(x_sp.clone(), x_seq.clone())
        injector.end_gen_pass()

        assert torch.allclose(out_inj_sp, out_clean_sp, atol=1e-5), (
            "Spatial feature gate was closed (step/T=0.5 ≤ tau_f=0.8) "
            "but spatial output was modified"
        )

    def test_both_gates_closed_full_pass_through(self, injector, unet):
        """Both gates closed (step/T=0.3 < tau_A=0.5) → output identical to clean run."""
        _do_ref_pass(injector, unet, seed=0)

        x_sp, x_seq = unet.rand_inputs(seed=3)
        out_clean_sp, out_clean_seq = self._clean_run(unet, x_sp, x_seq, injector)

        _do_ref_pass(injector, unet, seed=0)

        injector.begin_gen_pass(step=300, total_steps=1000)
        with torch.no_grad():
            out_inj_sp, out_inj_seq = unet(x_sp.clone(), x_seq.clone())
        injector.end_gen_pass()

        assert torch.allclose(out_inj_sp, out_clean_sp, atol=1e-5)
        assert torch.allclose(out_inj_seq, out_clean_seq, atol=1e-5)

    def test_gen_pass_without_prior_ref_raises(self, injector, unet):
        """Injecting with no captured features is a programming error — must raise."""
        x_sp, x_seq = unet.rand_inputs()
        injector.begin_gen_pass(step=900, total_steps=1000)
        with pytest.raises(RuntimeError):
            unet(x_sp, x_seq)
        injector.end_gen_pass()

    def test_end_gen_pass_returns_to_idle_state(self, injector, unet):
        """After end_gen_pass, a plain forward must not raise or inject stale data."""
        _do_ref_pass(injector, unet, seed=0)
        injector.begin_gen_pass(step=900, total_steps=1000)
        x_sp, x_seq = unet.rand_inputs(seed=1)
        with torch.no_grad():
            unet(x_sp, x_seq)
        injector.end_gen_pass()

        # Idle forward: must not raise
        with torch.no_grad():
            unet(x_sp, x_seq)
