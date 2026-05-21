"""PnP self-attention and spatial-feature injection.

Two-hook strategy per Plug-and-Play Diffusion Features (Tumanyan et al., CVPR 2023 §4)
cross-checked against the official implementation at github.com/MichalGeyer/plug-and-play:

  Ref pass (guidance):
    • Output hook on up_blocks[1].resnets[0] captures spatial features f^4_t.  (D3)
    • Output hooks on to_q / to_k of every self-attention (attn1) in
      up_blocks[1..N] capture Q and K tensors.  (D1, D4)

  Gen pass (translation):
    • Same hooks gate-check step / T against τ_f / τ_A (from config, never hardcoded):
        step/T > τ_f  →  replace ResNet output with captured f^4_t         (§ 4 Eq. 3)
        step/T > τ_A  →  replace to_q / to_k outputs with captured Q / K   (§ 4 Eq. 4)

Key deviations from the naive paper reading that source code clarifies:
  • What is injected is Q and K, not the full attention output A·V.  (D1)
  • up_blocks[0] (coarsest) attention is intentionally skipped.  (D4)
  • Spatial feature layer is up_blocks[1].resnets[0] (16×16), not up_blocks[0].  (D3)

Hooks are registered once and kept alive for the full pipeline run; mode
transitions (idle → capture → idle → inject → idle) control their behaviour.
Model weights are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig


# ---------------------------------------------------------------------------
# Public pure function
# ---------------------------------------------------------------------------


def inject_kv(ref_kv: torch.Tensor, gen_kv: torch.Tensor) -> torch.Tensor:
    """Inject reference K/V into generation stream.

    Pure replacement: the generation stream's K/V are discarded and replaced by
    the reference K/V so that the generated image follows the reference structure.

    Args:
        ref_kv: reference K/V, shape (B, H, N, D)
        gen_kv: generation K/V, shape (B, H, N, D)

    Returns:
        injected K/V, shape (B, H, N, D) — identical to ref_kv.

    Raises:
        ValueError: if ref_kv and gen_kv differ in shape.
    """
    if ref_kv.shape != gen_kv.shape:
        raise ValueError(
            f"ref_kv and gen_kv must have identical shape, "
            f"got {tuple(ref_kv.shape)} vs {tuple(gen_kv.shape)}"
        )
    return ref_kv


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class CapturedFeatures:
    """Features captured from the reference pass.

    Attributes:
        spatial_features: f^4_t from up_blocks[1].resnets[0], shape (B, C, H, W).
                          None until a ref_pass has run.
        attn_qk:          [(Q_i, K_i)] for each decoder self-attention layer in
                          up_blocks[1..N].  Q/K shapes are (B, N, D) — the linear
                          projection outputs before head-split.  Empty until a
                          ref_pass has run.
    """

    spatial_features: torch.Tensor | None = None
    attn_qk: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------

_Mode = Literal["idle", "capture", "inject"]


class AttentionInjector:
    """PnP feature + self-attention injector. Tumanyan et al., CVPR 2023 §4.

    Registers forward hooks on a diffusers-compatible UNet2DConditionModel
    decoder (or any nn.Module with the same up_blocks structure).  No model
    weights are modified at any point.

    Hooks on to_q / to_k (not attn1 output) let the gen-pass attention
    compute its own values V while the source Q and K drive the attention map —
    matching the official PnP behaviour.  up_blocks[0] (coarsest) is skipped
    for attention, consistent with output_blocks[4..11] in the LDM reference.

    Lifecycle per denoising step
    ----------------------------
    1. injector.begin_ref_pass()
    2. unet(z_ref, ...)                # ref/guidance forward — hooks capture Q/K
    3. caps = injector.end_ref_pass()
    4. injector.begin_gen_pass(t, T)
    5. unet(z_gen, ...)                # gen forward — hooks inject Q/K (gated)
    6. injector.end_gen_pass()

    Args:
        unet:  nn.Module with ``up_blocks[i].resnets[j]`` and
               ``up_blocks[i].attentions[k].transformer_blocks[m].attn1``
               (which must have ``to_q`` and ``to_k`` nn.Linear sub-modules).
        tau_f: fraction threshold for spatial-feature injection (0 < τ_f < 1).
        tau_A: fraction threshold for self-attention Q/K injection (0 < τ_A < 1).
    """

    def __init__(self, unet: nn.Module, tau_f: float, tau_A: float) -> None:
        self._unet = unet
        self.tau_f = tau_f
        self.tau_A = tau_A

        self._mode: _Mode = "idle"
        self._step: int = 0
        self._total_steps: int = 1

        self._captured: CapturedFeatures = CapturedFeatures()
        self._q_capture_buf: dict[int, torch.Tensor] = {}
        self._k_capture_buf: dict[int, torch.Tensor] = {}

        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._feature_module: nn.Module = self._find_feature_module()
        self._attn1_modules: list[nn.Module] = self._find_attn1_modules()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, unet: nn.Module, cfg: DictConfig) -> "AttentionInjector":
        """Build from an OmegaConf config node that must contain tau_f and tau_A.

        Args:
            unet: UNet module with decoder up_blocks.
            cfg:  config node; missing keys raise omegaconf.errors.ConfigAttributeError.

        Returns:
            AttentionInjector ready for hook registration.
        """
        tau_f = cfg.tau_f  # raises ConfigAttributeError if absent
        tau_A = cfg.tau_A  # raises ConfigAttributeError if absent
        return cls(unet, float(tau_f), float(tau_A))

    # ------------------------------------------------------------------
    # Module discovery (called once at construction, before any hooks)
    # ------------------------------------------------------------------

    def _find_feature_module(self) -> nn.Module:
        """Return up_blocks[1].resnets[0] as the f^4_t proxy layer.

        § 4 of PnP uses output_blocks[4] (LDM notation) for spatial features.
        In diffusers up_blocks, this corresponds to up_blocks[1].resnets[0]
        (16×16 spatial resolution for a 512px image), NOT up_blocks[0] (8×8).
        """
        return self._unet.up_blocks[1].resnets[0]  # type: ignore[index]

    def _find_attn1_modules(self) -> list[nn.Module]:
        """Collect every self-attention (attn1) module inside up_blocks[1..N].

        up_blocks[0] (coarsest, 8×8) is intentionally excluded, matching the
        official PnP which injects into output_blocks[4..11] only.

        Returns:
            Ordered list of attn1 nn.Modules from up_blocks[1] onward.
        """
        result: list[nn.Module] = []
        for block in list(self._unet.up_blocks)[1:]:  # type: ignore[attr-defined]
            for name, module in block.named_modules():
                if name == "attn1" or name.endswith(".attn1"):
                    result.append(module)
        return result

    # ------------------------------------------------------------------
    # Hook registration / removal
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        """Attach forward output hooks to the UNet decoder.

        Registers:
          • One output hook on the spatial feature ResNet block.
          • Two output hooks per attn1 (on to_q and to_k) for every module
            in up_blocks[1..N].

        Idempotent: calling this a second time removes the previous set of
        hooks before re-registering.
        """
        self.remove_hooks()  # idempotency guard

        handle_f = self._feature_module.register_forward_hook(self._feature_hook)
        self._hook_handles.append(handle_f)

        for idx, attn1 in enumerate(self._attn1_modules):
            h_q = attn1.to_q.register_forward_hook(self._make_q_hook(idx))
            h_k = attn1.to_k.register_forward_hook(self._make_k_hook(idx))
            self._hook_handles.extend([h_q, h_k])

        logger.debug(
            "AttentionInjector: {} hooks registered (1 feature + {} q/k pairs for {} attn1s)",
            len(self._hook_handles),
            len(self._attn1_modules),
            len(self._attn1_modules),
        )

    def remove_hooks(self) -> None:
        """Remove all registered hooks and clear the handle list."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------
    # Gate predicates  (§ 4 Algorithm 1 threshold comparisons)
    # ------------------------------------------------------------------

    def should_inject_features(self, step: int, total_steps: int) -> bool:
        """True iff step / total_steps > τ_f  (strictly greater than)."""
        return step / total_steps > self.tau_f

    def should_inject_attention(self, step: int, total_steps: int) -> bool:
        """True iff step / total_steps > τ_A  (strictly greater than)."""
        return step / total_steps > self.tau_A

    # ------------------------------------------------------------------
    # Ref-pass lifecycle
    # ------------------------------------------------------------------

    def begin_ref_pass(self) -> None:
        """Switch to capture mode.  Call immediately before the ref UNet forward."""
        self._captured = CapturedFeatures()
        self._q_capture_buf = {}
        self._k_capture_buf = {}
        self._mode = "capture"

    def end_ref_pass(self) -> CapturedFeatures:
        """End capture mode, assemble attn_qk pairs, and return captured features.

        Returns:
            CapturedFeatures populated by the most recent ref-pass forward.
        """
        n = len(self._attn1_modules)
        self._captured.attn_qk = [
            (self._q_capture_buf[i], self._k_capture_buf[i]) for i in range(n)
        ]
        self._mode = "idle"
        return self._captured

    @property
    def captured(self) -> CapturedFeatures:
        """The features captured by the most recent ref_pass (may be empty)."""
        return self._captured

    # ------------------------------------------------------------------
    # Gen-pass lifecycle
    # ------------------------------------------------------------------

    def begin_gen_pass(self, step: int, total_steps: int) -> None:
        """Switch to inject mode.  Call immediately before the gen UNet forward.

        Args:
            step:        current denoising step value (e.g. DDIM timestep).
            total_steps: total number of denoising steps T.
        """
        self._step = step
        self._total_steps = total_steps
        self._mode = "inject"

    def end_gen_pass(self) -> None:
        """Return to idle mode after the gen forward completes."""
        self._mode = "idle"

    # ------------------------------------------------------------------
    # Hook callbacks  (called by PyTorch during module.forward())
    # ------------------------------------------------------------------

    def _feature_hook(
        self,
        module: nn.Module,
        inputs: tuple,
        output: torch.Tensor,
    ) -> torch.Tensor | None:
        """Output hook on up_blocks[1].resnets[0] (spatial features f^4_t)."""
        if self._mode == "capture":
            self._captured.spatial_features = output.detach()
            return None

        if self._mode == "inject" and self.should_inject_features(
            self._step, self._total_steps
        ):
            if self._captured.spatial_features is None:
                raise RuntimeError(
                    "AttentionInjector: gen_pass with open feature gate but no "
                    "ref_pass has been run — call begin_ref_pass() first."
                )
            return self._captured.spatial_features

        return None  # idle or gate closed: pass through

    def _make_q_hook(self, idx: int):
        """Return an output hook for to_q of attn1_modules[idx].

        Capture mode  → stores the projected Q tensor.
        Inject mode   → replaces the to_q output with the ref-pass Q.
        """

        def _hook(
            module: nn.Module,
            inputs: tuple,
            output: torch.Tensor,
        ) -> torch.Tensor | None:
            if self._mode == "capture":
                self._q_capture_buf[idx] = output.detach()
                return None

            if self._mode == "inject" and self.should_inject_attention(
                self._step, self._total_steps
            ):
                if idx >= len(self._captured.attn_qk):
                    raise RuntimeError(
                        "AttentionInjector: gen_pass with open attention gate but no "
                        "ref_pass has been run — call begin_ref_pass() first."
                    )
                return self._captured.attn_qk[idx][0]  # stored Q

            return None

        return _hook

    def _make_k_hook(self, idx: int):
        """Return an output hook for to_k of attn1_modules[idx].

        Capture mode  → stores the projected K tensor.
        Inject mode   → replaces the to_k output with the ref-pass K.
        """

        def _hook(
            module: nn.Module,
            inputs: tuple,
            output: torch.Tensor,
        ) -> torch.Tensor | None:
            if self._mode == "capture":
                self._k_capture_buf[idx] = output.detach()
                return None

            if self._mode == "inject" and self.should_inject_attention(
                self._step, self._total_steps
            ):
                if idx >= len(self._captured.attn_qk):
                    raise RuntimeError(
                        "AttentionInjector: gen_pass with open attention gate but no "
                        "ref_pass has been run — call begin_ref_pass() first."
                    )
                return self._captured.attn_qk[idx][1]  # stored K

            return None

        return _hook
