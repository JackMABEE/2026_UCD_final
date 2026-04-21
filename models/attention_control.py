"""
Attention hook infrastructure for Stable Diffusion U-Net.

Two classes are provided:
  AttentionStore  — registers forward hooks that *capture* K/Q tensors
                    produced during a source (reference) forward pass.
  AttentionInjector — registers forward hooks that *replace* (or blend)
                    K/Q tensors during a target forward pass with the
                    stored source tensors.

Usage pattern
-------------
1. Run source image through the pipeline with AttentionStore active.
2. Call injector.load(store) to transfer captured tensors.
3. Run target forward pass with AttentionInjector active.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _layer_name(module: nn.Module, name: str) -> str:
    """Return a stable string key for a module."""
    return name


class _HookHandle:
    """Thin wrapper so callers can remove all hooks via a single .remove()."""

    def __init__(self, handles: list):
        self._handles = handles

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ---------------------------------------------------------------------------
# AttentionStore
# ---------------------------------------------------------------------------

class AttentionStore:
    """
    Captures self-attention Key and Query tensors from a U-Net forward pass.

    Attributes
    ----------
    store : Dict[str, List[Tensor]]
        Maps layer name → list of (K, Q) tuples, one per denoising step.
    """

    def __init__(self, target_layers: Optional[List[str]] = None):
        """
        Parameters
        ----------
        target_layers : list of str, optional
            Substring filters for layer names to hook
            (e.g. ``["up_blocks", "mid_block"]``).
            If None, all ``Attention`` modules are hooked.
        """
        self.target_layers = target_layers
        self.store: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        self._hook_handle: Optional[_HookHandle] = None
        self._step_index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, unet: nn.Module) -> "_HookHandle":
        """Attach forward hooks to all matching Attention sub-modules."""
        self.reset()
        handles = []
        for name, module in unet.named_modules():
            if isinstance(module, Attention) and self._should_hook(name):
                handle = module.register_forward_hook(
                    self._make_capture_hook(name)
                )
                handles.append(handle)
        self._hook_handle = _HookHandle(handles)
        return self._hook_handle

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reset(self):
        self.store.clear()
        self._step_index = 0

    def step(self):
        """Call once after each denoising step to advance the step counter."""
        self._step_index += 1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_hook(self, name: str) -> bool:
        if self.target_layers is None:
            return True
        return any(t in name for t in self.target_layers)

    def _make_capture_hook(self, layer_name: str):
        """Return a closure that captures K/Q for *layer_name*."""

        def hook(module: Attention, inputs, output):
            # Re-compute K and Q from the stored hidden state.
            # inputs[0] is the hidden_states tensor fed to this Attention block.
            hidden_states: torch.Tensor = inputs[0]
            batch, seq_len, dim = hidden_states.shape

            # Use the module's own projection weights.
            q = module.to_q(hidden_states)
            k = module.to_k(hidden_states)

            # Reshape to (batch, heads, seq, head_dim) — mirrors diffusers internals.
            head_dim = dim // module.heads
            q = q.view(batch, seq_len, module.heads, head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, module.heads, head_dim).transpose(1, 2)

            self.store[layer_name].append((k.detach().clone(), q.detach().clone()))

        return hook


# ---------------------------------------------------------------------------
# AttentionInjector
# ---------------------------------------------------------------------------

class AttentionInjector:
    """
    Injects (replaces or blends) Key/Query tensors captured by AttentionStore
    into a subsequent forward pass.

    Parameters
    ----------
    store : AttentionStore
        A store that has already been populated by a source forward pass.
    injection_steps : list of int
        Denoising step indices (0-based) during which injection is active.
    injection_mode : str
        ``"replace"`` — overwrite target K/Q with source K/Q.
        ``"blend"``   — linear mix: ``alpha * src + (1-alpha) * tgt``.
    blend_alpha : float
        Weight of the source component when ``injection_mode == "blend"``.
    """

    def __init__(
        self,
        store: AttentionStore,
        injection_steps: List[int],
        injection_mode: str = "replace",
        blend_alpha: float = 1.0,
    ):
        self.store = store
        self.injection_steps = set(injection_steps)
        self.injection_mode = injection_mode
        self.blend_alpha = blend_alpha

        self._hook_handle: Optional[_HookHandle] = None
        self._step_index: int = 0
        # Per-layer cursor: tracks which stored step to read from.
        self._layer_cursors: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, unet: nn.Module) -> "_HookHandle":
        """Attach forward pre-hooks that replace K/Q projections."""
        self.reset()
        handles = []
        for name, module in unet.named_modules():
            if isinstance(module, Attention) and name in self.store.store:
                handle = module.register_forward_pre_hook(
                    self._make_inject_hook(name)
                )
                handles.append(handle)
        self._hook_handle = _HookHandle(handles)
        return self._hook_handle

    def remove(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def reset(self):
        self._step_index = 0
        self._layer_cursors.clear()

    def step(self):
        self._step_index += 1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _should_inject(self) -> bool:
        return self._step_index in self.injection_steps

    def _make_inject_hook(self, layer_name: str):
        """
        Return a forward *pre*-hook that monkey-patches ``to_k`` / ``to_q``
        for one forward call so the stored tensors are returned instead.
        """

        def hook(module: Attention, inputs):
            if not self._should_inject():
                return  # pass-through

            cursor = self._layer_cursors[layer_name]
            stored = self.store.store.get(layer_name, [])
            if cursor >= len(stored):
                return  # no stored tensor for this step

            src_k, src_q = stored[cursor]
            self._layer_cursors[layer_name] += 1

            hidden_states: torch.Tensor = inputs[0]
            batch, seq_len, dim = hidden_states.shape
            head_dim = dim // module.heads

            def _patched_to_k(x, src=src_k, orig=module.to_k):
                tgt_k = orig(x)
                tgt_k = tgt_k.view(batch, seq_len, module.heads, head_dim).transpose(1, 2)
                if self.injection_mode == "replace":
                    result = src_k.to(tgt_k.device, tgt_k.dtype)
                else:
                    result = (
                        self.blend_alpha * src_k.to(tgt_k.device, tgt_k.dtype)
                        + (1.0 - self.blend_alpha) * tgt_k
                    )
                return result.transpose(1, 2).contiguous().view(batch, seq_len, dim)

            def _patched_to_q(x, src=src_q, orig=module.to_q):
                tgt_q = orig(x)
                tgt_q = tgt_q.view(batch, seq_len, module.heads, head_dim).transpose(1, 2)
                if self.injection_mode == "replace":
                    result = src_q.to(tgt_q.device, tgt_q.dtype)
                else:
                    result = (
                        self.blend_alpha * src_q.to(tgt_q.device, tgt_q.dtype)
                        + (1.0 - self.blend_alpha) * tgt_q
                    )
                return result.transpose(1, 2).contiguous().view(batch, seq_len, dim)

            # Temporarily replace projections for this one forward call.
            original_to_k = module.to_k
            original_to_q = module.to_q
            module.to_k = _patched_to_k  # type: ignore[method-assign]
            module.to_q = _patched_to_q  # type: ignore[method-assign]

            # Restore after the forward is done via a post-hook.
            def restore_hook(m, inp, out, ok=original_to_k, oq=original_to_q):
                m.to_k = ok
                m.to_q = oq

            restore_handle = module.register_forward_hook(restore_hook)
            # The restore hook removes itself after firing once.
            original_restore = restore_hook

            def self_removing_restore(m, inp, out, ok=original_to_k, oq=original_to_q, h=restore_handle):
                m.to_k = ok
                m.to_q = oq
                h.remove()

            restore_handle.remove()
            module.register_forward_hook(self_removing_restore)

        return hook
