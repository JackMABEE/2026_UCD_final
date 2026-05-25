"""Cross-attention mask extraction for Phase 2 region-aware texture transfer.

Implements the spatial mask extraction strategy from Prompt-to-Prompt
(Hertz et al., ICLR 2023) §3.1-3.2, cross-checked against the official
source at github.com/google/prompt-to-prompt.

The mask localises a garment region (e.g. "shirt") by accumulating cross-
attention maps from a dry-run inference pass, selecting maps that respond
to the target token(s), and thresholding to a binary spatial mask.

Dry-run pipeline
----------------
1. word_inds = get_word_inds(tokenizer, prompt, word)
2. cleanup  = register_attention_control(unet, store)
3. Run N denoising steps; after each step call store.between_steps().
4. cleanup()
5. avg  = store.get_average_attention()
6. agg  = aggregate_attention(avg, res=16)          # 16×16 maps for 512px SD 1.5
7. mask = build_mask(agg, word_inds, threshold=0.3, target_hw=(64, 64))

Key deviations from P2P source adapted for our codebase:
  • get_word_inds delegates to utils/tokenizer.find_keyword_token_indices so
    that BOS-offset and sub-word handling are handled consistently (CLAUDE.md §6 rule 1).
  • register_attention_control only patches attn2 (cross-attn) layers; attn1
    (self-attn) processors are left untouched (mirrors MasaCtrl's attn1-only strategy).
  • AttentionStore stores maps keyed by UNet block position ("down", "mid", "up")
    rather than the P2P "down_cross" / "up_cross" naming, for consistency with
    the place labels propagated by register_attention_control.

References
----------
P2P: Hertz et al., "Prompt-to-Prompt Image Editing with Cross Attention Control",
     ICLR 2023 (arXiv:2208.01626).
P2P source: AttentionStore, aggregate_attention, LocalBlend.__call__ in
            google/prompt-to-prompt/ptp_utils.py and prompt-to-prompt.ipynb.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F
from loguru import logger

from attn_texture.utils.tokenizer import find_keyword_token_indices

if TYPE_CHECKING:
    from transformers import CLIPTokenizer


# ---------------------------------------------------------------------------
# 1. Token index lookup
# ---------------------------------------------------------------------------


def get_word_inds(
    tokenizer: "CLIPTokenizer",
    prompt: str,
    word: str,
) -> list[int]:
    """Return token indices in *prompt* that correspond to *word*.

    Thin wrapper around utils/tokenizer.find_keyword_token_indices with
    domain-aligned naming for the P2P mask extraction pipeline.  Handles
    sub-word splits (e.g. "t-shirt" → multiple tokens) consistently with
    the rest of the codebase (CLAUDE.md §6 rule 1).

    Args:
        tokenizer: CLIP tokenizer instance.
        prompt:    Full generation prompt (e.g. "a person wearing a blue shirt").
        word:      Target garment word (e.g. "shirt").

    Returns:
        Sorted list of integer indices into the full token sequence (0-based,
        including BOS token at position 0).  Empty list if word not found.

    Raises:
        ValueError: if prompt or word is empty.
    """
    return find_keyword_token_indices(tokenizer, prompt, word)


# ---------------------------------------------------------------------------
# 2. Attention accumulator
# ---------------------------------------------------------------------------


class AttentionStore:
    """Accumulates cross-attention maps across denoising steps.

    Maps are recorded via _AttentionStoreProcessor instances installed by
    register_attention_control on every attn2 layer.  Call between_steps()
    once after each denoising step; call get_average_attention() after all
    steps to retrieve the time-averaged maps.

    Only maps whose spatial side length ≤ max_spatial_side are kept.
    P2P §3.2 uses layers with spatial resolution up to 32×32; for SD 1.5
    at 512px, 16×16 maps carry the most semantically useful attention.

    Maps are keyed by UNet block position ("down", "mid", "up") and stored
    as a list of (B*heads, N_spatial, N_tokens) tensors per position.

    Args:
        max_spatial_side: Largest spatial side to retain (default 32).
    """

    _PLACES: tuple[str, ...] = ("down", "mid", "up")

    def __init__(self, max_spatial_side: int = 32) -> None:
        self._max_spatial = max_spatial_side
        self._step_maps: dict[str, list[torch.Tensor]] = self._empty_store()
        self._cumulative: dict[str, list[torch.Tensor]] = {}
        self._num_steps: int = 0

    # ------------------------------------------------------------------

    def _empty_store(self) -> dict[str, list[torch.Tensor]]:
        return {p: [] for p in self._PLACES}

    def record(self, attn_weights: torch.Tensor, place: str) -> None:
        """Record one cross-attention map for the current step.

        Args:
            attn_weights: attention probabilities, shape (B*heads, N_spatial, N_tokens).
                          Maps whose implied spatial side > max_spatial_side are dropped.
            place:        UNet block label — one of "down", "mid", "up".

        Raises:
            ValueError: if place is not one of the allowed labels.
        """
        if place not in self._PLACES:
            raise ValueError(
                f"place must be one of {self._PLACES!r}, got '{place}'"
            )
        spatial_side = int(attn_weights.shape[-2] ** 0.5 + 0.5)
        if spatial_side <= self._max_spatial:
            self._step_maps[place].append(attn_weights.detach())

    def between_steps(self) -> None:
        """Accumulate this step's maps into the running sum.

        Must be called once after each denoising step's UNet forward.
        Resets the per-step buffer so that the next step starts clean.
        """
        if not self._cumulative:
            # First step: clone step_maps to initialise the cumulative sum.
            self._cumulative = {
                p: [t.clone() for t in ts]
                for p, ts in self._step_maps.items()
            }
        else:
            for place in self._PLACES:
                step_ts = self._step_maps[place]
                cum_ts  = self._cumulative[place]
                if len(step_ts) != len(cum_ts):
                    raise RuntimeError(
                        f"AttentionStore: step produced {len(step_ts)} maps for "
                        f"place='{place}' but cumulative has {len(cum_ts)}. "
                        "Ensure register_attention_control is called once and "
                        "between_steps() is called exactly once per denoising step."
                    )
                for i, t in enumerate(step_ts):
                    cum_ts[i] = cum_ts[i] + t

        self._num_steps += 1
        self._step_maps = self._empty_store()

    def get_average_attention(self) -> dict[str, list[torch.Tensor]]:
        """Return time-averaged cross-attention maps.

        Returns:
            Dict mapping place → list of (B*heads, N_spatial, N_tokens) tensors
            in float32.

        Raises:
            RuntimeError: if between_steps() has never been called.
        """
        if self._num_steps == 0:
            raise RuntimeError(
                "AttentionStore: no steps accumulated — call between_steps() "
                "at least once before get_average_attention()."
            )
        return {
            place: [t.float() / self._num_steps for t in ts]
            for place, ts in self._cumulative.items()
        }


# ---------------------------------------------------------------------------
# 3. Custom cross-attention processor for capturing maps
# ---------------------------------------------------------------------------


class _AttentionStoreProcessor:
    """Attention processor that records cross-attention weights in AttentionStore.

    Installed on attn2 (cross-attention) layers only by register_attention_control.
    Replicates standard AttnProcessor behaviour (identical to _MasaCtrlAttnProcessor's
    cross-attention path) while forwarding the attention probability matrix to store.

    Args:
        store: AttentionStore instance to receive maps.
        place: UNet block label passed through to store.record().
    """

    def __init__(self, store: AttentionStore, place: str) -> None:
        self._store = store
        self._place = place

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run cross-attention and record the attention probability matrix.

        Args:
            attn:                  diffusers Attention module.
            hidden_states:         (B, N_spatial, D_model) spatial query sequence.
            encoder_hidden_states: (B, N_tokens, D_text) text embeddings; falls back
                                   to hidden_states for self-attention (should not
                                   happen since this processor is attn2-only).
            attention_mask:        optional additive mask for get_attention_scores.

        Returns:
            (B, N_spatial, D_model) attended output.
        """
        kv_src = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = attn.to_q(hidden_states)
        k = attn.to_k(kv_src)
        v = attn.to_v(kv_src)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        # attn_weights: (B*heads, N_spatial, N_tokens)  — P2P §3.2 Eq. (2)
        attn_weights = attn.get_attention_scores(q, k, attention_mask)
        self._store.record(attn_weights, self._place)

        hidden_states = torch.bmm(attn_weights, v)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# 4. Spatial aggregation
# ---------------------------------------------------------------------------


def aggregate_attention(
    avg_attention: dict[str, list[torch.Tensor]],
    res: int,
    places: tuple[str, ...] = ("down", "up"),
) -> torch.Tensor:
    """Aggregate cross-attention maps at a given spatial resolution.

    Selects maps whose N_spatial dimension equals res*res, averages across
    the B*heads dimension and across selected layers, yielding a single
    (res, res, num_tokens) tensor.

    The default place selection ("down", "up") matches P2P LocalBlend §3.2:
    mid-resolution down + up layers balance spatial precision and semantic
    coherence; the mid-block alone is too coarse.

    Args:
        avg_attention: time-averaged maps from AttentionStore.get_average_attention().
                       Each value is a list of (B*heads, N_spatial, N_tokens) tensors.
        res:           Spatial side length to select (e.g. 16 for 16×16 maps).
        places:        Which block groups to include ("down", "mid", "up").

    Returns:
        (res, res, num_tokens) float32 tensor averaged over heads and selected layers.

    Raises:
        ValueError: if no maps exist at the requested resolution within *places*.
    """
    maps: list[torch.Tensor] = []
    for place in places:
        for attn_map in avg_attention.get(place, []):
            if attn_map.shape[-2] == res * res:
                # Average over B*heads → (res*res, N_tokens) → reshape (res, res, N_tokens)
                maps.append(attn_map.float().mean(0).reshape(res, res, -1))

    if not maps:
        raise ValueError(
            f"aggregate_attention: no maps found at resolution {res}×{res} "
            f"in places={places}. Verify AttentionStore.max_spatial_side ≥ {res} "
            "and that between_steps() was called after every denoising step."
        )

    return torch.stack(maps, dim=0).mean(0)  # (res, res, num_tokens)


# ---------------------------------------------------------------------------
# 5. Mask generation
# ---------------------------------------------------------------------------


def build_mask(
    aggregated: torch.Tensor,
    word_inds: list[int],
    threshold: float = 0.3,
    target_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Generate a binary spatial mask for the given token indices.

    Implements the LocalBlend mask computation from P2P §3.2 / ptp_utils.py:
      word_map  = Σ aggregated[:, :, i] for i in word_inds
      smoothed  = max_pool2d(word_map, kernel_size=3, stride=1, padding=1)
      norm      = smoothed / smoothed.max()             (no-op if max == 0)
      mask      = norm > threshold                      (binary)

    If target_hw is given, the float map is bilinearly upsampled before
    thresholding so the mask aligns with the latent spatial dimensions.

    Args:
        aggregated:  (res_h, res_w, num_tokens) float tensor from aggregate_attention().
        word_inds:   Token indices for the target garment word.  Out-of-range
                     indices are silently ignored.
        threshold:   Binarisation threshold in [0, 1]; default 0.3 per P2P paper.
        target_hw:   (H, W) to bilinearly upsample the mask before thresholding
                     (e.g. latent spatial size (64, 64) for a 512px image).

    Returns:
        Boolean tensor, shape (res_h, res_w) or target_hw.

    Raises:
        ValueError: if word_inds is empty.
    """
    if not word_inds:
        raise ValueError("build_mask: word_inds is empty — no token indices to mask.")

    res_h, res_w, num_tokens = aggregated.shape

    # One-hot over the token dimension; out-of-range indices produce 0.  (§3.2 α_t)
    alpha = torch.zeros(num_tokens, device=aggregated.device, dtype=aggregated.dtype)
    for i in word_inds:
        if 0 <= i < num_tokens:
            alpha[i] = 1.0

    # Sum over selected tokens → (res_h, res_w)
    word_map = (aggregated * alpha).sum(-1)

    # Smooth with max-pool (k=3) to suppress isolated pixels  — LocalBlend source
    smoothed = F.max_pool2d(
        word_map.unsqueeze(0).unsqueeze(0),  # (1, 1, res_h, res_w)
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(0).squeeze(0)  # (res_h, res_w)

    # Normalize to [0, 1]
    max_val = smoothed.max()
    if max_val > 0:
        smoothed = smoothed / max_val

    # Upsample to latent spatial size if requested
    if target_hw is not None:
        smoothed = F.interpolate(
            smoothed.unsqueeze(0).unsqueeze(0),  # (1, 1, res_h, res_w)
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # (target_H, target_W)

    return smoothed.gt(threshold)


# ---------------------------------------------------------------------------
# 6. Hook registration utility
# ---------------------------------------------------------------------------


def register_attention_control(
    unet,
    store: AttentionStore,
) -> Callable[[], None]:
    """Replace UNet cross-attention (attn2) processors with _AttentionStoreProcessor.

    Self-attention (attn1) processors are left untouched so that the UNet
    computes normally for all but the attention-map-capture purpose.

    Designed for a single dry-run inference pass:
      cleanup = register_attention_control(unet, store)
      ... run UNet forward(s) calling store.between_steps() after each step ...
      cleanup()  # restore original processors

    Args:
        unet:  Module with .attn_processors (dict) and .set_attn_processor(dict).
               Compatible with diffusers UNet2DConditionModel.
        store: AttentionStore that will receive maps via _AttentionStoreProcessor.

    Returns:
        A zero-argument callable that restores the original processors.
    """
    existing = unet.attn_processors        # full dict: name → current processor
    original = dict(existing)             # snapshot for restore
    new_processors = dict(existing)       # shallow copy; will mutate attn2 entries

    n_replaced = 0
    for name in existing:
        if "attn2" in name:
            place = _place_from_name(name)
            new_processors[name] = _AttentionStoreProcessor(store, place)
            n_replaced += 1

    unet.set_attn_processor(new_processors)
    logger.debug(
        "register_attention_control: replaced {} attn2 processors (store max_spatial={}px)",
        n_replaced,
        store._max_spatial,
    )

    def _cleanup() -> None:
        unet.set_attn_processor(original)
        logger.debug(
            "register_attention_control: restored {} original processors",
            len(original),
        )

    return _cleanup


def _place_from_name(name: str) -> str:
    """Derive the UNet block label from an attn_processors dict key.

    Args:
        name: processor key, e.g. "down_blocks.0.attentions.0...attn2.processor".

    Returns:
        "down", "mid", or "up".
    """
    if "down_block" in name:
        return "down"
    if "up_block" in name:
        return "up"
    return "mid"  # mid_block and any unexpected path
