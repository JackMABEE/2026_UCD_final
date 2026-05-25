"""Tests for core/mask_extraction.py — cross-attention mask extraction.

All tests use CPU mocks — no real model weights downloaded.

Public contract under test
--------------------------
get_word_inds(tokenizer, prompt, word) -> list[int]
    Thin wrapper around utils/tokenizer.find_keyword_token_indices.

AttentionStore
    record(attn_weights, place): store (B*heads, N_spatial, N_tokens) map for current step.
    between_steps():             accumulate step maps → running sum; reset step_maps.
    get_average_attention():     return cumulative / num_steps.

aggregate_attention(avg_attention, res, places) -> Tensor (res, res, N_tokens)
    Select maps at spatial resolution res×res, average over heads and layers.

build_mask(aggregated, word_inds, threshold, target_hw) -> BoolTensor
    One-hot on token dim → word_map → max_pool3 → normalize → threshold → upsample.

register_attention_control(unet, store) -> cleanup()
    Replace attn2 processors with _AttentionStoreProcessor; return restore callable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

from attn_texture.core.mask_extraction import (
    AttentionStore,
    aggregate_attention,
    build_mask,
    get_word_inds,
    register_attention_control,
    _AttentionStoreProcessor,
)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

_RES = 8          # spatial side for fast tests (8×8 maps)
_N_TOKENS = 12    # CLIP sequence tokens for mock
_N_HEADS = 4      # attention heads

_DOWN_NAME = "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor"
_MID_NAME  = "mid_block.attentions.0.transformer_blocks.0.attn2.processor"
_UP_NAME   = "up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor"
_SELF_NAME = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor"

_ALL_NAMES = [_DOWN_NAME, _MID_NAME, _UP_NAME, _SELF_NAME]
_CROSS_NAMES = [_DOWN_NAME, _MID_NAME, _UP_NAME]
_SELF_NAMES  = [_SELF_NAME]


def _make_attn_map(spatial_side: int = _RES, n_tokens: int = _N_TOKENS) -> torch.Tensor:
    """Fake (B*heads, N_spatial, N_tokens) attention probability map."""
    n_spatial = spatial_side * spatial_side
    return torch.rand(_N_HEADS, n_spatial, n_tokens)


def _make_avg_attention(
    spatial_side: int = _RES,
    n_tokens: int = _N_TOKENS,
    layers_per_place: int = 2,
) -> dict[str, list[torch.Tensor]]:
    """Build a minimal avg_attention dict usable by aggregate_attention."""
    return {
        "down": [_make_attn_map(spatial_side, n_tokens) for _ in range(layers_per_place)],
        "mid":  [_make_attn_map(spatial_side, n_tokens) for _ in range(layers_per_place)],
        "up":   [_make_attn_map(spatial_side, n_tokens) for _ in range(layers_per_place)],
    }


# ---------------------------------------------------------------------------
# Mock UNet for register_attention_control tests
# ---------------------------------------------------------------------------


class _MockUNet:
    """Minimal stub with attn_processors / set_attn_processor support."""

    def __init__(self) -> None:
        self._procs: dict = {name: MagicMock() for name in _ALL_NAMES}
        self._set_calls: list[dict] = []

    @property
    def attn_processors(self) -> dict:
        return dict(self._procs)

    def set_attn_processor(self, processors: dict) -> None:
        self._procs = dict(processors)
        self._set_calls.append(dict(processors))


# ---------------------------------------------------------------------------
# 1. TestGetWordInds
# ---------------------------------------------------------------------------


class TestGetWordInds:
    """get_word_inds delegates to find_keyword_token_indices from utils/tokenizer."""

    def test_delegates_to_find_keyword_token_indices(self):
        mock_tok = MagicMock()
        target_indices = [3, 4]
        with patch(
            "attn_texture.core.mask_extraction.find_keyword_token_indices",
            return_value=target_indices,
        ) as mock_fn:
            result = get_word_inds(mock_tok, "a person in a shirt", "shirt")
            mock_fn.assert_called_once_with(mock_tok, "a person in a shirt", "shirt")
        assert result == target_indices

    def test_returns_list(self):
        mock_tok = MagicMock()
        with patch(
            "attn_texture.core.mask_extraction.find_keyword_token_indices",
            return_value=[2],
        ):
            result = get_word_inds(mock_tok, "a shirt", "shirt")
        assert isinstance(result, list)

    def test_empty_list_when_word_absent(self):
        mock_tok = MagicMock()
        with patch(
            "attn_texture.core.mask_extraction.find_keyword_token_indices",
            return_value=[],
        ):
            result = get_word_inds(mock_tok, "a blue sky", "shirt")
        assert result == []

    def test_propagates_value_error_on_empty_prompt(self):
        mock_tok = MagicMock()
        with patch(
            "attn_texture.core.mask_extraction.find_keyword_token_indices",
            side_effect=ValueError("prompt must not be empty"),
        ):
            with pytest.raises(ValueError, match="prompt"):
                get_word_inds(mock_tok, "", "shirt")

    def test_propagates_value_error_on_empty_word(self):
        mock_tok = MagicMock()
        with patch(
            "attn_texture.core.mask_extraction.find_keyword_token_indices",
            side_effect=ValueError("keyword must not be empty"),
        ):
            with pytest.raises(ValueError):
                get_word_inds(mock_tok, "a shirt", "")


# ---------------------------------------------------------------------------
# 2. TestAttentionStore
# ---------------------------------------------------------------------------


class TestAttentionStore:
    """AttentionStore accumulates cross-attention maps across denoising steps."""

    @pytest.fixture()
    def store(self) -> AttentionStore:
        return AttentionStore(max_spatial_side=32)

    def test_initial_step_maps_all_empty(self, store: AttentionStore):
        for place in ("down", "mid", "up"):
            assert store._step_maps[place] == []

    def test_record_adds_to_step_maps(self, store: AttentionStore):
        attn = _make_attn_map(spatial_side=16)
        store.record(attn, "down")
        assert len(store._step_maps["down"]) == 1

    def test_record_detaches_from_graph(self, store: AttentionStore):
        attn = _make_attn_map().requires_grad_(True)
        store.record(attn.detach(), "down")
        assert not store._step_maps["down"][0].requires_grad

    def test_record_rejects_unknown_place(self, store: AttentionStore):
        attn = _make_attn_map()
        with pytest.raises(ValueError, match="place"):
            store.record(attn, "encoder")

    def test_record_filters_large_spatial(self, store: AttentionStore):
        """Maps larger than max_spatial_side should be silently discarded."""
        large = _make_attn_map(spatial_side=64)  # 64 > 32
        store.record(large, "down")
        assert len(store._step_maps["down"]) == 0

    def test_record_keeps_exactly_at_max(self, store: AttentionStore):
        at_max = _make_attn_map(spatial_side=32)  # 32 == 32
        store.record(at_max, "down")
        assert len(store._step_maps["down"]) == 1

    def test_between_steps_clears_step_maps(self, store: AttentionStore):
        store.record(_make_attn_map(), "down")
        store.between_steps()
        assert store._step_maps["down"] == []

    def test_between_steps_increments_num_steps(self, store: AttentionStore):
        store.record(_make_attn_map(), "down")
        store.between_steps()
        assert store._num_steps == 1

    def test_single_step_average_equals_map(self, store: AttentionStore):
        attn = _make_attn_map()
        store.record(attn, "down")
        store.between_steps()
        avg = store.get_average_attention()
        assert torch.allclose(avg["down"][0], attn.float())

    def test_two_step_average_is_mean(self, store: AttentionStore):
        torch.manual_seed(0)
        a1 = _make_attn_map()
        a2 = _make_attn_map()
        store.record(a1, "down")
        store.between_steps()
        store.record(a2, "down")
        store.between_steps()
        avg = store.get_average_attention()
        expected = ((a1 + a2) / 2.0).float()
        assert torch.allclose(avg["down"][0], expected, atol=1e-6)

    def test_get_average_before_steps_raises(self, store: AttentionStore):
        with pytest.raises(RuntimeError):
            store.get_average_attention()

    def test_multiple_places_recorded_independently(self, store: AttentionStore):
        store.record(_make_attn_map(), "down")
        store.record(_make_attn_map(), "up")
        store.between_steps()
        avg = store.get_average_attention()
        assert len(avg["down"]) == 1
        assert len(avg["up"])   == 1
        assert len(avg["mid"])  == 0


# ---------------------------------------------------------------------------
# 3. TestAggregateAttention
# ---------------------------------------------------------------------------


class TestAggregateAttention:
    """aggregate_attention selects maps at a given resolution and averages them."""

    def test_output_shape_is_res_res_tokens(self):
        avg = _make_avg_attention(spatial_side=_RES, n_tokens=_N_TOKENS)
        result = aggregate_attention(avg, res=_RES, places=("down",))
        assert result.shape == (_RES, _RES, _N_TOKENS)

    def test_raises_if_no_maps_at_resolution(self):
        avg = _make_avg_attention(spatial_side=8)
        with pytest.raises(ValueError, match="16"):
            aggregate_attention(avg, res=16, places=("down",))

    def test_selects_only_requested_resolution(self):
        # Mix 8×8 and 16×16 maps in "down"; request 8
        avg = {
            "down": [_make_attn_map(spatial_side=8), _make_attn_map(spatial_side=16)],
            "mid": [],
            "up": [],
        }
        result = aggregate_attention(avg, res=8, places=("down",))
        # Should have used only the 8×8 map (1 layer)
        assert result.shape == (8, 8, _N_TOKENS)

    def test_averages_over_heads_dim(self):
        # 1 layer, constant across all heads → average same as any head slice
        torch.manual_seed(1)
        single_map = torch.ones(_N_HEADS, _RES * _RES, _N_TOKENS)
        avg = {"down": [single_map], "mid": [], "up": []}
        result = aggregate_attention(avg, res=_RES, places=("down",))
        assert result.shape == (_RES, _RES, _N_TOKENS)
        assert torch.allclose(result, torch.ones(_RES, _RES, _N_TOKENS))

    def test_averages_across_layers(self):
        # Two identical constant maps → average is same constant
        const = torch.full((_N_HEADS, _RES * _RES, _N_TOKENS), 0.5)
        avg = {"down": [const.clone(), const.clone()], "mid": [], "up": []}
        result = aggregate_attention(avg, res=_RES, places=("down",))
        assert torch.allclose(result, torch.full((_RES, _RES, _N_TOKENS), 0.5))

    def test_places_parameter_filters_blocks(self):
        avg = {
            "down": [_make_attn_map(spatial_side=8)],
            "mid":  [_make_attn_map(spatial_side=8)],
            "up":   [],
        }
        # Request only "up" — no maps there at res=8
        with pytest.raises(ValueError):
            aggregate_attention(avg, res=8, places=("up",))

    def test_output_dtype_is_float32(self):
        avg = _make_avg_attention(spatial_side=_RES)
        result = aggregate_attention(avg, res=_RES)
        assert result.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. TestBuildMask
# ---------------------------------------------------------------------------


class TestBuildMask:
    """build_mask: one-hot token selection → max-pool → normalize → threshold → upsample."""

    _H = _W = _RES

    def _aggregated(self, val: float = 0.5) -> torch.Tensor:
        return torch.full((self._H, self._W, _N_TOKENS), val)

    def test_output_is_bool(self):
        agg = self._aggregated(0.5)
        mask = build_mask(agg, word_inds=[3], threshold=0.3)
        assert mask.dtype == torch.bool

    def test_output_shape_matches_aggregated(self):
        agg = self._aggregated()
        mask = build_mask(agg, word_inds=[2, 3], threshold=0.3)
        assert mask.shape == (self._H, self._W)

    def test_target_hw_upsample(self):
        agg = self._aggregated()
        mask = build_mask(agg, word_inds=[1], threshold=0.0, target_hw=(64, 64))
        assert mask.shape == (64, 64)

    def test_threshold_zero_gives_all_true_for_nonzero_attention(self):
        # Uniform non-zero attention → after normalization all pixels == 1.0 > 0.0
        agg = self._aggregated(0.5)
        mask = build_mask(agg, word_inds=[2], threshold=0.0)
        assert mask.all()

    def test_threshold_above_one_gives_all_false(self):
        agg = self._aggregated(0.5)
        mask = build_mask(agg, word_inds=[2], threshold=1.001)
        assert not mask.any()

    def test_empty_word_inds_raises(self):
        agg = self._aggregated()
        with pytest.raises(ValueError, match="empty"):
            build_mask(agg, word_inds=[], threshold=0.3)

    def test_out_of_range_inds_handled_gracefully(self):
        """Indices beyond num_tokens dimension are silently ignored (alpha stays 0)."""
        agg = self._aggregated(0.5)
        # index 999 is out of range for _N_TOKENS=12; valid index 0 should still work
        mask = build_mask(agg, word_inds=[0, 999], threshold=0.0)
        assert mask.all()

    def test_high_attention_pixel_in_mask(self):
        # Construct aggregated with token 3 having max attention at one pixel
        torch.manual_seed(2)
        agg = torch.rand(self._H, self._W, _N_TOKENS) * 0.1  # low background
        agg[0, 0, 3] = 1.0  # strong activation at top-left for token 3
        mask = build_mask(agg, word_inds=[3], threshold=0.3)
        # Top-left pixel should be unmasked (True)
        assert mask[0, 0].item()

    def test_all_zero_attention_gives_all_false(self):
        agg = torch.zeros(self._H, self._W, _N_TOKENS)
        mask = build_mask(agg, word_inds=[2], threshold=0.3)
        assert not mask.any()

    def test_normalization_is_applied(self):
        # Two tokens: token 0 has low attention, token 1 has high attention.
        # Selecting only token 0 should produce weak activation → mostly False at threshold=0.5.
        agg = torch.zeros(self._H, self._W, _N_TOKENS)
        agg[:, :, 0] = 0.1  # weak
        agg[:, :, 1] = 1.0  # strong
        mask_weak = build_mask(agg, word_inds=[0], threshold=0.5)
        mask_strong = build_mask(agg, word_inds=[1], threshold=0.5)
        # Strong token should produce more True pixels
        assert mask_strong.sum() >= mask_weak.sum()


# ---------------------------------------------------------------------------
# 5. TestRegisterAttentionControl
# ---------------------------------------------------------------------------


class TestRegisterAttentionControl:
    """register_attention_control patches attn2 processors; cleanup restores originals."""

    @pytest.fixture()
    def unet(self) -> _MockUNet:
        return _MockUNet()

    @pytest.fixture()
    def store(self) -> AttentionStore:
        return AttentionStore()

    def test_returns_callable(self, unet: _MockUNet, store: AttentionStore):
        cleanup = register_attention_control(unet, store)
        assert callable(cleanup)
        cleanup()

    def test_attn2_processors_replaced_with_store_processor(
        self, unet: _MockUNet, store: AttentionStore
    ):
        register_attention_control(unet, store)
        for name in _CROSS_NAMES:
            assert isinstance(unet._procs[name], _AttentionStoreProcessor), (
                f"Expected _AttentionStoreProcessor on '{name}'"
            )

    def test_attn1_processors_unchanged(self, unet: _MockUNet, store: AttentionStore):
        original_self_proc = unet._procs[_SELF_NAME]
        register_attention_control(unet, store)
        assert unet._procs[_SELF_NAME] is original_self_proc

    def test_cleanup_restores_original_processors(self, unet: _MockUNet, store: AttentionStore):
        originals = dict(unet._procs)
        cleanup = register_attention_control(unet, store)
        cleanup()
        for name in _ALL_NAMES:
            assert unet._procs[name] is originals[name], (
                f"Cleanup did not restore original processor for '{name}'"
            )

    def test_store_processor_has_correct_place_down(
        self, unet: _MockUNet, store: AttentionStore
    ):
        register_attention_control(unet, store)
        proc = unet._procs[_DOWN_NAME]
        assert isinstance(proc, _AttentionStoreProcessor)
        assert proc._place == "down"

    def test_store_processor_has_correct_place_mid(
        self, unet: _MockUNet, store: AttentionStore
    ):
        register_attention_control(unet, store)
        proc = unet._procs[_MID_NAME]
        assert isinstance(proc, _AttentionStoreProcessor)
        assert proc._place == "mid"

    def test_store_processor_has_correct_place_up(
        self, unet: _MockUNet, store: AttentionStore
    ):
        register_attention_control(unet, store)
        proc = unet._procs[_UP_NAME]
        assert isinstance(proc, _AttentionStoreProcessor)
        assert proc._place == "up"

    def test_store_processor_references_correct_store(
        self, unet: _MockUNet, store: AttentionStore
    ):
        register_attention_control(unet, store)
        proc = unet._procs[_DOWN_NAME]
        assert isinstance(proc, _AttentionStoreProcessor)
        assert proc._store is store

    def test_second_call_uses_new_store(self, unet: _MockUNet):
        store1 = AttentionStore()
        store2 = AttentionStore()
        cleanup1 = register_attention_control(unet, store1)
        cleanup1()
        register_attention_control(unet, store2)
        assert unet._procs[_DOWN_NAME]._store is store2
