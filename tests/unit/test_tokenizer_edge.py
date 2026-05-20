"""Tests for utils/tokenizer.py — sub-word merging and token index resolution.

All tests use a real CLIP tokenizer loaded from a tiny vocab fixture so that
no model weights are downloaded during CI. The tokenizer is instantiated once
per session via a module-level fixture.
"""

import pytest
from transformers import CLIPTokenizer

from attn_texture.utils.tokenizer import find_keyword_token_indices, tokenize_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clip_tok():
    """Minimal CLIP tokenizer (vocab only, no model weights)."""
    return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


# ---------------------------------------------------------------------------
# tokenize_prompt
# ---------------------------------------------------------------------------


class TestTokenizePrompt:
    def test_returns_list_of_strings(self, clip_tok):
        tokens = tokenize_prompt(clip_tok, "a cat")
        assert isinstance(tokens, list)
        assert all(isinstance(t, str) for t in tokens)

    def test_includes_bos_and_eos(self, clip_tok):
        tokens = tokenize_prompt(clip_tok, "cat")
        # CLIP wraps with <|startoftext|> ... <|endoftext|>
        assert tokens[0] == "<|startoftext|>"
        assert tokens[-1] == "<|endoftext|>"

    def test_simple_word_round_trips(self, clip_tok):
        tokens = tokenize_prompt(clip_tok, "shirt")
        content = [t for t in tokens if t not in ("<|startoftext|>", "<|endoftext|>")]
        assert any("shirt" in t for t in content)

    def test_length_matches_input_ids(self, clip_tok):
        prompt = "a striped silk shirt on a model"
        tokens = tokenize_prompt(clip_tok, prompt)
        ids = clip_tok(prompt, return_tensors="pt").input_ids[0]
        assert len(tokens) == len(ids)


# ---------------------------------------------------------------------------
# find_keyword_token_indices
# ---------------------------------------------------------------------------


class TestFindKeywordTokenIndices:
    def test_single_token_keyword(self, clip_tok):
        prompt = "a photo of a shirt"
        indices = find_keyword_token_indices(clip_tok, prompt, "shirt")
        assert len(indices) >= 1
        tokens = tokenize_prompt(clip_tok, prompt)
        for i in indices:
            assert 0 <= i < len(tokens)

    def test_returns_correct_position(self, clip_tok):
        prompt = "a photo of a shirt"
        indices = find_keyword_token_indices(clip_tok, prompt, "shirt")
        tokens = tokenize_prompt(clip_tok, prompt)
        matched = [tokens[i].rstrip("</w>") for i in indices]
        assert any("shirt" in t for t in matched)

    def test_multi_token_keyword_t_shirt(self, clip_tok):
        # "t-shirt" splits into multiple sub-words in CLIP
        prompt = "a person wearing a t-shirt"
        indices = find_keyword_token_indices(clip_tok, prompt, "t-shirt")
        assert len(indices) >= 1  # at least one fragment found

    def test_keyword_not_in_prompt_returns_empty(self, clip_tok):
        prompt = "a blue sky"
        indices = find_keyword_token_indices(clip_tok, prompt, "shirt")
        assert indices == []

    def test_keyword_appears_twice(self, clip_tok):
        prompt = "a shirt next to another shirt"
        indices = find_keyword_token_indices(clip_tok, prompt, "shirt")
        # Both occurrences should be captured
        assert len(indices) >= 2

    def test_raises_on_empty_keyword(self, clip_tok):
        with pytest.raises(ValueError, match="keyword"):
            find_keyword_token_indices(clip_tok, "a shirt", "")

    def test_raises_on_empty_prompt(self, clip_tok):
        with pytest.raises(ValueError, match="prompt"):
            find_keyword_token_indices(clip_tok, "", "shirt")
