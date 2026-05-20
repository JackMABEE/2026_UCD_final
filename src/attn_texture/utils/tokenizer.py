"""Tokenizer utilities for dynamic keyword-to-token-index resolution.

Never hardcode token indices (CLAUDE.md §6 rule 1). Always derive positions
through find_keyword_token_indices().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import CLIPTokenizer


def tokenize_prompt(tokenizer: "CLIPTokenizer", prompt: str) -> list[str]:
    """Encode *prompt* and return the full string-token list including BOS/EOS.

    The returned list length matches tokenizer(prompt).input_ids[0] exactly,
    so index i in the list corresponds to attention position i.

    Args:
        tokenizer: a CLIPTokenizer instance.
        prompt: text to encode.

    Returns:
        List of string tokens, e.g. ["<|startoftext|>", "a</w>", "shirt</w>", "<|endoftext|>"].
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    return tokenizer.convert_ids_to_tokens(input_ids.tolist())


def find_keyword_token_indices(
    tokenizer: "CLIPTokenizer",
    prompt: str,
    keyword: str,
) -> list[int]:
    """Return every token index (in the full prompt sequence) that belongs to *keyword*.

    Handles sub-word splits: "t-shirt" may become ["t", "-", "shirt</w>"] in
    the CLIP vocab; all three positions are returned. See CLAUDE.md §6 rule 1
    and §6 Core Algorithm Dynamic Masking note on tokenizer splits.

    Strategy: tokenize *keyword* without special tokens, then scan the
    prompt's token sequence for every contiguous match of that sub-sequence.

    Args:
        tokenizer: a CLIPTokenizer instance.
        prompt: the full generation prompt.
        keyword: the word/phrase to locate (e.g. "shirt", "t-shirt").

    Returns:
        Sorted list of integer indices into the prompt token sequence.
        Empty list if keyword is not found.

    Raises:
        ValueError: if *prompt* or *keyword* is empty/whitespace.
    """
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if not keyword.strip():
        raise ValueError("keyword must not be empty")

    prompt_tokens = tokenize_prompt(tokenizer, prompt)

    # Tokenize keyword without BOS/EOS so we get only its sub-words.
    kw_ids = tokenizer(keyword, add_special_tokens=False).input_ids
    kw_tokens = tokenizer.convert_ids_to_tokens(kw_ids)

    if not kw_tokens:
        return []

    # Sliding-window subsequence search over the prompt token list.
    n, k = len(prompt_tokens), len(kw_tokens)
    indices: list[int] = []
    for i in range(n - k + 1):
        if prompt_tokens[i : i + k] == kw_tokens:
            indices.extend(range(i, i + k))

    return indices
