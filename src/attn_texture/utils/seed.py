"""Seed utilities. All seeds must come from config; never inline them (CLAUDE.md §6 rule 10)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Seed every RNG used in the project for full reproducibility.

    Args:
        seed: integer seed value, sourced from config.

    Returns:
        The same *seed* value, so callers can do ``cfg.seed = seed_everything(cfg.seed)``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return seed
