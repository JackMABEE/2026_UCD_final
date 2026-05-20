"""Tests for utils/seed.py."""

import os
import random

import numpy as np
import pytest
import torch

from attn_texture.utils.seed import seed_everything


class TestSeedEverything:
    def test_returns_the_seed(self):
        assert seed_everything(42) == 42

    def test_torch_cpu_reproducible(self):
        seed_everything(7)
        a = torch.rand(8)
        seed_everything(7)
        b = torch.rand(8)
        assert torch.equal(a, b)

    def test_numpy_reproducible(self):
        seed_everything(7)
        a = np.random.rand(8)
        seed_everything(7)
        b = np.random.rand(8)
        assert np.array_equal(a, b)

    def test_stdlib_random_reproducible(self):
        seed_everything(7)
        a = [random.random() for _ in range(8)]
        seed_everything(7)
        b = [random.random() for _ in range(8)]
        assert a == b

    def test_different_seeds_differ(self):
        seed_everything(1)
        a = torch.rand(8)
        seed_everything(2)
        b = torch.rand(8)
        assert not torch.equal(a, b)

    def test_sets_pythonhashseed_env(self):
        seed_everything(99)
        assert os.environ.get("PYTHONHASHSEED") == "99"
