"""Tests for utils/memory.py — with_isolated_model() context manager."""

import gc
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from attn_texture.utils.memory import with_isolated_model


class FakeModel:
    """Stand-in for a diffusion pipeline — tracks whether it's been deleted."""

    def __init__(self):
        self.alive = True

    def __del__(self):
        self.alive = False


class TestWithIsolatedModel:
    def test_yields_the_model(self):
        model = FakeModel()
        factory = MagicMock(return_value=model)
        with with_isolated_model(factory) as m:
            assert m is model

    def test_factory_called_exactly_once(self):
        factory = MagicMock(return_value=FakeModel())
        with with_isolated_model(factory):
            pass
        factory.assert_called_once()

    def test_gc_collect_called_on_exit(self):
        factory = MagicMock(return_value=FakeModel())
        with patch("gc.collect") as mock_gc:
            with with_isolated_model(factory):
                pass
        mock_gc.assert_called()

    def test_empty_cache_called_on_mps_exit(self):
        factory = MagicMock(return_value=FakeModel())
        with patch("torch.backends.mps.is_available", return_value=True), \
             patch("torch.cuda.is_available", return_value=False), \
             patch("torch.mps.empty_cache") as mock_mps:
            with with_isolated_model(factory):
                pass
        mock_mps.assert_called_once()

    def test_empty_cache_called_on_cuda_exit(self):
        factory = MagicMock(return_value=FakeModel())
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.backends.mps.is_available", return_value=False), \
             patch("torch.cuda.empty_cache") as mock_cuda:
            with with_isolated_model(factory):
                pass
        mock_cuda.assert_called_once()

    def test_cleanup_runs_even_on_exception(self):
        factory = MagicMock(return_value=FakeModel())
        with patch("gc.collect") as mock_gc:
            with pytest.raises(RuntimeError):
                with with_isolated_model(factory):
                    raise RuntimeError("boom")
        mock_gc.assert_called()

    def test_model_reference_released_on_exit(self):
        """The context manager must not hold a reference that prevents GC."""
        import weakref

        refs: list[weakref.ref] = []

        def factory():
            m = FakeModel()
            refs.append(weakref.ref(m))
            return m

        with with_isolated_model(factory):
            pass

        gc.collect()
        assert refs[0]() is None, "Context manager leaked a reference to the model"
