"""Tests for eval/shootout.py — 5-way comparison panel and metrics aggregation.

All metric functions are patched to fixed floats so these tests are fast and
hermetic — the correctness of individual metrics is covered by test_metrics.py.

Contract under test
-------------------
run_shootout(exp_name, original, sdedit, controlnet, pnp_baseline, ours,
             gen_prompt, experiments_root) -> dict[str, dict[str, float]]

    Side effects:
      • experiments_root / exp_name / shootout.png  — 1×5 RGB panel
      • experiments_root / exp_name / metrics.json  — nested dict

    Return value mirrors the JSON structure:
      {"sdedit": {"ssim": …, "psnr": …, "lpips": …, "clip": …, "dino": …},
       "controlnet": {…}, "pnp_baseline": {…}, "ours": {…}}

    The panel has:
      • width  == 5 × source image width
      • height >= source image height  (extra rows may hold labels)
      • mode   == "RGB"
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_W, _H = 32, 32
_METHODS_NON_ORIG = ("sdedit", "controlnet", "pnp_baseline", "ours")
_METRIC_KEYS = frozenset({"ssim", "psnr", "lpips", "clip", "dino"})
_GEN_PROMPT = "a fabric with floral pattern in blue tones"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def images() -> dict[str, Image.Image]:
    return {
        "original":     Image.new("RGB", (_W, _H), (100, 100, 100)),
        "sdedit":       Image.new("RGB", (_W, _H), (110, 110, 110)),
        "controlnet":   Image.new("RGB", (_W, _H), (120, 120, 120)),
        "pnp_baseline": Image.new("RGB", (_W, _H), (80, 80, 80)),
        "ours":         Image.new("RGB", (_W, _H), (90, 90, 90)),
    }


@pytest.fixture(autouse=True)
def mock_metrics():
    """Patch all metric callables so shootout tests are fast and hermetic."""
    with (
        patch("attn_texture.eval.shootout.ssim", return_value=0.85),
        patch("attn_texture.eval.shootout.psnr", return_value=24.5),
        patch("attn_texture.eval.shootout.lpips_score", return_value=0.12),
        patch("attn_texture.eval.shootout.clip_score", return_value=0.75),
        patch("attn_texture.eval.shootout.dino_distance", return_value=0.25),
    ):
        yield


@pytest.fixture()
def result(images, tmp_path) -> dict:
    from attn_texture.eval.shootout import run_shootout

    return run_shootout(
        exp_name="test_run",
        original=images["original"],
        sdedit=images["sdedit"],
        controlnet=images["controlnet"],
        pnp_baseline=images["pnp_baseline"],
        ours=images["ours"],
        gen_prompt=_GEN_PROMPT,
        experiments_root=tmp_path,
    )


@pytest.fixture()
def exp_dir(result, tmp_path) -> Path:
    return tmp_path / "test_run"


# ---------------------------------------------------------------------------
# TestPanelImage
# ---------------------------------------------------------------------------


class TestPanelImage:
    def test_panel_png_saved(self, exp_dir):
        assert (exp_dir / "shootout.png").exists()

    def test_panel_width_is_5x_image_width(self, exp_dir):
        panel = Image.open(exp_dir / "shootout.png")
        assert panel.width == 5 * _W

    def test_panel_height_at_least_image_height(self, exp_dir):
        panel = Image.open(exp_dir / "shootout.png")
        assert panel.height >= _H

    def test_panel_is_rgb(self, exp_dir):
        panel = Image.open(exp_dir / "shootout.png")
        assert panel.mode == "RGB"

    def test_panel_images_are_pasted_in_correct_columns(self, images, tmp_path):
        """Each source image occupies its own column (left-most pixel matches)."""
        from attn_texture.eval.shootout import run_shootout

        colours = {
            "original":     (10, 20, 30),
            "sdedit":       (40, 50, 60),
            "controlnet":   (70, 80, 90),
            "pnp_baseline": (100, 110, 120),
            "ours":         (130, 140, 150),
        }
        imgs = {k: Image.new("RGB", (_W, _H), v) for k, v in colours.items()}

        run_shootout(
            "col_test",
            imgs["original"], imgs["sdedit"], imgs["controlnet"],
            imgs["pnp_baseline"], imgs["ours"],
            gen_prompt=_GEN_PROMPT,
            experiments_root=tmp_path,
        )

        panel = Image.open(tmp_path / "col_test" / "shootout.png")
        label_h = panel.height - _H
        for col_idx, (method, colour) in enumerate(colours.items()):
            x = col_idx * _W
            y = label_h
            r, g, b = panel.getpixel((x, y))[:3]
            assert (r, g, b) == colour, (
                f"Column {col_idx} ({method}): expected {colour}, got {(r, g, b)}"
            )


# ---------------------------------------------------------------------------
# TestMetricsJson
# ---------------------------------------------------------------------------


class TestMetricsJson:
    def test_metrics_json_saved(self, exp_dir):
        assert (exp_dir / "metrics.json").exists()

    def test_json_has_exactly_four_method_keys(self, exp_dir):
        data = json.loads((exp_dir / "metrics.json").read_text())
        assert set(data.keys()) == set(_METHODS_NON_ORIG)

    def test_each_method_has_exactly_five_metric_keys(self, exp_dir):
        data = json.loads((exp_dir / "metrics.json").read_text())
        for method in _METHODS_NON_ORIG:
            assert set(data[method].keys()) == _METRIC_KEYS, (
                f"method '{method}' missing expected metric keys"
            )

    def test_all_metric_values_are_numeric(self, exp_dir):
        data = json.loads((exp_dir / "metrics.json").read_text())
        for method in _METHODS_NON_ORIG:
            for key in _METRIC_KEYS:
                assert isinstance(data[method][key], (int, float)), (
                    f"{method}.{key} is not numeric: {data[method][key]!r}"
                )

    def test_json_is_valid_for_all_finite_values(self, images, tmp_path):
        """metrics.json must be valid JSON even when all values are finite floats."""
        from attn_texture.eval.shootout import run_shootout

        run_shootout(
            "valid_json",
            images["original"], images["sdedit"], images["controlnet"],
            images["pnp_baseline"], images["ours"],
            gen_prompt=_GEN_PROMPT,
            experiments_root=tmp_path,
        )
        raw = (tmp_path / "valid_json" / "metrics.json").read_text()
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# TestReturnValue
# ---------------------------------------------------------------------------


class TestReturnValue:
    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_dict_has_exactly_four_method_keys(self, result):
        assert set(result.keys()) == set(_METHODS_NON_ORIG)

    def test_each_method_has_five_metric_keys(self, result):
        for method in _METHODS_NON_ORIG:
            assert set(result[method].keys()) == _METRIC_KEYS

    def test_metric_values_are_floats(self, result):
        for method in _METHODS_NON_ORIG:
            for key in _METRIC_KEYS:
                assert isinstance(result[method][key], float), (
                    f"{method}.{key} is {type(result[method][key])}, expected float"
                )

    def test_each_method_measured_against_original(self, images, tmp_path):
        """ssim/psnr/lpips must each be called exactly 4 times (once per method)."""
        from attn_texture.eval.shootout import run_shootout

        with (
            patch("attn_texture.eval.shootout.ssim", return_value=0.9) as m_ssim,
            patch("attn_texture.eval.shootout.psnr", return_value=30.0),
            patch("attn_texture.eval.shootout.lpips_score", return_value=0.1),
        ):
            run_shootout(
                "call_count",
                images["original"], images["sdedit"], images["controlnet"],
                images["pnp_baseline"], images["ours"],
                gen_prompt=_GEN_PROMPT,
                experiments_root=tmp_path,
            )

        assert m_ssim.call_count == 4


# ---------------------------------------------------------------------------
# TestExperimentDirectory
# ---------------------------------------------------------------------------


class TestExperimentDirectory:
    def test_directory_is_created(self, exp_dir):
        assert exp_dir.is_dir()

    def test_nested_exp_name_creates_nested_dir(self, images, tmp_path):
        from attn_texture.eval.shootout import run_shootout

        run_shootout(
            "nested/sub/run",
            images["original"], images["sdedit"], images["controlnet"],
            images["pnp_baseline"], images["ours"],
            gen_prompt=_GEN_PROMPT,
            experiments_root=tmp_path,
        )
        assert (tmp_path / "nested" / "sub" / "run").is_dir()
