"""Unit tests for viz.lens_plots (Issue L8)."""

from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from viz.lens_plots import (  # noqa: E402
    build_commitment_table,
    compute_commitment_layer,
    load_overlay_data,
    plot_crossling_comparison,
    plot_model_overlay,
)


def test_compute_commitment_layer_monotonic():
    """Verify commitment layer for a strictly increasing curve."""
    curve = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0])
    c_lay, c_tau, max_v = compute_commitment_layer(curve, threshold_fraction=0.90)

    # Max is 1.0, 90% threshold is 0.90. First layer >= 0.90 is index 5 (0.95).
    assert c_lay == 5
    assert abs(c_tau - 5 / 6) < 1e-5
    assert max_v == 1.0


def test_compute_commitment_layer_peaking_early():
    """Verify commitment layer for a curve that peaks early then plateaus."""
    curve = np.array([0.1, 0.5, 0.92, 1.0, 0.98, 0.96])
    c_lay, c_tau, max_v = compute_commitment_layer(curve, threshold_fraction=0.90)

    # Max is 1.0. First layer >= 0.90 is index 2 (0.92).
    assert c_lay == 2
    assert abs(c_tau - 2 / 5) < 1e-5
    assert max_v == 1.0


def test_compute_commitment_layer_zeros():
    """Verify commitment layer handles non-positive curves gracefully."""
    curve = np.zeros(5)
    c_lay, c_tau, max_v = compute_commitment_layer(curve)
    assert c_lay == 0
    assert c_tau == 0.0
    assert max_v == 0.0


def test_compute_commitment_layer_empty_raises():
    """Verify empty input raises ValueError."""
    with pytest.raises(ValueError, match="Curve must be a non-empty 1D array"):
        compute_commitment_layer([])


def test_load_overlay_data_gpt2():
    """Verify load_overlay_data loads real existing outputs with expected structure."""
    data = load_overlay_data("gpt2", "pud_en", "ewt")
    assert data["model"] == "gpt2"
    assert len(data["layers"]) == 13
    assert len(data["taus"]) == 13
    assert len(data["probing_sel_next"]) == 13
    for lens in ("logit", "tuned", "jacobian"):
        assert lens in data["lenses"]
        assert len(data["lenses"][lens]["top1_acc"]) == 13
        assert len(data["lenses"][lens]["upos_top1_acc"]) == 13
        assert len(data["lenses"][lens]["entropy_mean"]) == 13
        assert len(data["lenses"][lens]["kl_mean"]) == 13


def test_build_commitment_table_schema():
    """Verify commitment table has exact expected columns and non-null values."""
    df = build_commitment_table(models=("gpt2", "geppetto"))
    expected_cols = [
        "model",
        "paradigm",
        "metric",
        "commitment_layer",
        "commitment_tau",
        "max_value",
    ]
    assert list(df.columns) == expected_cols
    assert len(df) == 2 * (1 + 3 * 2)  # 2 models x (1 probing + 3 lenses x 2 metrics) = 14 rows
    assert df["commitment_layer"].between(0, 12).all()
    assert df["commitment_tau"].between(0.0, 1.0).all()
    assert (df["max_value"] > 0).all()


def test_plot_model_overlay_runs(tmp_path: Path):
    """Verify plot_model_overlay runs and generates a valid image file."""
    save_file = tmp_path / "test_overlay.png"
    fig = plot_model_overlay("gpt2", "pud_en", "ewt", save_path=save_file)

    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    assert save_file.stat().st_size > 1000
    plt.close(fig)


def test_plot_crossling_comparison_runs(tmp_path: Path):
    """Verify plot_crossling_comparison runs and generates a valid image file."""
    save_file = tmp_path / "test_crossling.png"
    fig = plot_crossling_comparison(save_path=save_file)

    assert isinstance(fig, plt.Figure)
    assert save_file.exists()
    assert save_file.stat().st_size > 1000
    plt.close(fig)
