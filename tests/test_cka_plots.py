"""Tests for viz.cka_plots – CKA heatmap and selectivity plots.

Uses synthetic DataFrames and temporary output directories.
"""


import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from viz.cka_plots import (  # noqa: E402
    plot_cka_heatmap,
    plot_selectivity_comparison,
    plot_selectivity_comparison_plotly,
)


@pytest.fixture()
def cka_df():
    """Synthetic CKA long-format DataFrame (13 x 7)."""
    records = []
    for la in range(13):
        for lb in range(7):
            records.append({
                "layer_a": la,
                "layer_b": lb,
                "cka": float(max(0.0, 1.0 - 0.1 * abs(la - 2 * lb))),
                "n_tokens": 5000,
                "seed": 42,
            })
    return pd.DataFrame(records)


class TestPlotCkaHeatmap:
    """Test CKA heatmap generation (matplotlib + plotly)."""

    def test_figures_returned(self, cka_df):
        fig_mpl, fig_plotly = plot_cka_heatmap(cka_df, mark_diagonal=True, mark_best=True)
        assert fig_mpl is not None
        assert fig_plotly is not None
        # Verify shapes exist in Plotly figure
        assert len(fig_plotly.layout.shapes) > 0

    def test_save_files(self, cka_df, tmp_path):
        prefix = tmp_path / "test_cka"
        _, fig_plotly = plot_cka_heatmap(cka_df, save_prefix=prefix)
        assert (tmp_path / "test_cka_cka_heatmap.png").is_file()
        if fig_plotly is not None:
            assert (tmp_path / "test_cka_cka_heatmap.html").is_file()


class TestPlotSelectivityComparison:
    """Test selectivity curve comparison plot."""

    def test_figure_returned(self):
        bert_tau = np.linspace(0, 1, 13)
        distil_tau = np.linspace(0, 1, 7)
        bert_sel = np.sin(np.pi * bert_tau) * 0.5
        distil_sel = np.sin(np.pi * distil_tau) * 0.45

        fig = plot_selectivity_comparison(
            bert_sel, bert_tau, distil_sel, distil_tau,
            bert_cog_norm=0.52, distil_cog_norm=0.48,
        )
        assert fig is not None

    def test_save_file(self, tmp_path):
        bert_tau = np.linspace(0, 1, 13)
        distil_tau = np.linspace(0, 1, 7)
        bert_sel = np.sin(np.pi * bert_tau) * 0.5
        distil_sel = np.sin(np.pi * distil_tau) * 0.45

        out_file = tmp_path / "sel_plot.png"
        plot_selectivity_comparison(
            bert_sel, bert_tau, distil_sel, distil_tau,
            save_path=out_file,
        )
        assert out_file.is_file()

    def test_plotly_comparison(self):
        bert_tau = np.linspace(0, 1, 13)
        distil_tau = np.linspace(0, 1, 7)
        bert_sel = np.sin(np.pi * bert_tau) * 0.5
        distil_sel = np.sin(np.pi * distil_tau) * 0.45

        fig_plotly = plot_selectivity_comparison_plotly(
            bert_sel, bert_tau, distil_sel, distil_tau,
            bert_cog_norm=0.52, distil_cog_norm=0.48,
        )
        assert fig_plotly is not None
        assert len(fig_plotly.data) == 2
