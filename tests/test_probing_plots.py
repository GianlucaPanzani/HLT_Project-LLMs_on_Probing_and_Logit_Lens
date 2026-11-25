"""Minimal layout checks for the P8 comparison figure."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from viz.probing_plots import plot_probe_comparison  # noqa: E402


@pytest.mark.parametrize("x", ["layer", "tau"])
def test_comparison_panels_and_cog_labels(x):
    rows, summaries = [], []
    for model in ("bert", "gpt2"):
        for layer in range(13):
            row = {"model": model, "layer": layer, "L": 12, "tau": layer / 12,
                   "eval_treebank": "ewt", "seed": 42}
            for probe in ("linear", "mlp"):
                row.update({f"{probe}_acc": 0.8, f"{probe}_control_acc": 0.9, f"{probe}_selectivity": -0.1})
            rows.append(row)
        for probe in ("linear", "mlp"):
            summaries.append({"model": model, "probe": probe, "cog": 6.0, "cog_norm": 0.5})
    table, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    figure = plot_probe_comparison(table, summary, x=x)
    assert len(figure.data) == 12
    np.testing.assert_allclose(figure.data[0].x, np.arange(13) / (12 if x == "tau" else 1))
    assert min(figure.data[2].y) < 0
    labels = [a for a in figure.layout.annotations if "cog" in a.text]
    assert len(labels) == 4
    assert len({(a.yref, a.y) for a in labels}) == 4
    summary[["cog", "cog_norm"]] = np.nan
    undefined = plot_probe_comparison(table, summary, x=x)
    assert not any("cog" in a.text for a in undefined.layout.annotations)
