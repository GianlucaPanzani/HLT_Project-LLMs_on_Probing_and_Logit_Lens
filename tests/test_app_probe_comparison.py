"""Synthetic contract checks for the P8 page; no models or downloads."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from tabs import probe_comparison as comparison  # noqa: E402, I001


@pytest.fixture
def results():
    rows = []
    for model, kind in (("bert", "encoder"), ("gpt2", "decoder")):
        for probe in ("linear", "mlp"):
            for task in ("upos", "control"):
                for layer in range(13):
                    acc = 0.6 + layer * 0.01 if task == "upos" else 0.4
                    if probe == "mlp":
                        acc += 0.02 if task == "upos" else 0.10
                    rows.append({
                        "model": model, "kind": kind, "lang": "en", "probe": probe,
                        "train_treebank": "ewt", "eval_treebank": "ewt", "task": task,
                        "init": "pretrained", "pooling": "first", "seed": 42,
                        "layer": layer, "L": 12, "tau": layer / 12, "acc": acc,
                        "n_train": 100, "n_eval": 20, "hparams": "{}",
                    })
    return pd.DataFrame(rows)


def test_shuffled_pairs_deltas_and_summary(results):
    table = comparison.prepare_comparison(results)
    shuffled = comparison.prepare_comparison(results.sample(frac=1, random_state=7))
    pd.testing.assert_frame_equal(table, shuffled)
    assert len(table) == 26
    np.testing.assert_allclose(table["delta_acc_pp"], 2)
    np.testing.assert_allclose(table["delta_control_acc_pp"], 10)
    np.testing.assert_allclose(table["delta_selectivity_pp"], -8)
    summary = comparison.summarize_comparison(table)
    assert len(summary) == 4
    assert summary["selectivity_peak_layer"].eq(12).all()
    weights = 0.2 + np.arange(1, 13) * 0.01
    expected = np.dot(np.arange(1, 13), weights) / weights.sum() / 12
    assert summary.iloc[0]["cog_norm"] == pytest.approx(expected)


@pytest.mark.parametrize("column", ["n_train", "n_eval"])
@pytest.mark.parametrize("change_whole_probe", [False, True])
def test_incompatible_counts(results, column, change_whole_probe):
    # Both task/control mismatch and internally consistent but unequal probes.
    mask = results["probe"].eq("mlp") if change_whole_probe else results.index == 0
    results.loc[mask, column] += 1
    with pytest.raises(ValueError, match="Incompatible n_train/n_eval"):
        comparison.prepare_comparison(results)


@pytest.mark.parametrize("missing", ["probe", "controls", "layer", "seed", "duplicate"])
def test_incomplete_comparisons(results, missing):
    if missing == "probe":
        results = results.loc[results["probe"].ne("mlp")]
    elif missing == "controls":
        results = results.loc[results["task"].ne("control")]
    elif missing == "layer":
        results = results.iloc[1:]
    elif missing == "seed":
        results.loc[results["probe"].eq("mlp"), "seed"] = 43
    else:
        results = pd.concat([results, results.iloc[:1]])
    with pytest.raises(ValueError, match="Missing|Duplicate"):
        comparison.prepare_comparison(results)


@pytest.mark.parametrize("column,value", [("pooling", "last"), ("train_treebank", "isdt"), ("tau", 0.5)])
def test_incompatible_metadata(results, column, value):
    results.loc[0, column] = value
    with pytest.raises(ValueError, match="Incompatible|Inconsistent"):
        comparison.prepare_comparison(results)


def test_negative_selectivity_and_undefined_cog(results):
    results.loc[results["task"].eq("control"), "acc"] = 0.95
    table = comparison.prepare_comparison(results)
    assert table["linear_selectivity"].lt(0).all()
    assert comparison.summarize_comparison(table)["cog_norm"].isna().all()


def test_page_smoke_and_missing_file(results, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(comparison, "load_results", lambda: results)
    page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app/pages/05_Linear_vs_MLP.py"))
    page.run()
    assert not page.exception
    assert len(page.dataframe) == 2
    assert len(page.get("plotly_chart")) == 1
    page.radio[0].set_value("tau").run()
    assert not page.exception

    def missing_file():
        raise FileNotFoundError("Missing bert/mlp results")

    monkeypatch.setattr(comparison, "load_results", missing_file)
    page.run()
    assert not page.exception
    assert "Missing bert/mlp" in page.error[0].value
    assert not page.get("plotly_chart")
