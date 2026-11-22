"""Tests for cross-linguistic Delta CoG helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ws3_crossling.delta_cog import (
    DELTA_COG_COLUMNS,
    PAIRS,
    ModelPair,
    delta_cog,
    selectivity_curves,
)

SUMMARY_COLUMNS = [
    "model",
    "train_treebank",
    "eval_treebank",
    "lang",
    "probe",
    "init",
    "pooling",
    "task",
    "L",
    "cog",
    "cog_norm",
    "peak_layer",
    "peak_tau",
    "max_selectivity",
]


def _summary_row(
    *,
    model: str,
    train: str,
    eval_tb: str,
    lang: str,
    cog_norm: float,
    L: int = 12,
) -> dict[str, object]:
    return {
        "model": model,
        "train_treebank": train,
        "eval_treebank": eval_tb,
        "lang": lang,
        "probe": "linear",
        "init": "pretrained",
        "pooling": "first",
        "task": "upos",
        "L": L,
        "cog": cog_norm * L,
        "cog_norm": cog_norm,
        "peak_layer": 6,
        "peak_tau": 0.5,
        "max_selectivity": 0.4,
    }


def _fake_summary() -> pd.DataFrame:
    rows = [
        _summary_row(model="mbert", train="ewt", eval_tb="ewt", lang="en", cog_norm=0.60),
        _summary_row(model="mbert", train="isdt", eval_tb="isdt", lang="it", cog_norm=0.62),
        _summary_row(model="mbert", train="ewt", eval_tb="pud_en", lang="en", cog_norm=0.59),
        _summary_row(model="mbert", train="isdt", eval_tb="pud_it", lang="it", cog_norm=0.60),
        _summary_row(model="bert", train="ewt", eval_tb="ewt", lang="en", cog_norm=0.58),
        _summary_row(model="bert_it", train="isdt", eval_tb="isdt", lang="it", cog_norm=0.59),
        _summary_row(model="bert", train="ewt", eval_tb="pud_en", lang="en", cog_norm=0.57),
        _summary_row(model="bert_it", train="isdt", eval_tb="pud_it", lang="it", cog_norm=0.58),
    ]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def test_delta_cog_fake_summary() -> None:
    """Delta CoG matches IT minus EN on own test and PUD."""
    summary = _fake_summary()
    tiny_pairs = tuple(p for p in PAIRS if p.pair in ("mbert", "bert"))
    out = delta_cog(summary, pairs=tiny_pairs)

    mbert_own = out.loc[(out["pair"] == "mbert") & (out["eval_set"] == "own_test")].iloc[0]
    assert mbert_own["delta_cog"] == pytest.approx(0.02)
    mbert_pud = out.loc[(out["pair"] == "mbert") & (out["eval_set"] == "pud")].iloc[0]
    assert mbert_pud["delta_cog"] == pytest.approx(0.01)
    bert_own = out.loc[(out["pair"] == "bert") & (out["eval_set"] == "own_test")].iloc[0]
    assert bert_own["delta_cog"] == pytest.approx(0.01)

    assert list(out.columns) == DELTA_COG_COLUMNS


def test_delta_cog_missing_row_raises() -> None:
    summary = _fake_summary().iloc[:-1]
    with pytest.raises(ValueError, match="No summary row"):
        delta_cog(summary, pairs=(PAIRS[2],))


def test_delta_cog_duplicate_row_raises() -> None:
    summary = pd.concat([_fake_summary(), _fake_summary().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        delta_cog(summary, pairs=(PAIRS[0],))


def _probing_group_rows(model: str, train: str, eval_tb: str, lang: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    acc_upos = [0.5, 0.7, 0.9, 0.8]
    acc_ctrl = [0.5, 0.5, 0.5, 0.5]
    for layer, (u, c) in enumerate(zip(acc_upos, acc_ctrl)):
        for task, acc in (("upos", u), ("control", c)):
            rows.append(
                {
                    "model": model,
                    "train_treebank": train,
                    "eval_treebank": eval_tb,
                    "lang": lang,
                    "kind": "word",
                    "probe": "linear",
                    "task": task,
                    "init": "pretrained",
                    "pooling": "first",
                    "layer": layer,
                    "L": 3,
                    "tau": layer / 3,
                    "acc": acc,
                    "n_train": 100,
                    "n_eval": 50,
                    "hparams": {},
                    "seed": 42,
                }
            )
    return rows


def test_selectivity_curves_synthetic() -> None:
    results = pd.DataFrame(_probing_group_rows("mbert", "ewt", "ewt", "en"))
    curves = selectivity_curves(results)
    assert len(curves) == 4
    np.testing.assert_allclose(curves["tau"].to_numpy(), [0, 1 / 3, 2 / 3, 1])
    np.testing.assert_allclose(curves["selectivity"].to_numpy(), [0.0, 0.2, 0.4, 0.3])


def test_plot_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from viz.crossling_plots import plot_delta_cog, plot_selectivity_pairs

    summary = _fake_summary()
    tiny = (ModelPair("mbert", "multilingual", "mbert", "ewt", "mbert", "isdt", "ewt", "isdt", "pud_en", "pud_it"),)
    delta = delta_cog(summary, pairs=tiny)
    results = pd.DataFrame(_probing_group_rows("mbert", "ewt", "ewt", "en"))
    curves = selectivity_curves(results)
    plot_delta_cog(delta, tmp_path / "delta.png")
    plot_selectivity_pairs(curves, delta, tmp_path / "sel.png")
    assert (tmp_path / "delta.png").is_file()
    assert (tmp_path / "sel.png").is_file()