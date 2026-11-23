"""Tests for app Tab 3 CKA and compression explorer (issue A3)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from tabs import cka  # noqa: E402, I001


def test_cka_tab_exposes_render():
    """Verify that cka module exposes a callable render function."""
    assert callable(cka.render)


def test_parse_cka_filename():
    """Test extracting model_a, model_b, and treebank from CKA filename."""
    path = Path("results/cka/bert__vs__distilbert__ewt.parquet")
    m_a, m_b, tb = cka._parse_cka_filename(path)
    assert m_a == "bert"
    assert m_b == "distilbert"
    assert tb == "ewt"

    custom_path = Path("results/cka/model1__vs__model2__isdt.parquet")
    m_a, m_b, tb = cka._parse_cka_filename(custom_path)
    assert m_a == "model1"
    assert m_b == "model2"
    assert tb == "isdt"

    fallback_path = Path("some_arbitrary_name.parquet")
    m_a, m_b, tb = cka._parse_cka_filename(fallback_path)
    assert m_a == "bert"
    assert m_b == "distilbert"
    assert tb == "ewt"


def test_format_mapping_table():
    """Verify formatting of accuracy, selectivity, and difference columns."""
    df = pd.DataFrame({
        "distilbert_layer": [0, 1],
        "distilbert_acc": [0.55556, 0.66666],
        "distilbert_sel": [0.33334, 0.44444],
        "rigid_bert_layer": [0, 2],
        "rigid_bert_acc": [0.50000, 0.60000],
        "rigid_bert_sel": [0.30000, 0.40000],
        "rigid_acc_diff": [0.05556, 0.06666],
        "rigid_sel_diff": [0.03334, 0.04444],
        "cka_best_bert_layer": [0, 2],
        "cka_best_bert_acc": [0.50000, 0.60000],
        "cka_best_bert_sel": [0.30000, 0.40000],
        "cka_best_cka": [0.99999, 0.88888],
        "cka_best_acc_diff": [0.05556, 0.06666],
        "cka_best_sel_diff": [0.03334, 0.04444],
    })
    formatted = cka._format_mapping_table(df)

    assert formatted.iloc[0]["distilbert_acc"] == "0.5556"
    assert formatted.iloc[0]["distilbert_sel"] == "0.3333"
    assert formatted.iloc[0]["rigid_acc_diff"] == "+0.0556"
    assert formatted.iloc[0]["cka_best_cka"] == "1.0000"


def _make_mock_cka_df() -> pd.DataFrame:
    """Create a synthetic 13x7 CKA DataFrame."""
    records = []
    for la in range(13):
        for lb in range(7):
            records.append({
                "layer_a": la,
                "layer_b": lb,
                "cka": float(max(0.0, 1.0 - 0.1 * abs(la - 2 * lb))),
                "n_tokens": 1000,
                "seed": 42,
            })
    return pd.DataFrame(records)


def _make_mock_probing_df(model: str, num_layers: int, treebank: str = "ewt") -> pd.DataFrame:
    """Create a synthetic probing results DataFrame."""
    rows = []
    for layer in range(num_layers + 1):
        tau = layer / num_layers
        acc_upos = 0.4 + 0.4 * np.sin(np.pi * tau)
        acc_ctrl = 0.1
        for task, acc in [("upos", acc_upos), ("control", acc_ctrl)]:
            rows.append({
                "model": model,
                "train_treebank": treebank,
                "eval_treebank": treebank,
                "lang": "en",
                "kind": "encoder",
                "probe": "linear",
                "task": task,
                "init": "pretrained",
                "pooling": "first",
                "layer": layer,
                "L": num_layers,
                "tau": tau,
                "acc": acc,
                "n_train": 1000,
                "n_eval": 200,
                "hparams": '{"C": 1.0}',
                "seed": 42,
            })
    return pd.DataFrame(rows)


def test_render_missing_dir(monkeypatch):
    """Verify render gives st.info and does not crash when CKA dir is missing."""
    mock_st = MagicMock()
    monkeypatch.setattr(cka, "st", mock_st)
    monkeypatch.setattr(cka, "CKA_DIR", Path("/nonexistent/cka/dir"))

    cka.render()
    mock_st.info.assert_called_once()
    assert "No CKA results directory found" in mock_st.info.call_args[0][0]


def test_render_empty_dir(monkeypatch, tmp_path):
    """Verify render gives st.info when CKA dir exists but has no parquet files."""
    mock_st = MagicMock()
    monkeypatch.setattr(cka, "st", mock_st)
    monkeypatch.setattr(cka, "CKA_DIR", tmp_path)

    cka.render()
    mock_st.info.assert_called_once()
    assert "No CKA results found" in mock_st.info.call_args[0][0]


def test_render_with_full_data(monkeypatch, tmp_path):
    """Verify render with mock CKA and probing data displays all components."""
    # Create CKA parquet file
    cka_file = tmp_path / "bert__vs__distilbert__ewt.parquet"
    _make_mock_cka_df().to_parquet(cka_file)

    probing_dir = tmp_path / "probing"
    probing_dir.mkdir()
    bert_file = probing_dir / "bert__ewt__linear__pretrained__first.parquet"
    distil_file = probing_dir / "distilbert__ewt__linear__pretrained__first.parquet"
    _make_mock_probing_df("bert", 12).to_parquet(bert_file)
    _make_mock_probing_df("distilbert", 6).to_parquet(distil_file)

    mock_st = MagicMock()
    # Configure selectbox/checkbox behavior
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    mock_st.checkbox.return_value = True

    monkeypatch.setattr(cka, "st", mock_st)
    monkeypatch.setattr(cka, "CKA_DIR", tmp_path)
    monkeypatch.setattr(cka, "PROBING_DIR", probing_dir)

    with patch("lib.paths.probing_result_path") as mock_prp:
        mock_prp.side_effect = lambda m, *args: bert_file if m == "bert" else distil_file
        cka.render()

    # Heatmap should be rendered via plotly_chart or pyplot
    assert mock_st.plotly_chart.called or mock_st.pyplot.called
    # Dataframe table should be rendered
    assert mock_st.dataframe.called
    # Metric cards should be rendered
    assert mock_st.columns.called


def test_render_cka_present_probing_missing(monkeypatch, tmp_path):
    """Verify render when CKA is present but probing data is missing."""
    cka_file = tmp_path / "bert__vs__distilbert__ewt.parquet"
    _make_mock_cka_df().to_parquet(cka_file)

    empty_probing_dir = tmp_path / "probing"
    empty_probing_dir.mkdir()

    mock_st = MagicMock()
    mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
    mock_st.checkbox.return_value = True

    monkeypatch.setattr(cka, "st", mock_st)
    monkeypatch.setattr(cka, "CKA_DIR", tmp_path)
    monkeypatch.setattr(cka, "PROBING_DIR", empty_probing_dir)

    with patch("lib.paths.probing_result_path") as mock_prp:
        mock_prp.side_effect = lambda m, *args: empty_probing_dir / f"{m}.parquet"
        cka.render()

    # Heatmap was still rendered!
    assert mock_st.plotly_chart.called or mock_st.pyplot.called
    # And st.info was called to notify about missing probing results
    info_calls = [call[0][0] for call in mock_st.info.call_args_list]
    assert any("Probing results" in msg for msg in info_calls)
