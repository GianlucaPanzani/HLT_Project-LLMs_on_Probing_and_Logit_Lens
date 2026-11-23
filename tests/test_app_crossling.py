"""Tests for app Tab 4 cross-linguistic helpers (issue A4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from tabs import crossling  # noqa: E402, I001


def test_eval_treebanks_own_and_pud():
    spec = crossling._pair_spec("gpt2")
    en, it = crossling._eval_treebanks(spec, "own_test")
    assert en == "ewt" and it == "isdt"
    en, it = crossling._eval_treebanks(spec, "pud")
    assert en == "pud_en" and it == "pud_it"


def test_format_delta_table_rounds():
    delta = pd.DataFrame(
        {
            "pair": ["gpt2"],
            "kind": ["paired"],
            "eval_set": ["own_test"],
            "cog_norm_en": [0.123456],
            "cog_norm_it": [0.234567],
            "delta_cog": [0.111111],
        }
    )
    view = crossling._format_delta_table(delta)
    assert view.iloc[0]["delta_cog"] == 0.1111


def test_crossling_tab_exposes_render():
    assert callable(crossling.render)