"""Helpers for Page 03: CKA and compression explorer (issue A3/A5, owner Filippo).

Provides cached parquet loading, filename parsing, and table formatting helpers.
UI composition lives in app/pages/03_Compression.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.paths import cka_result_path, probing_result_path

CKA_DIR: Path = cka_result_path("bert", "distilbert", "ewt").parent
PROBING_DIR: Path = probing_result_path("bert", "ewt", "linear", "pretrained", "first").parent


@st.cache_data(show_spinner=False, max_entries=64)
def _read_parquet_cached(path_str: str, modified_ns: int, size: int) -> pd.DataFrame:
    """Read parquet file cached on file modification timestamp and size.

    Parameters
    ----------
    path_str : str
        String path to the parquet file.
    modified_ns : int
        File modification time in nanoseconds.
    size : int
        File size in bytes.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    return pd.read_parquet(path_str)


def _load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file with stat-based caching.

    Parameters
    ----------
    path : Path
        Path to the parquet file.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    stat = path.stat()
    return _read_parquet_cached(str(path), stat.st_mtime_ns, stat.st_size)


def _parse_cka_filename(path: Path) -> tuple[str, str, str]:
    """Parse model_a, model_b, and treebank from CKA filename.

    CONTRACTS.md §8: ``results/cka/{model_a}__vs__{model_b}__{treebank}.parquet``.

    Parameters
    ----------
    path : Path
        Parquet file path.

    Returns
    -------
    tuple[str, str, str]
        (model_a, model_b, treebank)
    """
    stem = path.stem
    match = re.match(r"^([a-zA-Z0-9_-]+)__vs__([a-zA-Z0-9_-]+)__([a-zA-Z0-9_-]+)$", stem)
    if match:
        return match.group(1), match.group(2), match.group(3)
    return "bert", "distilbert", "ewt"


def _format_mapping_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric columns of the layer mapping table for clean display.

    Parameters
    ----------
    df : pd.DataFrame
        Raw mapping table from ``build_layer_mapping_table``.

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame with rounded float values.
    """
    formatted = df.copy()
    float_cols = [
        "distilbert_acc", "distilbert_sel",
        "rigid_bert_acc", "rigid_bert_sel", "rigid_acc_diff", "rigid_sel_diff",
        "cka_best_bert_acc", "cka_best_bert_sel", "cka_best_cka",
        "cka_best_acc_diff", "cka_best_sel_diff",
    ]
    for col in float_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda v: f"{v:+.4f}" if "diff" in col else f"{v:.4f}"
            )
    return formatted
