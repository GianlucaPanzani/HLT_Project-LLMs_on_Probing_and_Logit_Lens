"""Centralised path helpers for all data, feature and result files.

Every function only builds a :class:`~pathlib.Path`; it never creates or
deletes anything on disk.  The repository root is derived from the location
of this file (``src/lib/paths.py`` → two parents up).

If the environment variable ``HLT_FEATURES_DIR`` is set, it replaces the
default ``<root>/features`` directory for all feature-related paths.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]
"""Repository root, detected from the location of this module."""


def _features_root() -> Path:
    """Return the features base directory, honouring ``HLT_FEATURES_DIR``."""
    env = os.environ.get("HLT_FEATURES_DIR")
    if env:
        return Path(env)
    return ROOT / "features"


# --------------------------------------------------------------------------- #
# Data paths
# --------------------------------------------------------------------------- #

def raw_dir(treebank: str) -> Path:
    """Directory for raw CoNLL-U files of a treebank.

    Parameters
    ----------
    treebank : str
        Treebank registry key, e.g. ``"ewt"``.

    Returns
    -------
    Path
        ``data/raw/{treebank}/``
    """
    return ROOT / "data" / "raw" / treebank


def word_table_path(treebank: str, split: str) -> Path:
    """Parquet path for a word table.

    Parameters
    ----------
    treebank : str
        Treebank registry key.
    split : str
        Split name (``"train"``, ``"dev"`` or ``"test"``).

    Returns
    -------
    Path
        ``data/processed/{treebank}_{split}.parquet``
    """
    return ROOT / "data" / "processed" / f"{treebank}_{split}.parquet"


def control_path(train_treebank: str) -> Path:
    """JSON path for control-task labels.

    Parameters
    ----------
    train_treebank : str
        The training treebank key (``"ewt"`` or ``"isdt"``).

    Returns
    -------
    Path
        ``data/processed/control_{train_treebank}.json``
    """
    return ROOT / "data" / "processed" / f"control_{train_treebank}.json"


# --------------------------------------------------------------------------- #
# Feature paths
# --------------------------------------------------------------------------- #

def feature_dir(model: str, init: str, pooling: str, treebank: str, split: str) -> Path:
    """Directory holding per-layer ``.npy`` feature files and ``meta.json``.

    Parameters
    ----------
    model : str
        Model registry key.
    init : str
        ``"pretrained"`` or ``"random"``.
    pooling : str
        ``"first"``, ``"mean"`` or ``"last"``.
    treebank : str
        Treebank registry key.
    split : str
        Split name.

    Returns
    -------
    Path
        ``features/{model}/{init}/{pooling}/{treebank}_{split}/``
    """
    return _features_root() / model / init / pooling / f"{treebank}_{split}"


def layer_path(model: str, init: str, pooling: str, treebank: str, split: str, layer: int) -> Path:
    """Path to a single layer's feature file.

    Parameters
    ----------
    model : str
        Model registry key.
    init : str
        ``"pretrained"`` or ``"random"``.
    pooling : str
        ``"first"``, ``"mean"`` or ``"last"``.
    treebank : str
        Treebank registry key.
    split : str
        Split name.
    layer : int
        Layer index (0 … L).

    Returns
    -------
    Path
        ``features/{model}/{init}/{pooling}/{treebank}_{split}/layer_{layer:02d}.npy``
    """
    return feature_dir(model, init, pooling, treebank, split) / f"layer_{layer:02d}.npy"


# --------------------------------------------------------------------------- #
# Result paths
# --------------------------------------------------------------------------- #

def probing_result_path(model: str, train_treebank: str, probe: str, init: str, pooling: str) -> Path:
    """Parquet path for probing results.

    Parameters
    ----------
    model : str
        Model registry key.
    train_treebank : str
        Training treebank key.
    probe : str
        ``"linear"`` or ``"mlp"``.
    init : str
        ``"pretrained"`` or ``"random"``.
    pooling : str
        ``"first"``, ``"mean"`` or ``"last"``.

    Returns
    -------
    Path
        ``results/probing/{model}__{train_treebank}__{probe}__{init}__{pooling}.parquet``
    """
    name = f"{model}__{train_treebank}__{probe}__{init}__{pooling}.parquet"
    return ROOT / "results" / "probing" / name


def cka_result_path(model_a: str, model_b: str, treebank: str) -> Path:
    """Parquet path for CKA results.

    Parameters
    ----------
    model_a : str
        First model registry key.
    model_b : str
        Second model registry key.
    treebank : str
        Treebank registry key.

    Returns
    -------
    Path
        ``results/cka/{model_a}__vs__{model_b}__{treebank}.parquet``
    """
    name = f"{model_a}__vs__{model_b}__{treebank}.parquet"
    return ROOT / "results" / "cka" / name


def lens_result_path(model: str, dataset: str, per_token: bool = False) -> Path:
    """Parquet path for lens results.

    Parameters
    ----------
    model : str
        Model registry key.
    dataset : str
        Dataset key (``"pud_en"``, ``"pud_it"`` or ``"wikitext"``).
    per_token : bool
        If ``True``, return the per-token results path under
        ``results/lens/tokens/``.

    Returns
    -------
    Path
        ``results/lens/{model}__{dataset}.parquet`` or
        ``results/lens/tokens/{model}__{dataset}.parquet``
    """
    name = f"{model}__{dataset}.parquet"
    if per_token:
        return ROOT / "results" / "lens" / "tokens" / name
    return ROOT / "results" / "lens" / name
