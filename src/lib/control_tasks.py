"""Control task labels (Hewitt and Liang, 2019), CONTRACTS.md section 5.

Each word type of a training treebank gets one random but fixed UPOS label,
drawn from the empirical UPOS distribution of that treebank. A probe can only
solve this task by memorising word types, so selectivity (real accuracy minus
control accuracy) measures how much of a probe's accuracy is memorisation.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from lib.data import UPOS_TAGS
from lib.paths import control_path


def _probabilities(control: dict) -> np.ndarray:
    '''
    Return the label distribution of a control mapping as an array.

    Parameters
    ----------
    - control (dict): Control object of CONTRACTS.md section 5.

    Returns
    -------
    np.ndarray: Probabilities in UPOS_TAGS order, renormalised to sum to 1.
    '''
    probs = np.array([control["distribution"][tag] for tag in UPOS_TAGS], dtype=np.float64)
    return probs / probs.sum()


def _unseen_label(form: str, seed: int, probs: np.ndarray) -> str:
    '''
    Label of a word type that is not in the mapping (dev, test or PUD only).

    Parameters
    ----------
    - form (str): Word form (case-sensitive).
    - seed (int): Seed of the control mapping.
    - probs (np.ndarray): Label distribution in UPOS_TAGS order.

    Returns
    -------
    str: One label drawn from a generator seeded with the md5 of
    "{seed}:{form}", so it is the same in every process (Python's hash() is
    randomised between processes and must never be used here).
    '''
    digest = hashlib.md5(f"{seed}:{form}".encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:8], 16))
    return UPOS_TAGS[int(rng.choice(len(UPOS_TAGS), p=probs))]


def build_control_mapping(train_table: pd.DataFrame, seed: int = 42) -> dict:
    '''
    Build the control mapping of a training treebank.

    Parameters
    ----------
    - train_table (pd.DataFrame): Word table of the train split
    (CONTRACTS.md section 2); uses the columns "form" and "upos".
    - seed (int): Seed of numpy.random.default_rng. Defaults to 42.

    Returns
    -------
    dict: JSON-ready object {"treebank", "seed", "distribution", "mapping"}.
    "distribution" holds the UPOS frequencies over word tokens for all 17
    tags in UPOS_TAGS order (zeros allowed). "mapping" gives every word type
    (case-sensitive form) one label; types are visited in sorted order and
    labels are drawn with a single generator in one vectorised call.
    "treebank" is None here; the caller (scripts/03_build_control.py) sets it.
    '''
    if train_table.empty:
        raise ValueError("train_table is empty.")
    unknown = set(train_table["upos"]) - set(UPOS_TAGS)
    if unknown:
        raise ValueError(f"Unknown UPOS tags in train_table: {sorted(unknown)}.")

    counts = train_table["upos"].value_counts()
    total = int(counts.sum())
    distribution = {tag: float(counts.get(tag, 0)) / total for tag in UPOS_TAGS}

    types = sorted(set(train_table["form"]))
    probs = np.array([distribution[tag] for tag in UPOS_TAGS], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(UPOS_TAGS), size=len(types), p=probs / probs.sum())
    mapping = {form: UPOS_TAGS[int(label)] for form, label in zip(types, draws)}

    return {"treebank": None, "seed": seed, "distribution": distribution, "mapping": mapping}


def control_labels(table: pd.DataFrame, control: dict) -> pd.Series:
    '''
    Control label of every row of a word table.

    Parameters
    ----------
    - table (pd.DataFrame): Word table (CONTRACTS.md section 2) of any split
    or treebank of the same language.
    - control (dict): Control object built from the matching train treebank.

    Returns
    -------
    pd.Series: Labels (str, one of UPOS_TAGS) aligned with the rows of table.
    Forms not in the mapping get the md5-seeded label of CONTRACTS.md section 5.
    '''
    mapping = control["mapping"]
    seed = int(control["seed"])
    probs = _probabilities(control)
    unseen = {form: _unseen_label(form, seed, probs) for form in set(table["form"]) if form not in mapping}
    labels = [mapping.get(form) or unseen[form] for form in table["form"]]
    return pd.Series(labels, index=table.index, name="control", dtype=object)


def control_next_labels(table: pd.DataFrame, control: dict) -> pd.Series:
    '''
    Control label of the next word in the same sentence (task control_next).

    Parameters
    ----------
    - table (pd.DataFrame): Word table (CONTRACTS.md section 2). Rows of a
    sentence must be contiguous and in word order, as in the contract.
    - control (dict): Control object built from the matching train treebank.

    Returns
    -------
    pd.Series: Labels aligned with the rows of table; None for the last word
    of each sentence (the same rows where next_upos is null).
    '''
    labels = control_labels(table, control).tolist()
    sent_ids = table["sent_id"].tolist()
    nxt = [
        labels[i + 1] if i + 1 < len(labels) and sent_ids[i + 1] == sent_ids[i] else None
        for i in range(len(labels))
    ]
    return pd.Series(nxt, index=table.index, name="control_next", dtype=object)


def load_control(train_treebank: str) -> dict:
    '''
    Load a control mapping written by scripts/03_build_control.py.

    Parameters
    ----------
    - train_treebank (str): Training treebank key, "ewt" or "isdt".

    Returns
    -------
    dict: Control object of CONTRACTS.md section 5.
    '''
    path = control_path(train_treebank)
    if not path.is_file():
        raise FileNotFoundError(
            f"Control mapping not found: {path}. Run: PYTHONPATH=src python scripts/03_build_control.py"
        )
    with path.open(encoding="utf-8") as file:
        return json.load(file)
