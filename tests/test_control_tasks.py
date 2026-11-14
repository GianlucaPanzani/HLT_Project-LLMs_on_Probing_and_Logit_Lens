"""Tests for lib.control_tasks (issue E3, #5)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# lib.data (which defines UPOS_TAGS) imports torch, which CI does not install.
pytest.importorskip("torch")

from lib.control_tasks import (  # noqa: E402
    build_control_mapping,
    control_labels,
    control_next_labels,
    load_control,
)
from lib.data import UPOS_TAGS  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"


def _table(sentences: list[list[tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for s, words in enumerate(sentences):
        for i, (form, upos) in enumerate(words):
            nxt = words[i + 1][1] if i + 1 < len(words) else None
            rows.append({"sent_id": f"s{s}", "word_idx": i, "form": form, "upos": upos, "next_upos": nxt, "lang": "en"})
    return pd.DataFrame(rows)


TRAIN = _table([
    [("The", "DET"), ("cat", "NOUN"), ("sat", "VERB"), (".", "PUNCT")],
    [("the", "DET"), ("dog", "NOUN"), ("ran", "VERB"), ("fast", "ADV"), (".", "PUNCT")],
    [("A", "DET"), ("cat", "NOUN"), ("!", "PUNCT")],
])


def test_same_seed_same_mapping_other_seed_differs():
    big = _table([[(f"w{i}", "NOUN" if i % 2 else "VERB") for i in range(200)]])
    a, b, c = build_control_mapping(big, 42), build_control_mapping(big, 42), build_control_mapping(big, 7)
    assert a == b
    assert a["mapping"] != c["mapping"]


def test_labels_and_distribution():
    control = build_control_mapping(TRAIN, 42)
    assert list(control["distribution"]) == list(UPOS_TAGS)  # all 17, in order
    assert control["distribution"]["NOUN"] == pytest.approx(3 / 12)
    assert control["distribution"]["ADJ"] == 0.0
    assert set(control["mapping"].values()) <= set(UPOS_TAGS)
    assert "The" in control["mapping"] and "the" in control["mapping"]  # case-sensitive types
    assert len(control["mapping"]) == 10
    json.dumps(control)  # JSON-ready


def test_labels_follow_the_mapping_and_unseen_words_are_stable():
    control = build_control_mapping(TRAIN, 42)
    test = _table([[("The", "DET"), ("zebra", "NOUN"), ("zebra", "NOUN"), (".", "PUNCT")]])
    labels = control_labels(test, control)
    assert labels.index.equals(test.index)
    assert labels.iloc[0] == control["mapping"]["The"]
    assert labels.iloc[1] == labels.iloc[2]  # same type, same label
    assert labels.iloc[1] in UPOS_TAGS
    again = control_labels(test, json.loads(json.dumps(control)))  # also after a JSON round trip
    assert labels.equals(again)


def test_unseen_label_is_the_same_in_another_process():
    control = build_control_mapping(TRAIN, 42)
    forms = ["zebra", "Zebra", "unicorn", "perché"]
    here = control_labels(_table([[(f, "X") for f in forms]]), control).tolist()
    code = (
        "import json, sys, pandas as pd\n"
        "from lib.control_tasks import control_labels\n"
        "control = json.loads(sys.stdin.read())\n"
        f"forms = {forms!r}\n"
        "t = pd.DataFrame({'sent_id': ['s'] * len(forms), 'form': forms})\n"
        "print(json.dumps(control_labels(t, control).tolist()))\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONHASHSEED": "123"}
    result = subprocess.run(
        [sys.executable, "-c", code], input=json.dumps(control), capture_output=True, text=True, env=env, check=True
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == here


def test_type_label_frequencies_match_distribution():
    rng = np.random.default_rng(0)
    probs = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    tags = ["NOUN", "VERB", "ADJ", "DET", "PUNCT"]
    upos = rng.choice(tags, size=10_000, p=probs)
    table = _table([[(f"type{i}", str(u)) for i, u in enumerate(upos)]])  # 10k distinct types
    control = build_control_mapping(table, 42)
    labels = pd.Series(list(control["mapping"].values()))
    freq = labels.value_counts(normalize=True)
    for tag in UPOS_TAGS:
        assert abs(freq.get(tag, 0.0) - control["distribution"][tag]) < 0.03


def test_control_next_is_none_exactly_where_next_upos_is_null():
    control = build_control_mapping(TRAIN, 42)
    labels = control_labels(TRAIN, control)
    nxt = control_next_labels(TRAIN, control)
    assert (nxt.isna() == TRAIN["next_upos"].isna()).all()
    assert nxt.iloc[0] == labels.iloc[1]
    assert nxt.iloc[3] is None  # last word of sentence 0, not the first word of sentence 1


def test_unknown_upos_is_rejected():
    with pytest.raises(ValueError, match="Unknown UPOS"):
        build_control_mapping(_table([[("x", "FOO")]]), 42)


def test_load_control_missing_file(monkeypatch, tmp_path):
    import lib.control_tasks as ct

    monkeypatch.setattr(ct, "control_path", lambda tb: tmp_path / f"control_{tb}.json")
    with pytest.raises(FileNotFoundError, match="03_build_control"):
        load_control("ewt")
    (tmp_path / "control_ewt.json").write_text(json.dumps({"seed": 42}))
    assert load_control("ewt") == {"seed": 42}
