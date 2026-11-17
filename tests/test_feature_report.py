"""Tests of the E6 status report, on fake feature folders (no model, no download)."""

import json

import pytest

from lib.feature_report import COLUMNS, E6_RUNS, MGPT_TRAIN_CAP, format_markdown, report_runs
from lib.paths import feature_dir


def _fake_run(model, treebank, split, pooling="first", *, rows=100, nan=0, max_rows=None):
    folder = feature_dir(model, "pretrained", pooling, treebank, split)
    folder.mkdir(parents=True)
    (folder / "layer_00.npy").write_bytes(b"x" * 1000)
    meta = {"n_rows": rows, "n_nan_rows": nan, "n_truncated_sentences": 0, "seconds": 12.4,
            "max_rows": max_rows, "dtype_model": "float32", "device": "cpu"}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture(autouse=True)
def features_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HLT_FEATURES_DIR", str(tmp_path))
    return tmp_path


def test_plan_covers_every_split_and_pooling_of_the_contract():
    report = report_runs()
    runs = {row["run"] for row in report}
    assert len(report) == len(runs)
    assert "bert_it isdt/train first" in runs and "bert_it pud_it/test first" in runs
    assert "geppetto isdt/dev last" in runs and "geppetto pud_it/test last" in runs
    assert "mbert pud_en/test first" in runs and "mgpt ewt/train first" in runs
    assert "bert_it pud_it/dev first" not in runs  # PUD only has a test split
    assert all(row["status"] == "missing" for row in report)
    assert len(report) == 3 + 1 + 6 + 2 + 3 + 3 + 1 + 1 + 3 + 3 + 1 + 1  # splits x poolings of E6_RUNS
    assert len(E6_RUNS) == 12


def test_done_run_reports_numbers():
    _fake_run("bert_it", "pud_it", "test", rows=23783, nan=2)
    row = next(r for r in report_runs(("bert_it",)) if r["run"] == "bert_it pud_it/test first")
    assert row["status"] == "done" and row["rows"] == 23783 and row["nan_rows"] == 2
    assert row["seconds"] == 12 and row["size_gb"] == 0.0  # about 1 KB of fake files
    assert row["model_dtype"] == "float32" and row["device"] == "cpu"


def test_models_filter_and_unknown_model():
    assert {r["run"].split()[0] for r in report_runs(("geppetto",))} == {"geppetto"}
    with pytest.raises(ValueError, match="Unknown model"):
        report_runs(("gpt2",))


def test_cap_is_only_allowed_for_mgpt_train():
    _fake_run("mgpt", "ewt", "train", rows=48000, max_rows=MGPT_TRAIN_CAP)
    _fake_run("mbert", "ewt", "train", rows=100, max_rows=50)
    status = {r["run"]: r["status"] for r in report_runs()}
    assert status["mgpt ewt/train first"] == "done"
    assert status["mbert ewt/train first"] == "check cap"


def test_markdown_table_and_summary():
    _fake_run("bert_it", "pud_it", "test", nan=3)
    text = format_markdown(report_runs(("bert_it",)))
    lines = text.splitlines()
    assert lines[0] == "| " + " | ".join(COLUMNS) + " |"
    assert lines[1].count("---") == len(COLUMNS)
    assert "| bert_it pud_it/test first | done | 100 |" in text
    assert "| bert_it isdt/train first | missing |" in text
    assert lines[-1].startswith("1 of 4 runs have a meta.json") and lines[-1].endswith("3 NaN rows in total.")
