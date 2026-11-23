"""Test result uploads with synthetic parquets and a fake HF client; no network."""

import io
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def upload(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = import_module("13_push_probing_results")
    calls = []

    class FakeApi:
        def create_commit(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(oid="a" * 40, commit_url="https://huggingface.co/test/commit/" + "a" * 40)

    monkeypatch.setattr(module, "HfApi", FakeApi)
    monkeypatch.setattr(module, "CommitOperationAdd", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return SimpleNamespace(module=module, calls=calls, root=tmp_path)


def write_result(root, model="bert", task="upos"):
    frame = pd.DataFrame([{
        "model": model, "train_treebank": "ewt", "eval_treebank": "ewt", "lang": "en",
        "kind": "decoder" if model == "gpt2" else "encoder", "probe": "linear", "task": task,
        "init": "pretrained", "pooling": "last" if model == "gpt2" else "first",
        "layer": 0, "L": 12, "tau": 0.0, "acc": 0.5, "n_train": 12, "n_eval": 8,
        "hparams": '{"C": 1.0}', "seed": 42,
    }])
    pooling = frame.pooling.iloc[0]
    path = root / "probing" / f"{model}__ewt__linear__pretrained__{pooling}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


@pytest.mark.parametrize("custom_dir", [False, True])
def test_uploads_only_runner_parquets_in_one_commit(upload, custom_dir, capsys):
    root = upload.root / ("alternative" if custom_dir else "results")
    paths = [write_result(root), write_result(root, "gpt2", "upos_next")]
    for folder in ("summary", "lens", "figures"):
        extra = root / folder / "ignored.parquet"
        extra.parent.mkdir(parents=True)
        extra.write_bytes(b"not a runner result")

    upload.module.main(["--results-dir", str(root)] if custom_dir else [])

    assert len(upload.calls) == 1
    call = upload.calls[0]
    assert call["repo_id"] == "hlt-group11/features"
    assert call["repo_type"] == "dataset" and call["revision"] == "main"
    operations = call["operations"]
    assert [op.path_in_repo for op in operations] == [f"results/probing/{path.name}" for path in paths]
    assert [op.path_or_fileobj for op in operations] == [path.read_bytes() for path in paths]
    output = capsys.readouterr().out
    assert all(op.path_in_repo in output for op in operations)
    assert "a" * 40 in output and "https://huggingface.co/test/commit/" in output


def test_snapshot_bytes_do_not_change_when_local_file_changes(upload):
    root = upload.root / "results"
    path = write_result(root)
    original = path.read_bytes()
    snapshot = upload.module._read_results(root)
    write_result(root, task="control")
    assert snapshot[0][1] == original
    assert pd.read_parquet(io.BytesIO(snapshot[0][1])).task.tolist() == ["upos"]
    assert pd.read_parquet(path).task.tolist() == ["control"]


@pytest.mark.parametrize("exists", [False, True])
def test_missing_results_fail_before_contacting_hf(upload, exists):
    root = upload.root / "results"
    if exists:
        root.mkdir()
    with pytest.raises(FileNotFoundError, match="directory|No probing"):
        upload.module.main([])
    assert not upload.calls


@pytest.mark.parametrize("problem", ["corrupt", "empty", "columns", "multiple_jobs", "filename"])
def test_invalid_result_prevents_the_entire_upload(upload, problem):
    root = upload.root / "results"
    write_result(root)  # A valid file must not be uploaded before all inputs are checked.
    path = write_result(root, "gpt2")
    frame = pd.read_parquet(path)
    if problem == "corrupt":
        path.write_bytes(b"incomplete parquet")
    elif problem == "filename":
        path.rename(path.with_name("wrong.parquet"))
    else:
        if problem == "empty":
            frame = frame.iloc[:0]
        elif problem == "columns":
            frame = frame.drop(columns="acc")
        else:
            frame = pd.concat([frame, frame.assign(model="bert")], ignore_index=True)
        frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        upload.module.main([])
    assert not upload.calls


def test_missing_hf_dependency_has_clear_error(upload, monkeypatch):
    write_result(upload.root / "results")
    monkeypatch.setattr(upload.module, "HfApi", None)
    with pytest.raises(ImportError, match="huggingface_hub"):
        upload.module.main([])
    assert not upload.calls


@pytest.mark.parametrize("option", ["--repo-id", "--revision", "--token", "--remote-path"])
def test_cli_rejects_options_other_than_results_dir(upload, option):
    with pytest.raises(SystemExit) as error:
        upload.module.main([option, "value"])
    assert error.value.code == 2 and not upload.calls
