"""Test result downloads with synthetic parquets and fake HF clients; no network."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from test_push_probing_results import write_result


@pytest.fixture
def download(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = import_module("14_pull_probing_results")
    remote_files = {}
    for model in ("bert", "gpt2"):
        path = write_result(tmp_path / "remote", model)
        remote_files[f"results/probing/{path.name}"] = path.read_bytes()
    remote_files["results/summary/cog.parquet"] = b"ignored"
    remote_files["results/probing/nested/file.parquet"] = b"ignored"
    remote_files["results/probing/README.md"] = b"ignored"
    remote_files["features/layer_00.npy"] = b"ignored"
    calls, reads = [], []
    commit = "b" * 40

    class FakeApi:
        def list_repo_files(self, **kwargs):
            calls.append(kwargs)
            return list(reversed(remote_files))

    class FakeSource:
        def __init__(self, repo_id, revision, *, api):
            assert repo_id == "hlt-group11/features" and revision == "main"
            assert isinstance(api, FakeApi)
            self.revision = commit

        def size(self, path):
            return len(remote_files[path])

        def read(self, path, start, length):
            reads.append((path, start, length))
            return remote_files[path][start:start + length]

    monkeypatch.setattr(module, "HfApi", FakeApi)
    monkeypatch.setattr(module, "HfRangeSource", FakeSource)
    monkeypatch.setattr(module, "ROOT", tmp_path / "local")
    return SimpleNamespace(module=module, files=remote_files, calls=calls, reads=reads,
                           root=tmp_path / "local", commit=commit, source=FakeSource)


@pytest.mark.parametrize("custom_dir", [False, True])
def test_downloads_only_probing_results_at_one_commit(download, custom_dir, capsys):
    root = download.root / ("alternative" if custom_dir else "results")
    args = ["--results-dir", str(root)] if custom_dir else []
    download.module.main(args)

    assert download.calls == [{"repo_id": "hlt-group11/features", "repo_type": "dataset",
                               "revision": download.commit}]
    expected = sorted(path for path in download.files if path.endswith("first.parquet") or path.endswith("last.parquet"))
    assert [path for path, _, _ in download.reads] == expected
    assert sorted(path.name for path in (root / "probing").iterdir()) == [Path(path).name for path in expected]
    for path in expected:
        local = root / "probing" / Path(path).name
        assert local.read_bytes() == download.files[path]
        assert not pd.read_parquet(local).empty
    output = capsys.readouterr().out
    assert download.commit in output and str(root / "probing") in output
    assert not list(root.glob(".pull-probing-*"))


def test_replaces_matching_files_and_keeps_other_local_outputs(download):
    root = download.root / "results"
    existing = write_result(root, task="control")
    extra = write_result(root, model="distilbert")
    extra_bytes = extra.read_bytes()
    summary = root / "summary" / "cog.parquet"
    summary.parent.mkdir()
    summary.write_bytes(b"local summary")
    download.module.main([])
    assert pd.read_parquet(existing).task.tolist() == ["upos"]
    assert extra.read_bytes() == extra_bytes
    assert summary.read_bytes() == b"local summary"


@pytest.mark.parametrize("problem", ["corrupt", "filename", "read_error", "short_read"])
def test_failed_download_preserves_existing_results(download, monkeypatch, problem):
    root = download.root / "results"
    existing = write_result(root, task="control")
    original = existing.read_bytes()
    remote = "results/probing/gpt2__ewt__linear__pretrained__last.parquet"
    if problem == "corrupt":
        download.files[remote] = b"invalid parquet"
    elif problem == "filename":
        download.files["results/probing/wrong.parquet"] = download.files.pop(remote)
    else:
        read = download.source.read

        def failing_read(self, path, start, length):
            if path == remote:
                if problem == "read_error":
                    raise ConnectionError("synthetic network failure")
                return b"short"
            return read(self, path, start, length)

        monkeypatch.setattr(download.source, "read", failing_read)
    with pytest.raises((ValueError, ConnectionError)):
        download.module.main([])
    assert existing.read_bytes() == original
    assert not (root / "probing" / Path(remote).name).exists()
    assert not list(root.glob(".pull-probing-*"))


def test_no_remote_results_leaves_destination_untouched(download):
    download.files.clear()
    with pytest.raises(FileNotFoundError, match="No probing"):
        download.module.main([])
    assert not download.root.exists()


def test_destination_must_be_a_directory(download):
    download.root.mkdir()
    (download.root / "results").write_text("not a directory")
    with pytest.raises(ValueError, match="not a directory"):
        download.module.main([])
    assert not download.calls


def test_missing_hf_dependency_has_clear_error(download, monkeypatch):
    monkeypatch.setattr(download.module, "HfApi", None)
    with pytest.raises(ImportError, match="huggingface_hub"):
        download.module.main([])
    assert not download.root.exists()


@pytest.mark.parametrize("option", ["--repo-id", "--revision", "--token", "--remote-path"])
def test_cli_only_accepts_results_dir(download, option):
    with pytest.raises(SystemExit) as error:
        download.module.main([option, "value"])
    assert error.value.code == 2 and not download.calls
