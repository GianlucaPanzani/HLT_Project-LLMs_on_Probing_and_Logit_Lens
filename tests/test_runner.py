"""Runner tests with temporary feature files and a local HF transport; no downloads."""

# ruff: noqa: E402 -- Skip before importing modules that require torch.

import json
from collections import Counter
from functools import partial
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from lib.control_tasks import build_control_mapping, control_labels
from lib.data import UPOS_TAGS
from lib.paths import feature_dir, layer_path
from lib.remote_features import RemoteBatch, iter_remote_feature_batches
from ws1_probing import runner
from ws1_probing.probes import LinearProbe, MLPProbe

N_ROWS = 50
HIDDEN_SIZE = 4
NAN_ROWS = [2, 7]
TAG_IDS = {tag: index for index, tag in enumerate(UPOS_TAGS)}
RESULT_TYPES = {
    "model": "string", "train_treebank": "string", "eval_treebank": "string",
    "lang": "string", "kind": "string", "probe": "string", "task": "string",
    "init": "string", "pooling": "string", "layer": "int64", "L": "int64",
    "tau": "float64", "acc": "float64", "n_train": "int64", "n_eval": "int64",
    "hparams": "string", "seed": "int64",
}


class LocalSource:
    """Serve HF byte-range requests from synthetic files without network access."""

    repo_id = "test/features"
    revision = "a" * 40

    def __init__(self, paths):
        self.paths = paths
        self.missing = set()
        self.read_errors = {}
        self.bytes_transferred = 0

    def size(self, path):
        if path in self.missing or path not in self.paths:
            raise FileNotFoundError(path)
        return self.paths[path].stat().st_size

    def read(self, path, start, length):
        self.size(path)
        if path in self.read_errors:
            raise self.read_errors[path]
        with self.paths[path].open("rb") as file:
            file.seek(start)
            data = file.read(length)
        self.bytes_transferred += len(data)
        return data


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """Two layers, 50 rows per split, real parquet/NPY files and a fake probe."""
    monkeypatch.setenv("HLT_FEATURES_DIR", str(tmp_path / "features"))
    monkeypatch.setenv("HLT_HF_REPO", LocalSource.repo_id)
    monkeypatch.setenv("HLT_HF_REVISION", LocalSource.revision)
    paths, tables, arrays = {}, {}, {}
    for treebank, split, offset in [
        ("ewt", "train", 0), ("ewt", "dev", 100), ("ewt", "test", 200), ("pud_en", "test", 300)
    ]:
        table = pd.DataFrame({
            "sent_id": [f"s{i // 5}" for i in range(N_ROWS)],
            "word_idx": np.arange(N_ROWS) % 5,
            "form": [f"word{i % 7}" if i < 40 else f"{treebank}_{split}_{i}" for i in range(N_ROWS)],
            "upos": ["NOUN" if i % 2 else "VERB" for i in range(N_ROWS)],
            "next_upos": [None if i % 5 == 4 else ("VERB" if i % 2 else "NOUN") for i in range(N_ROWS)],
            "lang": ["en"] * N_ROWS,
        })
        # Unique forms on NaN rows must still contribute to the control mapping.
        table.loc[2, "form"] = f"{split}_nan_only"
        table_path = tmp_path / "data" / "processed" / f"{treebank}_{split}.parquet"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(table_path, index=False, row_group_size=9)
        paths[f"data/processed/{treebank}_{split}.parquet"] = table_path
        tables[(treebank, split)] = table

        folder = feature_dir("bert", "pretrained", "first", treebank, split)
        folder.mkdir(parents=True)
        prefix = f"bert/pretrained/first/{treebank}_{split}"
        meta = dict(model="bert", init="pretrained", pooling="first", treebank=treebank,
                    split=split, num_layers=1, hidden_size=HIDDEN_SIZE, n_rows=N_ROWS,
                    dtype_saved="float16", max_rows=None)
        meta_path = folder / "meta.json"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        paths[f"{prefix}/meta.json"] = meta_path
        for layer in (0, 1):
            values = np.column_stack((np.arange(N_ROWS), np.full(N_ROWS, offset),
                                      np.arange(N_ROWS) % 2, np.full(N_ROWS, layer))).astype(np.float16)
            values[NAN_ROWS] = np.nan
            path = layer_path("bert", "pretrained", "first", treebank, split, layer)
            np.save(path, values)
            paths[f"{prefix}/layer_{layer:02d}.npy"] = path
            arrays[(treebank, split, layer)] = values

    source = LocalSource(paths)
    source_calls, stream_calls, fits, predictions = [], [], [], []

    def open_source(repo_id, revision):
        source_calls.append((repo_id, revision))
        return source

    def stream(*args, **kwargs):
        assert kwargs["source"] is source
        assert kwargs["revision"] == source.revision
        # Force several batches and parquet row-group boundaries with only 50 rows.
        kwargs["batch_size"] = 7
        stream_calls.append((args, kwargs.copy()))
        return iter_remote_feature_batches(*args, **kwargs)

    class RecordingProbe:
        def __init__(self, seed):
            self.hparams = {"seed": seed}

        def fit(self, X_train, y_train, X_dev, y_dev):
            fits.append({"X_train": X_train.copy(), "y_train": y_train.copy(),
                         "X_dev": X_dev.copy(), "y_dev": y_dev.copy(), "seed": self.hparams["seed"]})
            return self

        def predict(self, X):
            predictions.append(X.copy())
            # Deliberately simple oracle: the third feature encodes the real UPOS.
            return np.where(X[:, 2] == 1, TAG_IDS["NOUN"], TAG_IDS["VERB"])

    spec = runner.get_model_spec("bert")
    monkeypatch.setattr(runner, "get_model_spec", lambda key: {**spec, "num_layers": 1, "hidden_size": HIDDEN_SIZE})
    monkeypatch.setattr(runner, "HfRangeSource", open_source)
    monkeypatch.setattr(runner, "iter_remote_feature_batches", stream)
    monkeypatch.setattr(runner, "LinearProbe", RecordingProbe)
    monkeypatch.setattr(runner, "MLPProbe", RecordingProbe)
    output = tmp_path / "results" / "probing" / "result.parquet"
    monkeypatch.setattr(runner, "probing_result_path", lambda *args: output)
    return SimpleNamespace(source=source, tables=tables, arrays=arrays, fits=fits, predictions=predictions,
                           output=output, source_calls=source_calls, stream_calls=stream_calls)


def run(**kwargs):
    options = dict(model="bert", train_treebank="ewt", probe="linear", init="pretrained", pooling="first")
    return runner.run_probing(**{**options, **kwargs})


def scan(dataset, split="train", collect_control=True):
    return runner._scan_data("bert", "pretrained", "first", "ewt", split,
                             source=dataset.source, collect_control_enabled=collect_control)


def expected_labels(table, task, control):
    labels = table["upos"] if task == "upos" else control_labels(table, control)
    return labels.map(TAG_IDS).to_numpy(dtype=np.int64)


@pytest.mark.parametrize("probe", ["linear", "mlp"])
def test_result_schema_accuracy_and_parquet(dataset, probe):
    results = run(probe=probe, max_train_rows=12)
    assert results.columns.tolist() == list(RESULT_TYPES)
    assert {column: str(dtype) for column, dtype in results.dtypes.items()} == RESULT_TYPES
    assert len(results) == 8
    assert not results.duplicated(["eval_treebank", "task", "layer", "seed"]).any()
    assert set(results.eval_treebank) == {"ewt", "pud_en"}
    assert set(results.task) == {"upos", "control"}
    assert set(results.layer) == {0, 1}
    assert (results.L == 1).all()
    assert (results.kind == "encoder").all()
    assert (results.lang == "en").all()
    assert (results.probe == probe).all()
    assert (results.n_train == 12).all()
    assert (results.n_eval == 48).all()
    np.testing.assert_array_equal(results.tau, results.layer)
    control = build_control_mapping(dataset.tables[("ewt", "train")])
    valid = np.setdiff1d(np.arange(N_ROWS), NAN_ROWS)
    for row in results.itertuples():
        table = dataset.tables[(row.eval_treebank, "test")].iloc[valid]
        truth = expected_labels(table, row.task, control)
        predictions = expected_labels(table, "upos", control)
        assert row.acc == float(np.mean(predictions == truth))
        assert json.loads(row.hparams) == {"seed": 42}
    pd.testing.assert_frame_equal(pd.read_parquet(dataset.output), results)
    assert dataset.source_calls == [(LocalSource.repo_id, LocalSource.revision)]
    assert dataset.source.bytes_transferred > 0
    assert all(len(batch) <= 7 and batch.dtype == np.float32 for batch in dataset.predictions)


@pytest.mark.parametrize("limit", [12, 100, None])
def test_sampling_is_shared_and_labels_follow_original_rows(dataset, limit):
    run(max_train_rows=limit, seed=19)
    valid = np.setdiff1d(np.arange(N_ROWS), NAN_ROWS)
    expected_rows = valid
    if limit == 12:
        expected_rows = valid[np.random.default_rng(19).choice(len(valid), size=limit, replace=False)]
    control = build_control_mapping(dataset.tables[("ewt", "train")], seed=42)
    for index, fit in enumerate(dataset.fits):
        layer, task_index = divmod(index, 2)
        task = ("upos", "control")[task_index]
        np.testing.assert_array_equal(fit["X_train"], dataset.arrays[("ewt", "train", layer)][expected_rows])
        np.testing.assert_array_equal(fit["X_dev"], dataset.arrays[("ewt", "dev", layer)][valid])
        assert fit["X_train"].dtype == fit["X_dev"].dtype == np.float32
        assert fit["y_train"].dtype == fit["y_dev"].dtype == np.int64
        np.testing.assert_array_equal(fit["y_train"], expected_labels(
            dataset.tables[("ewt", "train")].iloc[expected_rows], task, control))
        np.testing.assert_array_equal(fit["y_dev"], expected_labels(
            dataset.tables[("ewt", "dev")].iloc[valid], task, control))
        assert fit["seed"] == 19


def test_reproducibility_and_overwrite(dataset):
    first = run(max_train_rows=12)
    second = run(max_train_rows=12)
    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(dataset.fits[0]["X_train"], dataset.fits[4]["X_train"])
    subset = run(max_train_rows=12, tasks=("upos",), layers=[1], eval_treebanks=("pud_en",))
    assert len(subset) == 1
    pd.testing.assert_frame_equal(pd.read_parquet(dataset.output), subset)


@pytest.mark.parametrize("tasks", [("upos",), ("control",), ("control", "upos")])
def test_task_subsets_and_order(dataset, tasks):
    results = run(tasks=tasks, layers=[1], eval_treebanks=("ewt",), max_train_rows=12)
    assert results.task.tolist() == list(tasks)
    assert len(dataset.fits) == len(tasks)


def test_scan_keeps_global_positions_and_all_control_statistics(dataset):
    table, counts, forms = scan(dataset)
    original = dataset.tables[("ewt", "train")]
    valid = np.setdiff1d(np.arange(N_ROWS), NAN_ROWS)
    pd.testing.assert_frame_equal(table, original.iloc[valid][["form", "upos"]])
    assert counts == Counter(original.upos)
    assert forms == set(original.form)
    assert original.loc[2, "form"] in forms
    expected = build_control_mapping(original)
    expected["treebank"] = "ewt"
    assert runner._build_control_mapping("ewt", counts, forms) == expected
    _, counts, forms = scan(dataset, split="dev", collect_control=False)
    assert not counts and not forms


@pytest.mark.parametrize("label", [None, "UNKNOWN"])
def test_invalid_train_labels_for_control(dataset, label):
    table = dataset.tables[("ewt", "train")].copy()
    table.loc[1, "upos"] = label
    table.to_parquet(dataset.source.paths["data/processed/ewt_train.parquet"], index=False)
    with pytest.raises(ValueError, match="non-null|Unknown"):
        scan(dataset)


def test_null_upos_is_dropped_without_control(dataset):
    table = dataset.tables[("ewt", "train")].copy()
    table.loc[1, "upos"] = None
    table.to_parquet(dataset.source.paths["data/processed/ewt_train.parquet"], index=False)
    valid, _, _ = scan(dataset, collect_control=False)
    assert len(valid) == 47 and 1 not in valid.index


def test_no_valid_rows(dataset):
    path = dataset.source.paths["bert/pretrained/first/ewt_train/layer_00.npy"]
    np.save(path, np.full((N_ROWS, HIDDEN_SIZE), np.nan, dtype=np.float16))
    with pytest.raises(ValueError, match="No valid rows"):
        run()


def test_empty_control_statistics():
    with pytest.raises(ValueError, match="empty"):
        runner._build_control_mapping("ewt", Counter(), set())


def batches_for_rows(row_groups, invalid=None):
    for rows in row_groups:
        values = np.repeat(np.asarray(rows, dtype=np.float16)[:, None], HIDDEN_SIZE, axis=1)
        if invalid is not None:
            values[0, 0] = invalid
        yield RemoteBatch(rows=np.array(rows), features=values, table=pd.DataFrame())


def select(batches, rows):
    return runner._select_features_iterator(batches, 1, rows, "ewt", HIDDEN_SIZE, "train")


def test_select_iterator_preserves_positions_and_skips_unselected_batches():
    consumed = []

    def batches():
        for batch in batches_for_rows([[0, 1], [3, 4], [6, 7]]):
            consumed.append(batch.rows.tolist())
            yield batch

    selected = select(batches(), [7, 1, 6])
    assert not consumed
    positions, features = next(selected)
    assert consumed == [[0, 1]]
    np.testing.assert_array_equal(positions, [1])
    np.testing.assert_array_equal(features, [[1] * HIDDEN_SIZE])
    positions, features = next(selected)
    np.testing.assert_array_equal(positions, [2, 0])
    np.testing.assert_array_equal(features[:, 0], [6, 7])
    assert features.dtype == np.float32
    with pytest.raises(StopIteration):
        next(selected)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_selected_features_must_be_finite(bad_value):
    with pytest.raises(ValueError, match="Invalid selected features"):
        list(select(batches_for_rows([[1, 2]], invalid=bad_value), [1]))


def test_selected_feature_dimension_must_match_registry():
    batch = RemoteBatch(np.array([1]), np.ones((1, 3)), pd.DataFrame())
    with pytest.raises(ValueError, match="Invalid selected features"):
        list(select([batch], [1]))


@pytest.mark.parametrize("row_groups", [[], [[0, 1]], [[0, 1], [1]]])
def test_missing_selected_rows_raise_instead_of_leaving_uninitialized_values(row_groups):
    with pytest.raises(ValueError, match="Missing selected feature rows"):
        list(select(batches_for_rows(row_groups), [0, 1, 2]))


@pytest.mark.parametrize("artifact", ["layer_01.npy", "meta.json", "table", "layer_00.npy"])
def test_missing_eval_artifacts_skip_treebank(dataset, artifact):
    path = ("data/processed/pud_en_test.parquet" if artifact == "table"
            else f"bert/pretrained/first/pud_en_test/{artifact}")
    dataset.source.missing.add(path)
    # Only layer 1 is requested: layer 0 is still required for the validity mask.
    with pytest.warns(RuntimeWarning, match="Skipping pud_en"):
        results = run(layers=[1], max_train_rows=12)
    assert set(results.eval_treebank) == {"ewt"}
    assert len(results) == 2


def test_no_eval_sets_left_does_not_write_results(dataset):
    dataset.source.missing.add("bert/pretrained/first/ewt_test/layer_01.npy")
    with pytest.warns(RuntimeWarning, match="Skipping ewt"):
        with pytest.raises(ValueError, match="No evaluation treebank"):
            run(eval_treebanks=("ewt",))
    assert not dataset.output.exists()


def test_missing_train_features_are_not_skipped(dataset):
    dataset.source.missing.add("bert/pretrained/first/ewt_train/layer_01.npy")
    with pytest.raises(FileNotFoundError):
        run()
    assert not dataset.output.exists()


def test_missing_eval_layer_during_read_skips_all_results_of_treebank(dataset):
    path = "bert/pretrained/first/pud_en_test/layer_01.npy"
    dataset.source.read_errors[path] = FileNotFoundError("missing during read")
    with pytest.warns(RuntimeWarning, match="Skipping pud_en"):
        results = run(max_train_rows=12)
    assert set(results.eval_treebank) == {"ewt"}
    assert len(results) == 4


def test_permission_error_is_not_treated_as_missing_features(dataset):
    dataset.source.read_errors["bert/pretrained/first/pud_en_test/meta.json"] = PermissionError("denied")
    with pytest.raises(PermissionError, match="denied"):
        run()


def test_capped_train_builds_control_from_the_complete_word_table(dataset):
    """Capped train features (mGPT): the control mapping is the one of the complete train split."""
    full = run()
    path = dataset.source.paths["bert/pretrained/first/ewt_train/meta.json"]
    meta = json.loads(path.read_text())
    meta["max_rows"] = 30
    meta["n_rows"] = 30
    path.write_text(json.dumps(meta), encoding="utf-8")
    for layer in (0, 1):
        array_path = dataset.source.paths[f"bert/pretrained/first/ewt_train/layer_{layer:02d}.npy"]
        np.save(array_path, np.load(array_path)[:30])
    dataset.fits.clear()
    capped = run()

    table = dataset.tables[("ewt", "train")]
    counts, types = runner._complete_train_statistics(dataset.source, "ewt")
    assert counts == Counter(table["upos"]) and types == set(table["form"])  # all 50 rows, not the first 30
    valid = int((~np.isnan(dataset.arrays[("ewt", "train", 0)][:30]).any(axis=1)).sum())
    assert set(capped["n_train"]) == {valid} and valid < 30  # the capped rows, without their NaN rows
    assert list(capped.columns) == list(full.columns) and len(capped) == len(full)
    control_rows = capped[capped["task"] == "control"]
    assert control_rows["acc"].notna().all()


def test_capped_train_without_control_needs_no_word_table(dataset):
    path = dataset.source.paths["bert/pretrained/first/ewt_train/meta.json"]
    meta = json.loads(path.read_text())
    meta["max_rows"] = 50
    path.write_text(json.dumps(meta), encoding="utf-8")
    assert len(run(tasks=("upos",))) > 0


def test_missing_revision_fails_before_opening_source(dataset, monkeypatch):
    monkeypatch.delenv("HLT_HF_REVISION")
    with pytest.raises(ValueError, match="HLT_HF_REVISION"):
        run()
    assert not dataset.source_calls


@pytest.mark.parametrize("options", [
    {"probe": "bad"}, {"init": "bad"}, {"pooling": "max"},
    {"tasks": ()}, {"tasks": ("bad",)}, {"tasks": ("upos", "upos")},
    {"layers": []}, {"layers": [-1]}, {"layers": [2]}, {"layers": [0.5]},
    {"layers": [True]}, {"layers": [0, 0]},
    {"max_train_rows": 0}, {"max_train_rows": -1}, {"max_train_rows": 1.5}, {"max_train_rows": True},
    {"seed": -1}, {"seed": 1.5}, {"seed": True},
    {"eval_treebanks": ()}, {"eval_treebanks": ("ewt", "ewt")}, {"train_treebank": "pud_en"},
])
def test_invalid_parameters_fail_before_network_access(dataset, options):
    with pytest.raises(ValueError):
        run(**options)
    assert not dataset.source_calls


@pytest.mark.parametrize("probe", ["linear", "mlp"])
def test_real_probes_on_synthetic_remote_features(dataset, monkeypatch, probe):
    monkeypatch.setattr(runner, "LinearProbe", partial(LinearProbe, Cs=(1.0,), max_iter=100))
    monkeypatch.setattr(runner, "MLPProbe", partial(
        MLPProbe, hidden=8, max_epochs=1, batch_size=16, device="cpu"))
    result = run(probe=probe, max_train_rows=24)
    assert len(result) == 8 and result.acc.between(0, 1).all()
    key = "C" if probe == "linear" else "hidden"
    assert all(key in json.loads(value) for value in result.hparams)


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return import_module("10_run_probing"), import_module("11_summarize_probing")


def test_cli_runs_and_prints_path(dataset, scripts, monkeypatch, capsys):
    cli, _ = scripts
    monkeypatch.setattr(cli, "probing_result_path", lambda *args: dataset.output)
    cli.main(["--model", "bert", "--train-treebank", "ewt", "--probe", "linear",
              "--init", "pretrained", "--pooling", "first", "--max-train-rows", "none"])
    assert (pd.read_parquet(dataset.output).n_train == 48).all()
    output = capsys.readouterr().out
    assert str(dataset.output) in output
    assert "Layer 0, ewt: upos=" in output and "Layer 1, pud_en: upos=" in output


def test_yaml_jobs_merge_defaults_and_override(scripts, tmp_path, monkeypatch):
    cli, _ = scripts
    config = tmp_path / "jobs.yaml"
    config.write_text("""defaults:
  train_treebank: ewt
  pooling: first
  tasks: [upos, control]
  max_train_rows: 12
  seed: 42
jobs:
  - {model: bert, probe: linear, init: pretrained}
  - {model: distilbert, probe: mlp, init: random, max_train_rows: null, seed: 7}
""", encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli, "run_probing", lambda **kwargs: calls.append(kwargs))
    cli.main(["--config", str(config), "--quiet"])
    assert len(calls) == 2
    assert calls[0]["tasks"] == ["upos", "control"]
    assert calls[0]["max_train_rows"] == 12 and calls[0]["seed"] == 42
    assert calls[1]["max_train_rows"] is None and calls[1]["seed"] == 7
    assert all(call["train_treebank"] == "ewt" and call["verbose"] == 0 for call in calls)


def test_ws1_config_covers_required_jobs(scripts):
    cli, _ = scripts
    jobs = cli._load_jobs_from_yaml(Path(__file__).resolve().parents[1] / "configs" / "ws1.yaml")
    assert {(job["model"], job["probe"], job["init"]) for job in jobs} == {
        (model, probe, init) for model in ("bert", "distilbert", "gpt2")
        for probe in ("linear", "mlp") for init in ("pretrained", "random")
    }
    assert len(jobs) == 12
    assert all(job["tasks"] == ["upos", "control"] and job["pooling"] == "first" for job in jobs)


@pytest.mark.parametrize("content", [
    "[]", "{}", "jobs: []", "defaults: []\njobs: [{}]", "jobs: [bad]",
    "jobs: [{}]", "defaults: {typo: true}\njobs: [{}]",
    "jobs:\n  - {model: bert, train_treebank: ewt, probe: linear, init: pretrained, pooling: first, tasks: upos}",
    "defaults: {model: bert, train_treebank: ewt, probe: linear, init: pretrained, pooling: first}\njobs: [{}, {}]",
])
def test_invalid_config_fails_before_running(scripts, tmp_path, monkeypatch, content):
    cli, _ = scripts
    config = tmp_path / "invalid.yaml"
    config.write_text(content, encoding="utf-8")
    calls = []
    monkeypatch.setattr(cli, "run_probing", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(config)])
    assert error.value.code == 2 and not calls


@pytest.mark.parametrize("args", [[], ["--max-train-rows", "0"], ["--max-train-rows", "bad"],
                                  ["--config", "configs/ws1.yaml", "--model", "bert"]])
def test_invalid_cli_arguments(scripts, args):
    with pytest.raises(SystemExit) as error:
        scripts[0].main(args)
    assert error.value.code == 2


def test_summary_reads_all_results_and_overwrites(dataset, scripts, monkeypatch, capsys):
    from lib.metrics import summarize_probing

    _, summary_cli = scripts
    root = dataset.output.parents[2]
    monkeypatch.setattr(summary_cli, "ROOT", root)
    linear = run(max_train_rows=12)
    linear.to_parquet(dataset.output.with_name("linear.parquet"), index=False)
    mlp = run(probe="mlp", max_train_rows=12)
    summary_cli.main([])
    output = root / "results" / "summary" / "cog.parquet"
    expected = summarize_probing(pd.concat([linear, mlp], ignore_index=True))
    pd.testing.assert_frame_equal(pd.read_parquet(output), expected)
    summary_cli.main([])
    pd.testing.assert_frame_equal(pd.read_parquet(output), expected)
    assert str(output) in capsys.readouterr().out


def test_summary_missing_results_fails(scripts, tmp_path, monkeypatch):
    _, cli = scripts
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="No probing"):
        cli.main([])
    assert not (tmp_path / "results" / "summary").exists()


def test_summary_missing_control_does_not_write(dataset, scripts, monkeypatch):
    _, cli = scripts
    monkeypatch.setattr(cli, "ROOT", dataset.output.parents[2])
    run(tasks=("upos",))
    with pytest.warns(UserWarning, match="missing control"):
        with pytest.raises(ValueError, match="No complete"):
            cli.main([])
    assert not (dataset.output.parents[1] / "summary" / "cog.parquet").exists()
