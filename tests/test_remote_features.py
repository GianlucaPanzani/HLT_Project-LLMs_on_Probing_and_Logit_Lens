"""Tests of lib.remote_features on synthetic data and an in-memory transport (no network, no credentials)."""

import io
import json
import os

import numpy as np
import pandas as pd
import pytest
from numpy.lib.format import open_memmap

from lib.remote_features import (
    HEADER_PROBE_BYTES,
    HfRangeSource,
    TransientReadError,
    iter_remote_feature_batches,
    parse_npy_header,
)

N_ROWS, D, N_LAYERS = 23, 4, 2
FOLDER = "fake/pretrained/first/ewt_train"
LAYER_FILE = f"{FOLDER}/layer_01.npy"
TABLE_FILE = "data/processed/ewt_train.parquet"
ROW_BYTES = D * 2


class MemorySource:
    """In-memory transport that records every read and can fail on demand."""

    def __init__(self, files: dict[str, bytes], failures: dict[int, Exception] | None = None):
        self.files = files
        self.revision = "a" * 40
        self.bytes_transferred = 0
        self.reads: list[tuple[str, int, int]] = []
        self.failures = failures or {}
        self.n_calls = 0

    def _maybe_fail(self):
        self.n_calls += 1
        if self.n_calls in self.failures:
            raise self.failures[self.n_calls]

    def size(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return len(self.files[path])

    def read(self, path, start, length):
        self._maybe_fail()
        if path not in self.files:
            raise FileNotFoundError(path)
        data = self.files[path][start: start + length]
        self.reads.append((path, start, length))
        self.bytes_transferred += len(data)
        return data


def _word_table() -> pd.DataFrame:
    sentences = [3, 5, 1, 4, 6, 4]  # 23 rows
    rows = []
    for number, length in enumerate(sentences):
        for index in range(length):
            rows.append({
                "sent_id": f"s{number}", "word_idx": index, "form": f"w{number}_{index}",
                "upos": "NOUN" if index % 2 else "VERB",
                "next_upos": ("NOUN" if index % 2 else "VERB") if index + 1 < length else None,
                "lang": "en",
            })
    frame = pd.DataFrame(rows)
    frame["next_upos"] = frame["next_upos"].astype("string")
    return frame


@pytest.fixture
def local(tmp_path):
    """Synthetic local features and word table, plus the files a remote repository would hold."""
    table = _word_table()
    assert len(table) == N_ROWS
    parquet = tmp_path / "table.parquet"
    table.to_parquet(parquet, row_group_size=7)  # several row groups, not aligned with the batches
    rng = np.random.default_rng(0)
    layers = []
    files = {TABLE_FILE: parquet.read_bytes()}
    for layer in range(N_LAYERS + 1):
        array = rng.normal(size=(N_ROWS, D)).astype(np.float16)
        array[[2, 9, 10]] = np.nan
        path = tmp_path / f"layer_{layer:02d}.npy"
        memmap = open_memmap(path, mode="w+", dtype=np.float16, shape=array.shape)
        memmap[:] = array
        memmap.flush()
        del memmap
        layers.append(array)
        files[f"{FOLDER}/layer_{layer:02d}.npy"] = path.read_bytes()
    meta = {"model": "fake", "init": "pretrained", "pooling": "first", "treebank": "ewt", "split": "train",
            "num_layers": N_LAYERS, "hidden_size": D, "n_rows": N_ROWS, "dtype_saved": "float16", "max_rows": None}
    files[f"{FOLDER}/meta.json"] = json.dumps(meta).encode()
    reference = pd.read_parquet(parquet)
    return files, layers, reference, meta


def _stream(source, layer=1, **kwargs):
    kwargs.setdefault("retry_delay", 0.0)
    return iter_remote_feature_batches(
        "team/features", "fake", "pretrained", "first", "ewt", "train", layer,
        revision="main", source=source, **kwargs,
    )


def _collect(batches):
    batches = list(batches)
    if not batches:
        return np.empty(0, dtype=np.int64), np.empty((0, D), dtype=np.float16), pd.DataFrame()
    return (
        np.concatenate([b.rows for b in batches]),
        np.concatenate([b.features for b in batches]),
        pd.concat([b.table for b in batches]),
    )


def test_full_stream_matches_local(local):
    files, layers, reference, _ = local
    batches = list(_stream(MemorySource(files), batch_size=5))
    assert [len(b.rows) for b in batches] == [5, 5, 5, 5, 3]
    rows, features, table = _collect(batches)
    np.testing.assert_array_equal(rows, np.arange(N_ROWS))
    assert features.dtype == np.float16
    np.testing.assert_array_equal(features, layers[1])
    assert np.isnan(features).any(axis=1).tolist() == np.isnan(layers[1]).any(axis=1).tolist()
    pd.testing.assert_frame_equal(table, reference)


def test_partial_range_and_final_partial_batch(local):
    files, layers, reference, _ = local
    rows, features, table = _collect(_stream(MemorySource(files), batch_size=4, start_row=3, stop_row=18))
    np.testing.assert_array_equal(rows, np.arange(3, 18))
    np.testing.assert_array_equal(features, layers[1][3:18])
    pd.testing.assert_frame_equal(table, reference.iloc[3:18])


def test_batches_have_the_requested_size_and_the_last_one_is_smaller(local):
    files, *_ = local
    sizes = [len(b.rows) for b in _stream(MemorySource(files), batch_size=10, start_row=1, stop_row=22)]
    assert sizes == [10, 10, 1]


@pytest.mark.parametrize("start,stop", [(5, 5), (0, 0), (N_ROWS, N_ROWS)])
def test_empty_range_yields_nothing(local, start, stop):
    files, *_ = local
    assert list(_stream(MemorySource(files), start_row=start, stop_row=stop)) == []


def test_next_upos_is_stored_value_even_when_a_batch_splits_a_sentence(local):
    files, _, reference, _ = local
    batches = list(_stream(MemorySource(files), batch_size=2))
    # Batch [2, 4) starts at the last word of sentence s0 (3 words) and ends inside sentence s1.
    split = batches[1]
    assert list(split.rows) == [2, 3]
    assert pd.isna(split.table["next_upos"].iloc[0])  # row 2 ends sentence s0
    assert not pd.isna(split.table["next_upos"].iloc[1])  # row 3 starts s1, next word exists
    pd.testing.assert_series_equal(split.table["next_upos"], reference["next_upos"].iloc[2:4])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0}, {"batch_size": -3}, {"start_row": -1}, {"start_row": 8, "stop_row": 3},
        {"stop_row": N_ROWS + 1}, {"start_row": N_ROWS + 1}, {"layer": N_LAYERS + 1}, {"layer": -1},
        {"max_retries": -1},
    ],
)
def test_invalid_arguments_fail_when_the_function_is_called(local, kwargs):
    files, *_ = local
    with pytest.raises(ValueError):
        _stream(MemorySource(files), **kwargs)  # no iteration: errors are raised eagerly


def test_empty_revision_is_rejected(local):
    files, *_ = local
    with pytest.raises(ValueError, match="revision"):
        iter_remote_feature_batches(
            "t/f", "fake", "pretrained", "first", "ewt", "train", 1, revision="", source=MemorySource(files)
        )


@pytest.mark.parametrize(
    "change",
    [{"model": "other"}, {"split": "test"}, {"hidden_size": D + 1}, {"n_rows": N_ROWS - 1},
     {"dtype_saved": "float32"}, {"pooling": "last"}],
)
def test_metadata_mismatch_fails(local, change):
    files, _, _, meta = local
    files = dict(files)
    files[f"{FOLDER}/meta.json"] = json.dumps({**meta, **change}).encode()
    with pytest.raises(ValueError):
        _stream(MemorySource(files))


def test_missing_metadata_means_incomplete_extraction(local):
    files, *_ = local
    files = {k: v for k, v in files.items() if not k.endswith("meta.json")}
    with pytest.raises(FileNotFoundError):
        _stream(MemorySource(files))


def test_missing_layer_and_missing_word_table(local):
    files, *_ = local
    with pytest.raises(FileNotFoundError):
        _stream(MemorySource({k: v for k, v in files.items() if k != LAYER_FILE}))
    with pytest.raises(FileNotFoundError):
        _stream(MemorySource({k: v for k, v in files.items() if k != TABLE_FILE}))


def test_truncated_layer_file_is_rejected(local):
    files, *_ = local
    files = dict(files)
    files[LAYER_FILE] = files[LAYER_FILE][:-ROW_BYTES]
    with pytest.raises(ValueError, match="incomplete"):
        _stream(MemorySource(files))


def test_word_table_with_other_row_count_is_rejected(local):
    files, _, reference, _ = local
    short = io.BytesIO()
    reference.iloc[:20].to_parquet(short)
    files = {**files, TABLE_FILE: short.getvalue()}
    with pytest.raises(ValueError, match="word table"):
        _stream(MemorySource(files))


def test_capped_features_cover_a_prefix_of_the_word_table(local, tmp_path):
    files, layers, reference, meta = local
    cap = 15
    capped = dict(files)
    for layer in range(N_LAYERS + 1):
        path = tmp_path / f"cap_{layer}.npy"
        memmap = open_memmap(path, mode="w+", dtype=np.float16, shape=(cap, D))
        memmap[:] = layers[layer][:cap]
        memmap.flush()
        del memmap
        capped[f"{FOLDER}/layer_{layer:02d}.npy"] = path.read_bytes()
    capped[f"{FOLDER}/meta.json"] = json.dumps({**meta, "n_rows": cap, "max_rows": cap}).encode()
    rows, features, table = _collect(_stream(MemorySource(capped), batch_size=4))
    np.testing.assert_array_equal(rows, np.arange(cap))  # default stop is n_rows, not the table length
    np.testing.assert_array_equal(features, layers[1][:cap])
    pd.testing.assert_frame_equal(table, reference.iloc[:cap])
    with pytest.raises(ValueError, match="beyond"):
        _stream(MemorySource(capped), stop_row=cap + 1)


def test_only_the_requested_layer_and_row_range_are_fetched_and_buffers_are_bounded(local):
    files, *_ = local
    source = MemorySource(files)
    batch_size = 4
    list(_stream(source, batch_size=batch_size, start_row=8, stop_row=16))
    touched = {path for path, _, _ in source.reads}
    assert touched <= {f"{FOLDER}/meta.json", LAYER_FILE, TABLE_FILE}
    layer_reads = [(start, length) for path, start, length in source.reads if path == LAYER_FILE]
    header, *data_reads = layer_reads
    assert header == (0, HEADER_PROBE_BYTES)
    assert max(length for _, length in data_reads) <= batch_size * ROW_BYTES  # one batch at a time
    transferred_layer = sum(min(length, len(files[LAYER_FILE]) - start) for start, length in data_reads)
    assert transferred_layer == (16 - 8) * ROW_BYTES  # exactly the requested rows, nothing else


def test_stream_is_lazy_and_reads_batch_by_batch(local):
    files, *_ = local
    source = MemorySource(files)
    iterator = iter(_stream(source, batch_size=4))
    before = len([r for r in source.reads if r[0] == LAYER_FILE])
    next(iterator)
    after_first = len([r for r in source.reads if r[0] == LAYER_FILE])
    assert after_first == before + 1  # exactly one data read per batch, no read-ahead
    next(iterator)
    assert len([r for r in source.reads if r[0] == LAYER_FILE]) == after_first + 1


def test_features_prefix_selects_a_folder_of_the_repository(local):
    files, layers, *_ = local
    nested = {(f"features/{k}" if k != TABLE_FILE else k): v for k, v in files.items()}
    rows, features, _ = _collect(_stream(MemorySource(nested), features_prefix="features/"))
    np.testing.assert_array_equal(features, layers[1])
    with pytest.raises(FileNotFoundError):
        _stream(MemorySource(nested))  # default prefix: the root has no such folder


def test_retries_do_not_duplicate_or_skip_rows(local):
    files, layers, reference, _ = local
    probe = MemorySource(files)
    list(_stream(probe, batch_size=5))
    total_calls = probe.n_calls
    # Fail every single read call once (one at a time), including header, metadata, parquet and data reads.
    for failing_call in range(1, total_calls + 1):
        source = MemorySource(files, failures={failing_call: TransientReadError("boom")})
        rows, features, table = _collect(_stream(source, batch_size=5, max_retries=2))
        np.testing.assert_array_equal(rows, np.arange(N_ROWS))
        np.testing.assert_array_equal(features, layers[1])
        pd.testing.assert_frame_equal(table, reference)


def test_retries_are_bounded(local):
    files, *_ = local
    clean = MemorySource(files)
    first = next(iter(_stream(clean, batch_size=5)))
    data_call = clean.n_calls  # the call that read the first batch
    failures = {data_call + i: TransientReadError("boom") for i in range(1, 10)}
    source = MemorySource(files, failures=failures)
    batches = _stream(source, batch_size=5, max_retries=2)
    got = [next(batches)]
    with pytest.raises(ConnectionError, match="after 2 retries"):
        for batch in batches:
            got.append(batch)
    np.testing.assert_array_equal(got[0].rows, first.rows)  # what was yielded before the failure is intact
    assert len(got) == 1


@pytest.mark.parametrize("error", [PermissionError("denied"), FileNotFoundError("gone")])
def test_access_errors_are_not_retried(local, error):
    files, *_ = local
    source = MemorySource(files, failures={1: error})
    with pytest.raises(type(error)):
        _stream(source, max_retries=3)
    assert source.n_calls == 1


def test_parse_npy_header(local):
    files, *_ = local
    shape, dtype, offset = parse_npy_header(files[LAYER_FILE][:HEADER_PROBE_BYTES])
    assert shape == (N_ROWS, D) and dtype == np.float16
    assert len(files[LAYER_FILE]) == offset + N_ROWS * ROW_BYTES


def test_parse_npy_header_rejects_other_formats():
    def header_of(array, **kwargs):
        stream = io.BytesIO()
        np.lib.format.write_array(stream, array, **kwargs)
        return stream.getvalue()[:HEADER_PROBE_BYTES]

    with pytest.raises(ValueError):
        parse_npy_header(b"not a npy file at all")
    with pytest.raises(ValueError, match="float16"):
        parse_npy_header(header_of(np.zeros((3, 2), dtype=np.float32)))
    with pytest.raises(ValueError, match="2-D"):
        parse_npy_header(header_of(np.zeros(3, dtype=np.float16)))
    with pytest.raises(ValueError, match="C-ordered"):
        parse_npy_header(header_of(np.asfortranarray(np.zeros((3, 2), dtype=np.float16))))


class FakeFileSystem:
    def __init__(self, files=None, error=None):
        self.files, self.error, self.opened = files or {}, error, []

    def size(self, path):
        if self.error:
            raise self.error
        return len(self.files[path])

    def open(self, path, mode, **kwargs):
        if self.error:
            raise self.error
        self.opened.append((path, kwargs))
        return io.BytesIO(self.files[path])


class FakeApi:
    def __init__(self, sha="b" * 40):
        self.sha = sha
        self.calls = []

    def repo_info(self, repo_id, repo_type, revision):
        self.calls.append((repo_id, repo_type, revision))
        return type("Info", (), {"sha": self.sha})()


def test_hf_source_pins_the_commit_and_reads_exact_ranges():
    path = "features/x/file.bin"
    api = FakeApi()
    fs = FakeFileSystem({f"datasets/team/features@{api.sha}/{path}": bytes(range(100))})
    source = HfRangeSource("team/features", "main", fs=fs, api=api)
    assert api.calls == [("team/features", "dataset", "main")] and source.revision == api.sha
    assert source.size(path) == 100
    assert source.read(path, 10, 5) == bytes(range(10, 15))
    assert source.bytes_transferred == 5
    assert fs.opened[0][1] == {"cache_type": "none"}  # no read-ahead beyond the requested range


def test_hf_source_errors_have_clear_messages_without_details():
    api = FakeApi()
    with pytest.raises(FileNotFoundError, match="not found"):
        HfRangeSource("t/f", "main", fs=FakeFileSystem(error=FileNotFoundError("secret-token-xyz")), api=api).size("a")
    with pytest.raises(PermissionError, match="login") as denied:
        HfRangeSource("t/f", "main", fs=FakeFileSystem(error=PermissionError("secret-token-xyz")), api=api).size("a")
    assert "secret-token-xyz" not in str(denied.value)
    with pytest.raises(TransientReadError, match="Network error") as network:
        HfRangeSource("t/f", "main", fs=FakeFileSystem(error=ConnectionError("secret-token-xyz")), api=api).size("a")
    assert "secret-token-xyz" not in str(network.value)
    with pytest.raises(ValueError, match="repo_type"):
        HfRangeSource("t/f", "main", repo_type="nope", fs=FakeFileSystem(), api=api)


@pytest.mark.slow
def test_integration_small_range_from_private_repository():
    """Reads 64 rows from a pinned private revision. Set HLT_HF_TEST_REPO and HLT_HF_TEST_REVISION (a commit hash)."""
    repo, revision = os.environ.get("HLT_HF_TEST_REPO"), os.environ.get("HLT_HF_TEST_REVISION")
    if not repo or not revision:
        pytest.skip("HLT_HF_TEST_REPO and HLT_HF_TEST_REVISION are not set")
    model = os.environ.get("HLT_HF_TEST_MODEL", "gpt2")
    treebank = os.environ.get("HLT_HF_TEST_TREEBANK", "ewt")
    source = HfRangeSource(repo, revision)
    batches = list(iter_remote_feature_batches(
        repo, model, "pretrained", "first", treebank, "test", 6,
        revision=revision, batch_size=32, start_row=0, stop_row=64, source=source,
    ))
    assert sum(len(b.rows) for b in batches) == 64
    print(f"transferred {source.bytes_transferred} bytes for 64 rows")
