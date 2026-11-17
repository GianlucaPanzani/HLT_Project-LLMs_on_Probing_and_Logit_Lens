"""Optional streaming of extracted features from a private Hugging Face repository.

Implements CONTRACTS.md sections 4.1 and 10 (``lib/remote_features.py``). The
local workflow (``lib.features.load_layer`` and friends) is not touched: this
module only adds a batch iterator that reads a row range of one layer, and the
matching word-table rows, with HTTP range requests.

A ``.npy`` file is a small header followed by raw row-major data, so row ``i``
of a float16 ``[N, d]`` layer starts at ``header_length + i * d * 2``. Nothing
is written to disk and nothing falls back to downloading a whole layer.
"""

import io
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypeVar

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib import format as npy_format

try:  # optional dependency: only the default Hugging Face transport needs it
    from huggingface_hub import HfApi, HfFileSystem
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )
except ImportError:  # pragma: no cover - exercised only without huggingface_hub
    HfApi = HfFileSystem = None

    class HfHubHTTPError(Exception):
        """Placeholder so that the except clauses below stay valid without huggingface_hub."""

    class GatedRepoError(HfHubHTTPError):
        """Placeholder."""

    class RepositoryNotFoundError(HfHubHTTPError):
        """Placeholder."""

    class RevisionNotFoundError(HfHubHTTPError):
        """Placeholder."""

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None
try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

logger = logging.getLogger(__name__)

T = TypeVar("T")

HEADER_PROBE_BYTES = 1024
"""Bytes read from the start of a ``.npy`` file to parse its header."""

FEATURE_DTYPE = np.dtype("<f2")
REPO_TYPE_PREFIX = {"dataset": "datasets", "model": "models", "space": "spaces"}
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_NETWORK_ERRORS: tuple[type[BaseException], ...] = tuple(
    error
    for error in (
        OSError,
        getattr(httpx, "TransportError", None),
        getattr(requests, "RequestException", None),
    )
    if error is not None
)


class TransientReadError(OSError):
    """A read that failed for a reason worth retrying (network, timeout, 5xx, short read)."""


class RangeSource(Protocol):
    """Byte transport of a remote repository pinned at one commit."""

    revision: str
    bytes_transferred: int

    def size(self, path: str) -> int:
        """Size in bytes of a file of the repository."""

    def read(self, path: str, start: int, length: int) -> bytes:
        """Read ``length`` bytes of a file from offset ``start`` (fewer at the end of the file)."""


@dataclass(frozen=True)
class RemoteBatch:
    """One batch of streamed features.

    Attributes
    ----------
    rows : np.ndarray
        int64 ``[B]``, indices of the rows in the word table, increasing.
    features : np.ndarray
        float16 ``[B, d]``, NaN rows kept.
    table : pd.DataFrame
        The ``B`` word-table rows (all columns of CONTRACTS.md section 2);
        its index is ``rows``.
    """

    rows: np.ndarray
    features: np.ndarray
    table: pd.DataFrame


@dataclass(frozen=True)
class _LayerInfo:
    """Where the data of one remote layer file starts and how it is shaped."""

    offset: int
    n_rows: int
    hidden_size: int


class HfRangeSource:
    """Range reads from a Hugging Face repository, pinned to a commit.

    The user's own Hugging Face credentials are used (``hf auth login``, formerly
    ``huggingface-cli login``, or the ``HF_TOKEN`` environment variable). Tokens are never stored, logged or
    put in error messages.

    Parameters
    ----------
    repo_id : str
        For example ``"team/features"``.
    revision : str
        Branch, tag or commit; resolved once to an immutable commit hash.
    repo_type : str
        ``"dataset"`` (default), ``"model"`` or ``"space"``.
    fs, api : optional
        Injected ``HfFileSystem`` and ``HfApi`` (for tests).

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    PermissionError
        If the repository does not exist for this user or access is denied.
    FileNotFoundError
        If the revision does not exist.
    """

    def __init__(self, repo_id: str, revision: str, repo_type: str = "dataset", *, fs=None, api=None):
        if repo_type not in REPO_TYPE_PREFIX:
            raise ValueError(f"repo_type must be one of {sorted(REPO_TYPE_PREFIX)}, got '{repo_type}'.")
        if (fs is None or api is None) and HfApi is None:
            raise ImportError("huggingface_hub is required to read from Hugging Face: pip install huggingface_hub")
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.bytes_transferred = 0
        self._fs = fs if fs is not None else HfFileSystem()
        api = api if api is not None else HfApi()
        try:
            self.revision = api.repo_info(repo_id, repo_type=repo_type, revision=revision).sha
        except RevisionNotFoundError:
            raise FileNotFoundError(f"Revision '{revision}' not found in {repo_type} repository '{repo_id}'.") from None
        except (RepositoryNotFoundError, GatedRepoError):
            raise PermissionError(
                f"Cannot access {repo_type} repository '{repo_id}': it does not exist or your Hugging Face "
                "login has no access. Log in with `hf auth login` (or set HF_TOKEN) using an account of the team."
            ) from None
        except HfHubHTTPError as error:
            raise self._translate(error, f"repository '{repo_id}'") from None
        except _NETWORK_ERRORS as error:
            raise ConnectionError(f"Cannot reach Hugging Face ({type(error).__name__}).") from None

    def _full_path(self, path: str) -> str:
        return f"{REPO_TYPE_PREFIX[self.repo_type]}/{self.repo_id}@{self.revision}/{path}"

    @staticmethod
    def _translate(error: Exception, what: str) -> Exception:
        """Map an HTTP error to a builtin exception, without echoing request details."""
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        if status in (401, 403):
            return PermissionError(f"Access denied to {what} (HTTP {status}); check your Hugging Face login.")
        if status == 404:
            return FileNotFoundError(f"{what} not found (HTTP 404).")
        if status in _TRANSIENT_STATUS:
            return TransientReadError(f"Temporary error reading {what} (HTTP {status}).")
        return OSError(f"Error reading {what} (HTTP {status}).")

    def _call(self, function: Callable[[], T], path: str) -> T:
        what = f"'{path}' in {self.repo_id}@{self.revision[:8]}"
        try:
            return function()
        except FileNotFoundError:
            raise FileNotFoundError(f"{what} not found.") from None
        except PermissionError:
            raise PermissionError(f"Access denied to {what}; check your Hugging Face login.") from None
        except HfHubHTTPError as error:
            raise self._translate(error, what) from None
        except _NETWORK_ERRORS as error:
            raise TransientReadError(f"Network error reading {what} ({type(error).__name__}).") from None

    def size(self, path: str) -> int:
        return int(self._call(lambda: self._fs.size(self._full_path(path)), path))

    def read(self, path: str, start: int, length: int) -> bytes:
        def fetch() -> bytes:
            # cache_type="none": fetch exactly the requested range, no read-ahead.
            with self._fs.open(self._full_path(path), "rb", cache_type="none") as file:
                file.seek(start)
                return file.read(length)

        data = self._call(fetch, path)
        self.bytes_transferred += len(data)
        return data


class _RangeFile(io.RawIOBase):
    """Read-only seekable file object on top of a ``RangeSource`` (for pyarrow)."""

    def __init__(self, source: RangeSource, path: str):
        self._source = source
        self._path = path
        self._size = source.size(path)
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._position = offset
        elif whence == io.SEEK_CUR:
            self._position += offset
        elif whence == io.SEEK_END:
            self._position = self._size + offset
        else:
            raise ValueError(f"Invalid whence {whence}.")
        return self._position

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._position
        size = max(0, min(size, self._size - self._position))
        data = self._source.read(self._path, self._position, size) if size else b""
        self._position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


class _TableReader:
    """Reads row ranges of a remote word table, one parquet row group at a time."""

    def __init__(self, source: RangeSource, path: str, retry):
        self._retry = retry
        self._parquet = retry(lambda: pq.ParquetFile(_RangeFile(source, path)))
        groups = self._parquet.metadata
        self.num_rows = groups.num_rows
        self._bounds = np.cumsum([0] + [groups.row_group(i).num_rows for i in range(groups.num_row_groups)])
        self._cached: tuple[int, pd.DataFrame] | None = None

    def _group(self, index: int) -> pd.DataFrame:
        if self._cached is None or self._cached[0] != index:
            frame = self._retry(lambda: self._parquet.read_row_group(index).to_pandas())
            self._cached = (index, frame.reset_index(drop=True))
        return self._cached[1]

    def rows(self, start: int, stop: int) -> pd.DataFrame:
        """Word-table rows ``[start, stop)`` with ``RangeIndex(start, stop)``."""
        first = int(np.searchsorted(self._bounds, start, side="right")) - 1
        pieces = []
        for index in range(first, len(self._bounds) - 1):
            low, high = int(self._bounds[index]), int(self._bounds[index + 1])
            if low >= stop:
                break
            frame = self._group(index)
            pieces.append(frame.iloc[max(start, low) - low: min(stop, high) - low])
        result = pd.concat(pieces)
        result.index = pd.RangeIndex(start, stop)
        return result


def parse_npy_header(prefix: bytes) -> tuple[tuple[int, ...], np.dtype, int]:
    """Parse the header of a ``.npy`` file.

    Parameters
    ----------
    prefix : bytes
        The first bytes of the file (at least the whole header).

    Returns
    -------
    tuple
        ``(shape, dtype, data_offset)`` where ``data_offset`` is the byte
        offset of the first array element.

    Raises
    ------
    ValueError
        If the header is invalid, truncated, uses Fortran order or is not a
        2-D little-endian float16 array.
    """
    stream = io.BytesIO(prefix)
    try:
        version = npy_format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = npy_format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = npy_format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"unsupported .npy version {version}")
    except (ValueError, EOFError, SyntaxError) as error:
        raise ValueError(f"Invalid or truncated .npy header: {error}") from None
    if fortran or len(shape) != 2 or dtype != FEATURE_DTYPE:
        raise ValueError(f"Expected a C-ordered 2-D float16 array, got shape {shape}, dtype {dtype}, fortran={fortran}.")
    return tuple(shape), dtype, stream.tell()


def _make_retry(max_retries: int, retry_delay: float) -> Callable[[Callable[[], T]], T]:
    """Return ``retry(function)``: call it again after a ``TransientReadError``, up to ``max_retries`` times."""

    def retry(function: Callable[[], T]) -> T:
        for attempt in range(max_retries + 1):
            try:
                return function()
            except TransientReadError as error:
                if attempt == max_retries:
                    raise ConnectionError(f"Remote read failed after {max_retries} retries: {error}") from None
                logger.warning("Retrying remote read (%d/%d): %s", attempt + 1, max_retries, error)
                time.sleep(retry_delay)
        raise AssertionError("unreachable")

    return retry


def _read_json(source: RangeSource, path: str, retry) -> dict:
    data = retry(lambda: source.read(path, 0, source.size(path)))
    return json.loads(data.decode("utf-8"))


def _check_arguments(layer: int, batch_size: int, start_row: int, stop_row: int | None, revision: str) -> None:
    if not isinstance(revision, str) or not revision:
        raise ValueError("revision must be a non-empty string (a branch, tag or commit).")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}.")
    if not isinstance(layer, int) or layer < 0:
        raise ValueError(f"layer must be a non-negative integer, got {layer!r}.")
    if start_row < 0:
        raise ValueError(f"start_row must be >= 0, got {start_row}.")
    if stop_row is not None and stop_row < start_row:
        raise ValueError(f"stop_row ({stop_row}) must be >= start_row ({start_row}).")


def _validate(meta: dict, expected: dict, layer: int) -> None:
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"meta.json says {key}={meta.get(key)!r}, expected {value!r}.")
    for key in ("num_layers", "hidden_size", "n_rows", "dtype_saved"):
        if key not in meta:
            raise ValueError(f"meta.json has no '{key}'.")
    if meta["dtype_saved"] != "float16":
        raise ValueError(f"meta.json says dtype_saved={meta['dtype_saved']!r}, expected 'float16'.")
    if not 0 <= layer <= meta["num_layers"]:
        raise ValueError(f"layer must be in [0, {meta['num_layers']}], got {layer}.")


def iter_remote_feature_batches(
    repo_id: str,
    model: str,
    init: str,
    pooling: str,
    treebank: str,
    split: str,
    layer: int,
    *,
    revision: str,
    batch_size: int = 512,
    start_row: int = 0,
    stop_row: int | None = None,
    repo_type: str = "dataset",
    max_retries: int = 3,
    retry_delay: float = 1.0,
    source: RangeSource | None = None,
) -> Iterator[RemoteBatch]:
    """Stream one layer of extracted features, with its word-table rows, in bounded batches.

    Everything is checked when this function is called, before the first
    batch is read (see CONTRACTS.md section 10). Only the requested layer
    range, ``meta.json`` and the row groups of the word table that cover the
    range are fetched; nothing is written to disk.

    Parameters
    ----------
    repo_id : str
        Private Hugging Face repository, for example ``"team/features"``.
    model, init, pooling, treebank, split : str
        Feature folder keys (CONTRACTS.md section 4).
    layer : int
        Layer index, 0 to L.
    revision : str
        Branch, tag or commit; resolved once to a commit hash that is used for
        the features, the word table and the metadata.
    batch_size : int
        Word rows per batch; the last batch may be smaller. Defaults to 512.
    start_row, stop_row : int, int | None
        Half-open range of word-table rows. ``stop_row=None`` means the number
        of feature rows (``meta["n_rows"]``). Defaults to the whole layer.
    repo_type : str
        ``"dataset"`` (default), ``"model"`` or ``"space"``.
    max_retries : int
        Retries of a failed read before giving up. Defaults to 3.
    retry_delay : float
        Seconds between retries. Defaults to 1.0.
    source : RangeSource | None
        Byte transport. Defaults to ``HfRangeSource(repo_id, revision, repo_type)``.

    Returns
    -------
    Iterator[RemoteBatch]
        Batches in row order. Rows are never filtered: NaN rows and null
        ``next_upos`` are kept, and ``rows`` are global word-table indices.

    Raises
    ------
    ValueError
        Invalid argument, row range outside the features, or metadata, header
        and word table that do not match.
    FileNotFoundError
        Missing revision or artifact (also an incomplete extraction: no ``meta.json``).
    PermissionError
        Missing or denied access to the repository.
    ConnectionError
        A read still failing after ``max_retries`` retries.
    """
    _check_arguments(layer, batch_size, start_row, stop_row, revision)
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}.")
    if source is None:
        source = HfRangeSource(repo_id, revision, repo_type)
    retry = _make_retry(max_retries, retry_delay)

    folder = f"features/{model}/{init}/{pooling}/{treebank}_{split}"
    meta = _read_json(source, f"{folder}/meta.json", retry)
    _validate(
        meta,
        {"model": model, "init": init, "pooling": pooling, "treebank": treebank, "split": split},
        layer,
    )

    layer_file = f"{folder}/layer_{layer:02d}.npy"
    header = retry(lambda: source.read(layer_file, 0, HEADER_PROBE_BYTES))
    (n_rows, hidden_size), _, offset = parse_npy_header(header)
    if (n_rows, hidden_size) != (meta["n_rows"], meta["hidden_size"]):
        raise ValueError(
            f"{layer_file} has shape {(n_rows, hidden_size)} but meta.json says "
            f"{(meta['n_rows'], meta['hidden_size'])}."
        )
    file_size = retry(lambda: source.size(layer_file))
    if file_size != offset + n_rows * hidden_size * FEATURE_DTYPE.itemsize:
        raise ValueError(f"{layer_file} has {file_size} bytes: incomplete or corrupted upload.")
    info = _LayerInfo(offset, n_rows, hidden_size)

    table = _TableReader(source, f"data/processed/{treebank}_{split}.parquet", retry)
    if info.n_rows > table.num_rows or (meta.get("max_rows") is None and info.n_rows != table.num_rows):
        raise ValueError(
            f"Features have {info.n_rows} rows but the word table {treebank}_{split} has {table.num_rows} "
            f"(max_rows={meta.get('max_rows')})."
        )

    stop = info.n_rows if stop_row is None else stop_row
    if stop > info.n_rows:
        raise ValueError(f"stop_row ({stop}) is beyond the {info.n_rows} feature rows of this layer.")
    if start_row > stop:
        raise ValueError(f"start_row ({start_row}) is beyond the {info.n_rows} feature rows of this layer.")

    logger.info("Streaming %s rows [%d, %d) of %s at commit %s", layer_file, start_row, stop, repo_id, source.revision)
    return _generate_batches(source, retry, layer_file, info, table, start_row, stop, batch_size)


def _generate_batches(
    source: RangeSource,
    retry,
    layer_file: str,
    info: _LayerInfo,
    table: _TableReader,
    start: int,
    stop: int,
    batch_size: int,
) -> Iterator[RemoteBatch]:
    row_bytes = info.hidden_size * FEATURE_DTYPE.itemsize
    for low in range(start, stop, batch_size):
        high = min(low + batch_size, stop)

        def read_features(low=low, high=high) -> np.ndarray:
            data = source.read(layer_file, info.offset + low * row_bytes, (high - low) * row_bytes)
            if len(data) != (high - low) * row_bytes:
                raise TransientReadError(f"short read of {layer_file}: {len(data)} bytes")
            return np.frombuffer(data, dtype=FEATURE_DTYPE).reshape(high - low, info.hidden_size).astype(np.float16)

        features = retry(read_features)
        rows = table.rows(low, high)
        yield RemoteBatch(rows=np.arange(low, high, dtype=np.int64), features=features, table=rows)
