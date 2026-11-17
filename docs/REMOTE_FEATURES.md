# Streaming features from the private Hugging Face repository (issue DS)

Optional. Use it when the shared features are too large to download in full. Local extraction and
`load_layer` / `load_meta` / `load_word_table` are unchanged and remain the default.
The interface is defined in [CONTRACTS.md](CONTRACTS.md) sections 4.1 and 10.

## 1. What it does

`lib.remote_features.iter_remote_feature_batches` reads **one layer** of one feature folder, and the matching word-table rows,
in batches of `batch_size` word rows, with HTTP range requests. Nothing is written to disk and the whole layer is never downloaded.
A `.npy` file is a header followed by raw float16 rows, so the row range `[a, b)` is the byte range
`header + a*d*2 ... header + b*d*2`; the existing files already support this, so no conversion or sharding is needed.

## 2. Setup

1. `pip install -r requirements.txt` (`huggingface_hub` is installed with `transformers`; the unit tests do not need it).
2. Log in with your own account, which must have access to the team repository: `hf auth login`
   (older versions: `huggingface-cli login`), or set the `HF_TOKEN` environment variable. **Never** put a token in code,
   in a committed file or in a command that is logged. The module never prints or stores it.
3. Ask the owner for the repository id and for a **commit hash** to read. Always read a commit, not `main`:
   a branch name is resolved once to its commit, and that commit is used for the features, the word table and `meta.json`,
   so a result can be reproduced. The CLI prints the resolved commit.

Layout of the repository `hlt-group11/features` (dataset, private; see CONTRACTS.md 4.1):

    {model}/{init}/{pooling}/{treebank}_{split}/layer_{l:02d}.npy
    {model}/{init}/{pooling}/{treebank}_{split}/meta.json
    data/processed/{treebank}_{split}.parquet        # the word tables

The feature folders are at the root (the content of `features/`); use `features_prefix` / `--features-prefix` if a copy keeps them under a folder.
The word tables were copied from `data/processed/` after checking that their row counts equal `n_rows` of all 25 `meta.json` files.
When the Italian or multilingual features are uploaded, upload the matching word tables too (they are already there for `isdt` and `pud_it`).

## 3. Use

```python
from lib.remote_features import iter_remote_feature_batches

for batch in iter_remote_feature_batches(
    "hlt-group11/features", "gpt2", "pretrained", "first", "ewt", "train", 6,
    revision="<commit hash>", batch_size=512,
):
    batch.rows       # int64 [B], indices in the word table
    batch.features   # float16 [B, d], NaN rows kept
    batch.table      # the B word-table rows (sent_id, word_idx, form, upos, next_upos, lang), index = batch.rows
```

- Rows are never filtered. Drop NaN rows yourself, as with `load_layer`.
- `next_upos` is the stored value, also for the last row of a batch that cuts a sentence.
- **Control labels** are not in the word table. Build them on the **whole** word table and index with `batch.rows`:
  `control_next_labels` needs the next word of the same sentence, which may be in another batch.
- If `meta.json` has `max_rows` (capped mGPT train), the default `stop_row` is `n_rows` (the capped size), not the word-table length.
- Arguments, credentials, revision, metadata, the `.npy` header and the word table are checked when the function is called
  (not at the first `next()`), so mistakes appear before any data is read.

Command line (reads the batches one by one and keeps nothing):

```bash
PYTHONPATH=src python scripts/05_stream_features.py --repo-id hlt-group11/features --revision <commit> \
    --model gpt2 --treebank ewt --split test --layer 6 --start-row 0 --stop-row 2000 --batch-size 512
```

It prints the commit, one line per batch (`--print-every` to thin it), the rows, NaN rows, bytes transferred and peak memory
(peak memory is not available on Windows).

## 4. Memory and transfer

- Features: one request per batch, of exactly `batch_size * d * 2` bytes (for example 512 rows of a 768-dim model: 0.79 MB;
  mGPT, d = 2048: 2.1 MB). **No read-ahead**: the transport opens files with `cache_type="none"`.
- Word table: parquet is read by row group. A row group is the smallest readable unit, so the memory used by the word table is
  one row group (the whole split if it was written as a single group; ISDT train, 276,014 rows, is a few tens of MB as a DataFrame).
  The footer is read first (a few tens of KB).
- Also read once: `meta.json` and the first 1 KB of the layer file (the header).
- Memory does not depend on the size of the dataset, only on `batch_size`, `d` and the row-group size.
- **Repeated passes transfer the same data again**: nothing is cached on disk. For several epochs over the same rows, download that
  slice once instead (or accept the repeated traffic).
- "Bytes transferred" in the CLI counts the payload of the range reads, not HTTP headers.

## 5. Errors and retries

| Situation | Exception |
| :-- | :-- |
| invalid `batch_size`, `layer`, row range, metadata or header mismatch, truncated upload, word-table mismatch | `ValueError` |
| missing revision, layer, `meta.json` (incomplete extraction) or word table | `FileNotFoundError` |
| repository not visible to your login, or access denied | `PermissionError` |
| network, timeout, HTTP 408/429/5xx, short read | retried `max_retries` times (default 3, `retry_delay` 1 s), then `ConnectionError` |

A batch is yielded only after its reads succeeded, so a retry never duplicates or skips rows. Messages never contain tokens or request headers.
Access errors are not retried. It never falls back to downloading the layer or the dataset.

## 6. Tests

`tests/test_remote_features.py` uses an in-memory transport on synthetic features: exactness against local slices, batch and
row-group boundaries, NaN rows, partial/empty ranges, invalid arguments, metadata mismatches, instrumented reads (only the requested
layer and rows are fetched, one batch per read), a failure injected in every read in turn (no duplicate or missing rows),
and the Hugging Face transport with a fake file system. No network and no credentials.

One optional integration test reads 64 rows from a real private revision:

```bash
HLT_HF_TEST_REPO=hlt-group11/features HLT_HF_TEST_REVISION=<commit> PYTHONPATH=src pytest -q -m slow tests/test_remote_features.py -s
```

Paste the printed transferred bytes (and the peak memory of the CLI run above) in the PR or issue.

## 7. Limits

- Only one layer per stream; it is not wired into `LinearProbe`, `MLPProbe`, CKA or their runners.
- No shuffling, standardisation or hyperparameter search over a stream.
- Speed depends on the network: a batch is one round trip.
