# Feature extraction: how to run, maintain and extend it

Owner until Nov 16: Marcin (issues E1 #3, E2 #4, E3 #5, E4 #6, E5 #7). After that: whoever runs extraction (Arcangelo for #8).
Formats are defined in [CONTRACTS.md](CONTRACTS.md) sections 3 to 5; this file only explains how to produce them.

## 1. What the code does

| Module | Role |
| :-- | :-- |
| `src/lib/alignment.py` | `encode_words` turns pre-split UD words into model inputs (BOS prepended for decoders) and keeps `word_ids`; `pool_words` turns subword states into one vector per UD word (`first`, `last` or `mean`). Words without any subword (truncated, or characters the tokenizer deletes) become NaN rows, never a shift. |
| `src/lib/features.py` | `extract_features` runs a frozen model over a word table and writes `layer_00.npy ... layer_{L}.npy` (float16, `[N, d]`, rows in word-table order) and then `meta.json`. `load_layer` and `load_meta` read them. `extract_into` is the same loop without files (used by the tests). |
| `src/lib/control_tasks.py` | Control labels of Hewitt and Liang (2019): one random fixed label per word type of the train treebank, unseen words seeded with md5. |
| `scripts/01_extract_features.py` | Command line for `extract_features`. |
| `scripts/03_build_control.py` | Writes `data/processed/control_ewt.json` and `control_isdt.json`. |
| `src/lib/feature_report.py`, `scripts/04_report_features.py` | Read-only status table of the Italian and multilingual runs (E6): which folders exist, rows, size, time, NaN rows, cap. Never loads a model. |

## 2. Rebuild everything from scratch

Run from the repository root.

```bash
pip install -r requirements.txt
PYTHONPATH=src python scripts/00_download_data.py          # D1: UD r2.19 to data/raw/
PYTHONPATH=src python scripts/02_build_word_tables.py      # D2: data/processed/*.parquet
PYTHONPATH=src python scripts/03_build_control.py          # E3: control_ewt.json, control_isdt.json
PYTHONPATH=src python scripts/01_extract_features.py --model bert --treebank ewt   # E2/E4: one model
```

`01_extract_features.py` flags: `--model` (registry key), `--treebank`, `--split` (default: all splits of the treebank),
`--init pretrained|random`, `--pooling first|last|mean`, `--batch-size` (default 32), `--max-rows`, `--overwrite`.
It prints the folder, its size and the time per split.

Expected control mapping sizes (UD r2.19, measured): `ewt` 19,674 word types (204,578 train tokens), `isdt` 28,302 word types (276,014 train tokens).

## 3. Where the features are

- Default: `features/{model}/{init}/{pooling}/{treebank}_{split}/` (gitignored).
- Set `HLT_FEATURES_DIR=/path/to/disk` to write and read them somewhere else (external disk, mounted shared drive).
- **Shared storage (all 25 English folders, 32.4 GB):** 
  Google Drive folder `features` with the same structure as `features/` here, plus a `README.txt`.
  Download only the folders you need into `features/` (same structure), or anywhere else and set `HLT_FEATURES_DIR`.
- **Hugging Face (private dataset `hlt-group11/features`):** the same feature folders at the repository root (`{model}/{init}/{pooling}/{treebank}_{split}/`),
  plus the word tables in `data/processed/`. It can also be read in bounded batches without downloading it, see [REMOTE_FEATURES.md](REMOTE_FEATURES.md).
  **When you upload features for a new treebank, upload its word table too** (`data/processed/{treebank}_{split}.parquet`), from the same build used for the
  extraction: its number of rows must equal `n_rows` of every `meta.json`, otherwise streaming refuses to read it. Upload only folders that have a `meta.json`.
  Use a token with write access only for the upload and keep it out of files, commands in the repository and logs.

## 4. Time and disk (E4, measured)

Disk is exactly `rows x (L + 1) x d x 2 bytes` (float16): for a 768-dim model, 1,536 bytes per row and layer.
Word table rows (UD r2.19): EWT train 204,578, dev 25,148, test 25,094; PUD EN 21,180; ISDT train 276,014, dev 11,907, test 10,417; PUD IT 23,783.

Measured on Nov 13 on an NVIDIA GTX 1650 Ti (4 GB), batch size 32. Time is the extraction loop summed over the splits
(model loading adds a few seconds per split). Every run: 0 NaN rows, 0 truncated sentences.

| Run | Rows | Disk | Time | `n_nan_rows` | Done |
| :-- | --: | --: | --: | --: | :-- |
| bert ewt pretrained first (train/dev/test) | 254,820 | 5.09 GB | 171 s | 0 | [x] |
| bert pud_en pretrained first | 21,180 | 0.42 GB | 10 s | 0 | [x] |
| bert ewt random first | 254,820 | 5.09 GB | 167 s | 0 | [x] |
| distilbert ewt pretrained first | 254,820 | 2.74 GB | 87 s | 0 | [x] |
| distilbert pud_en pretrained first | 21,180 | 0.23 GB | 5 s | 0 | [x] |
| distilbert ewt random first | 254,820 | 2.74 GB | 90 s | 0 | [x] |
| gpt2 ewt pretrained first | 254,820 | 5.09 GB | 145 s | 0 | [x] |
| gpt2 ewt pretrained last | 254,820 | 5.09 GB | 148 s | 0 | [x] |
| gpt2 pud_en pretrained first | 21,180 | 0.42 GB | 10 s | 0 | [x] |
| gpt2 pud_en pretrained last | 21,180 | 0.42 GB | 10 s | 0 | [x] |
| gpt2 ewt random first | 254,820 | 5.09 GB | 150 s | 0 | [x] |
| Optional: bert / gpt2 ewt mean pooling | 254,820 | 5.09 GB each | about 150 s each | | [ ] |

Total for E4: 25 folders, 32.4 GB, about 20 minutes of GPU time.
Reference speed on the GTX 1650 Ti: about 100 sentences/s for BERT-base and GPT-2, about 190 sentences/s for DistilBERT
(EWT train has 12,544 sentences). On 2 CPU cores a BERT-base-sized model reaches about 23 sentences/s.

## 5. Italian and multilingual runs (E6, issue #8)

**Status: done (issue #8, results in a comment of #8).** All 28 folders (bert_it 4, geppetto 8, mbert 8, mgpt 8) were extracted on an
NVIDIA GeForce RTX 4080 SUPER and uploaded to the Hugging Face repository `hlt-group11/features` (section 3). The runs need a machine with a GPU and enough
free disk (about 53 GB with the mGPT cap); they were not run on the team's 8 GB RAM Mac (16 to 21 GB of free disk), and mGPT does not fit comfortably in 8 GB of RAM.
The rest of this section is the runbook, kept to redo or extend the runs; paste the table of `04_report_features.py` in the issue (see "Report").

Result for every run: `n_nan_rows = 0` and `n_truncated_sentences = 0`, as expected. The Italian alignment was checked in X1
([ws3_italian_notes.md](results/ws3_italian_notes.md)): every ISDT and PUD IT word gets at least one subword with all four
tokenizers, and the longest sequence is 403 positions (mBERT, ISDT train) against a limit of 512. A different result is a bug: report it in #8.

### 5.1 Before the first run

Run from the repository root (PowerShell users: see section 6). Needed once per machine:

```bash
PYTHONPATH=src python scripts/00_download_data.py --treebank ewt isdt pud_en pud_it
PYTHONPATH=src python scripts/02_build_word_tables.py --treebank ewt isdt pud_en pud_it
PYTHONPATH=src python scripts/01_extract_features.py --model bert_it --treebank pud_it    # smoke test, about 1 minute, 0.47 GB
PYTHONPATH=src python scripts/04_report_features.py --models bert_it                        # the pud_it row must say done, 0 NaN rows
```

### 5.2 The runs, in order

Order follows what each run unblocks. Every line covers all splits of the treebank (`train`, `dev`, `test`; PUD only `test`) unless a `--split` is given.
Disk is exact (`rows x (L + 1) x d x 2 bytes`; 12-layer models 19,968 bytes per row, mGPT 102,400 bytes per row).
Time, measured on the RTX 4080 SUPER: about 5,100 rows/s for the 12-layer models (ISDT train, 276,014 rows, in 54 s) and about 1,700 rows/s for mGPT
with `--batch-size 4` (capped EWT train, 49,986 rows, in 30 s), so the whole of E6 took minutes of compute. Per run times are in the report of #8.
On the GTX 1650 Ti of section 4 the 12-layer models are about 2.5 times slower; mGPT was not run there and is much slower on CPU.

| Step | Command (`--init pretrained` is the default) | Disk | Unblocks |
| :-- | :-- | --: | :-- |
| 1 | `--model bert_it --treebank isdt --pooling first` | 5.96 GB | X4, probing of Italian BERT |
| 2 | `--model bert_it --treebank pud_it --pooling first` | 0.47 GB | X4 |
| 3 | `--model geppetto --treebank isdt --pooling first` | 5.96 GB | X4 |
| 4 | `--model geppetto --treebank isdt --pooling last` | 5.96 GB | P5 (`upos_next`), L8 |
| 5 | `--model geppetto --treebank pud_it --pooling first` and `--pooling last` | 0.95 GB | X4, L7, L8 |
| 6 | `--model mbert --treebank ewt --pooling first` | 5.09 GB | X2 |
| 7 | `--model mbert --treebank isdt --pooling first` | 5.96 GB | X2 |
| 8 | `--model mbert --treebank pud_en --pooling first` and `--treebank pud_it` | 0.89 GB | X2 |
| 9 | mGPT, see 5.3 | 22.3 GB (61.2 GB without the cap) | X3 |

Totals: bert_it 6.4 GB, geppetto 12.9 GB, mbert 11.9 GB, mGPT 22.3 GB with the cap.
Prefix every command with `PYTHONPATH=src python scripts/01_extract_features.py`.
Steps 1 to 5 are what P5, X4, L7 and L8 wait for; do them first. Steps 6 to 8 are for X2; step 9 is for X3 and is the slowest, so start it early on a machine of its own if there is one.
Steps are independent: they can be split between machines (each run writes only its own folder).

### 5.3 mGPT (1.3B parameters, 24 layers, d = 2048)

The capped train splits keep whole sentences up to 50,000 rows (so slightly fewer); dev, test and PUD are never capped:

```bash
PYTHONPATH=src python scripts/01_extract_features.py --model mgpt --treebank ewt  --split train --max-rows 50000 --batch-size 4
PYTHONPATH=src python scripts/01_extract_features.py --model mgpt --treebank ewt  --split dev test --batch-size 4
PYTHONPATH=src python scripts/01_extract_features.py --model mgpt --treebank isdt --split train --max-rows 50000 --batch-size 4
PYTHONPATH=src python scripts/01_extract_features.py --model mgpt --treebank isdt --split dev test --batch-size 4
PYTHONPATH=src python scripts/01_extract_features.py --model mgpt --treebank pud_en --batch-size 4
PYTHONPATH=src python scripts/01_extract_features.py --model mgpt --treebank pud_it --batch-size 4
```

| Run | Rows | Disk |
| :-- | --: | --: |
| ewt train (capped) + dev + test | at most 100,242 | 10.26 GB |
| isdt train (capped) + dev + test | at most 72,324 | 7.41 GB |
| pud_en | 21,180 | 2.17 GB |
| pud_it | 23,783 | 2.44 GB |
| Without the cap: ewt 26.09 GB, isdt 30.55 GB | | 61.2 GB in total |

- The cap is the one allowed by CONTRACTS.md section 4 (mGPT train only) and is recorded as `max_rows` in `meta.json`. It must be stated in the X3 results (#29), and every model compared with mGPT there must use the same `--max-train-rows`.
- Weights in float16 take about 2.7 GB of GPU memory, so a 4 GB GPU is tight: keep `--batch-size 4` (try 8 on larger GPUs). Batches are sorted longest first, so an out-of-memory error shows in the first seconds.
- On CPU the script loads it in bfloat16 (recorded in `meta.json`); it works but is very slow.

### 5.4 When the disk is smaller than the output

Run a few steps, upload the folders to the shared storage (section 3), delete them locally and continue. Upload a folder only when it has a `meta.json`
(an incomplete folder cannot be read). Do not delete anything before the upload is checked: the report below can be run again from the shared copy by setting `HLT_FEATURES_DIR`.
Give the shared copy the same structure as `features/`.
For the Hugging Face repository `hlt-group11/features`, also upload the word tables of the treebanks you add (`data/processed/{treebank}_{split}.parquet`; `isdt` and `pud_it` are already there): see section 3 once the streaming PR is merged.

### 5.5 Report

After each batch of runs, on the machine that holds the folders (or with `HLT_FEATURES_DIR` pointing to them):

```bash
PYTHONPATH=src python scripts/04_report_features.py                          # all E6 runs
PYTHONPATH=src python scripts/04_report_features.py --models bert_it geppetto   # only some models
```

Paste the table in a comment of #8: it gives, for every run, the status (`done`, `missing`, or `check cap` if a cap was used where the contract does not allow one),
rows, size, time, NaN rows, truncated sentences, cap, model dtype and device. Then tick the lines of #8 and add the link to the shared storage.
The `size_gb` column is the size of the folder where the report runs; check it against the Disk column above.

## 6. Setup on Windows (tested on Nov 13)

- Use **Python 3.11** (`py -3.11 -m venv .venv`). With Python 3.14 the CUDA builds of PyTorch did not install.
- Install PyTorch with CUDA first, then the rest:
  `pip install torch --index-url https://download.pytorch.org/whl/cu126`, then `pip install -r requirements.txt`.
  Check: `python -c "import torch; print(torch.cuda.is_available())"` must print `True`.
- PowerShell does not understand `PYTHONPATH=src python ...`: run `$env:PYTHONPATH = "src"` once per terminal, then `python ...`.
- For multi-line Python in PowerShell, pipe a here-string: `@' ...code... '@ | python -`.
- Harmless messages: `huggingface_hub ... symlinks` (Windows cache), `unauthenticated requests ... HF_TOKEN`,
  `BertModel LOAD REPORT ... UNEXPECTED cls.*` (the MLM head is not needed), `generation flags are not valid`,
  and `Token indices sequence length is longer than ... (602 > 512)` (the untruncated length check in `lib.alignment`).
- A `.npy` file cannot be overwritten while it is memory-mapped (for example by `load_layer` in an open notebook).
  `extract_features(..., overwrite=True)` then stops with a message saying to close it.

## 7. Things to know when using the features

- **Word table nulls.** The word table uses pandas `string` columns, so a missing `next_upos` (or `control_next`) is `pd.NA`/`None`: test it with `.isna()`, never with `is None`.
- **NaN rows.** Rows of words without any subword are NaN in every layer. Drop them (and the same label rows) before fitting: `mask = ~np.isnan(X).any(axis=1)`. `meta.json` gives `n_nan_rows`.
- **float16 on disk.** Cast to float32 before training: `X = np.asarray(load_layer(...), dtype=np.float32)`.
- **Decoders, last layer.** Hugging Face applies the final layer norm (`ln_f`) only to `hidden_states[L]`. Layers 0..L-1 of GPT-2 are raw residual-stream states whose norm grows with depth, layer L is normalised. `meta.json` records this as `"final_norm_on_last_layer": true`. Standardise features (for example `StandardScaler` fitted on train) before probing so that the probe's regularisation means the same thing at every layer.
- **Decoders, `first` pooling.** The first subword of a word has not seen the rest of the word (causal model). This is intended (main results use `first` for all models); `last` is needed for `upos_next` and the lens bridge.
- **Truncation.** UD sentences are below the limits (longest: 159 words in EWT, 310 in ISDT train), and all English runs had `n_truncated_sentences = 0`. For the Italian models, X1 measured the longest ISDT sequence at 403 positions (mBERT) against a limit of 512, so nothing is expected to be truncated; `n_truncated_sentences` and `n_nan_rows` in the E6 report should be 0. If a word is only partly cut, it keeps the subwords that remain.
- **Random weights.** `--init random` always uses seed 42, so every member gets the same random network.
- **Decoders run without their LM head.** `extract_into` calls `model.base_model`, which returns the same hidden states without computing `[B, T, vocab]` logits (saves GPU memory).
- **Extra `meta.json` fields** (not in the contract): `n_sentences`, `max_length`, `batch_size`, `device`, `seconds`, `final_norm_on_last_layer`.

## 8. Recovering from problems

- **Interrupted run.** A folder without `meta.json` is incomplete; `load_layer` refuses it. Re-run the same command with `--overwrite` for that split.
- **Out of GPU memory.** Lower `--batch-size` (16 or 8). Batches are sorted longest first, so an OOM shows up in the first seconds.
- **Float16 overflow error.** Values outside the float16 range: report it in the issue; do not silence it.
- **mGPT on CPU.** float16 is loaded as bfloat16 on CPU and recorded in `dtype_model`.

## 9. Adding a new model

1. Add an entry to `configs/models.yaml` (registry key, Hugging Face id, kind, L, d, dtype), see CONTRACTS.md section 1.1.
2. Run `PYTHONPATH=src python scripts/01_extract_features.py --model <key> --treebank <tb>`.
3. Nothing else: decoders automatically get BOS and `add_prefix_space=True`.

## 10. Tests

```bash
ruff check src tests scripts app
PYTHONPATH=src pytest -q -m "not slow"   # no downloads; skipped in CI because torch is not installed there
PYTHONPATH=src pytest -q -m slow          # real tokenizers and BERT on EWT test (needs internet and data/)
```

The fast tests build tiny WordPiece and byte-level BPE tokenizers locally (`tests/conftest.py`), so they never download anything.
