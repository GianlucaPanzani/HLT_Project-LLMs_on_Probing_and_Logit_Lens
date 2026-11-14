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
- **Shared storage:** TODO (link to be added when the uploads are done).

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

## 5. Things to know when using the features

- **Word table nulls.** The word table uses pandas `string` columns, so a missing `next_upos` (or `control_next`) is `pd.NA`/`None`: test it with `.isna()`, never with `is None`.
- **NaN rows.** Rows of words without any subword are NaN in every layer. Drop them (and the same label rows) before fitting: `mask = ~np.isnan(X).any(axis=1)`. `meta.json` gives `n_nan_rows`.
- **float16 on disk.** Cast to float32 before training: `X = np.asarray(load_layer(...), dtype=np.float32)`.
- **Decoders, last layer.** Hugging Face applies the final layer norm (`ln_f`) only to `hidden_states[L]`. Layers 0..L-1 of GPT-2 are raw residual-stream states whose norm grows with depth, layer L is normalised. `meta.json` records this as `"final_norm_on_last_layer": true`. Standardise features (for example `StandardScaler` fitted on train) before probing so that the probe's regularisation means the same thing at every layer.
- **Decoders, `first` pooling.** The first subword of a word has not seen the rest of the word (causal model). This is intended (main results use `first` for all models); `last` is needed for `upos_next` and the lens bridge.
- **Truncation.** UD sentences are far below the limits (longest: 159 words in EWT, 310 in ISDT), so `n_truncated_sentences` should be 0. If a word is only partly cut, it keeps the subwords that remain.
- **Random weights.** `--init random` always uses seed 42, so every member gets the same random network.
- **Decoders run without their LM head.** `extract_into` calls `model.base_model`, which returns the same hidden states without computing `[B, T, vocab]` logits (saves GPU memory).
- **Extra `meta.json` fields** (not in the contract): `n_sentences`, `max_length`, `batch_size`, `device`, `seconds`, `final_norm_on_last_layer`.

## 6. Recovering from problems

- **Interrupted run.** A folder without `meta.json` is incomplete; `load_layer` refuses it. Re-run the same command with `--overwrite` for that split.
- **Out of GPU memory.** Lower `--batch-size` (16 or 8). Batches are sorted longest first, so an OOM shows up in the first seconds.
- **Float16 overflow error.** Values outside the float16 range: report it in the issue; do not silence it.
- **mGPT on CPU.** float16 is loaded as bfloat16 on CPU and recorded in `dtype_model`.

## 7. Adding a new model

1. Add an entry to `configs/models.yaml` (registry key, Hugging Face id, kind, L, d, dtype), see CONTRACTS.md section 1.1.
2. Run `PYTHONPATH=src python scripts/01_extract_features.py --model <key> --treebank <tb>`.
3. Nothing else: decoders automatically get BOS and `add_prefix_space=True`.

## 8. Tests

```bash
ruff check src tests scripts app
PYTHONPATH=src pytest -q -m "not slow"   # no downloads; skipped in CI because torch is not installed there
PYTHONPATH=src pytest -q -m slow          # real tokenizers and BERT on EWT test (needs internet and data/)
```

The fast tests build tiny WordPiece and byte-level BPE tokenizers locally (`tests/conftest.py`), so they never download anything.
