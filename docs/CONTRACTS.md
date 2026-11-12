# Contracts: registries, file formats and Python interfaces

This file is the single source of truth shared by all tasks. If an issue and this file disagree, this file wins; fix the issue.
Changing anything here requires a PR that touches only this file, approved by one other member, and a note in every affected issue.

All paths are relative to the repository root. `data/`, `features/`, `results/` and `checkpoints/` are gitignored (large files live on the team's shared storage).
Code imports shared modules as `from lib.<module> import ...` with `PYTHONPATH=src`.

## 1. Registries

### 1.1 Models (`configs/models.yaml`, read with `lib.registry.get_model_spec`)

| Key | Hugging Face id | kind | lang | L | d | dtype |
| :-- | :-- | :-- | :-- | :-: | :-: | :-- |
| `bert` | `bert-base-cased` | encoder | en | 12 | 768 | float32 |
| `distilbert` | `distilbert-base-cased` | encoder | en | 6 | 768 | float32 |
| `gpt2` | `gpt2` | decoder | en | 12 | 768 | float32 |
| `bert_it` | `dbmdz/bert-base-italian-cased` | encoder | it | 12 | 768 | float32 |
| `geppetto` | `LorenzoDeMattei/GePpeTto` | decoder | it | 12 | 768 | float32 |
| `mbert` | `bert-base-multilingual-cased` | encoder | multi | 12 | 768 | float32 |
| `mgpt` | `ai-forever/mGPT` | decoder | multi | 24 | 2048 | float16 |

- `kind` is the value passed to `lib.data.load_model(kind=...)`: `encoder` loads `AutoModel`, `decoder` loads `AutoModelForCausalLM`.
- Decoders (GPT-2 family BPE) are always loaded with `tokenizer_kwargs={"add_prefix_space": True}`.
- `L` counts Transformer blocks. `hidden_states` returned by Hugging Face has `L + 1` entries: index 0 is the embedding output, index `l` is the output of block `l`.

### 1.2 Treebanks (`configs/datasets.yaml`)

All from Universal Dependencies release tag **`r2.19`**, downloaded to `data/raw/{key}/`.

| Key | Repository | Files | lang | Splits |
| :-- | :-- | :-- | :-- | :-- |
| `ewt` | `UniversalDependencies/UD_English-EWT` | `en_ewt-ud-{split}.conllu` | en | train, dev, test |
| `isdt` | `UniversalDependencies/UD_Italian-ISDT` | `it_isdt-ud-{split}.conllu` | it | train, dev, test |
| `pud_en` | `UniversalDependencies/UD_English-PUD` | `en_pud-ud-test.conllu` | en | test |
| `pud_it` | `UniversalDependencies/UD_Italian-PUD` | `it_pud-ud-test.conllu` | it | test |

Raw URL pattern: `https://raw.githubusercontent.com/UniversalDependencies/{repo}/r2.19/{file}`.
PUD has only a test split: probes are always trained on `ewt` (for English) or `isdt` (for Italian) and PUD is an extra evaluation set. `lib.data.load_ud` must be called with `splits=("test",)` for PUD.
"Train treebank of" a treebank: `ewt -> ewt`, `pud_en -> ewt`, `isdt -> isdt`, `pud_it -> isdt`.

WikiText: Hugging Face dataset `Salesforce/wikitext`, config `wikitext-103-raw-v1`, split `test` (dataset key `wikitext`). Used by the lens only.

### 1.3 UPOS tags
The 17 UD tags, in this order (used as class indices 0..16 everywhere):
`ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X`. Exposed as `lib.data.UPOS_TAGS`.

## 2. Word table (`data/processed/{treebank}_{split}.parquet`)
One row per UD syntactic word. Multiword token ranges (for example Italian `della` = `di` + `la`) are not rows; their component words are. Empty nodes (ids like `8.1`) are skipped. Row order: sentences in file order, words in sentence order. **Every feature file is aligned to this row order.**

| Column | Type | Note |
| :-- | :-- | :-- |
| `sent_id` | str | from the CoNLL-U metadata |
| `word_idx` | int | 0-based position of the word in the sentence |
| `form` | str | word form, case preserved |
| `upos` | str | one of the 17 tags |
| `next_upos` | str or null | UPOS of the next word in the same sentence; null for the last word |
| `lang` | str | `en` or `it` |

## 3. Alignment rules (how a word gets one vector)
- Tokenize pre-split words with `is_split_into_words=True` and map subwords to words with `encoding.word_ids()`. Special tokens have `word_id = None` and are never used as word vectors.
- Decoders: prepend `tokenizer.bos_token` (fall back to `eos_token`) to every sentence so that the first word has a context position; its hidden state is discarded.
- Pooling modes:
  - `first`: hidden state of the first subword of the word (default for all main results).
  - `mean`: mean over the word's subwords (ablation).
  - `last`: hidden state of the last subword of the word. Required for decoders on the `upos_next` task and for the lens bridge, because that position is the one that predicts the next word.
- Truncation: sequences longer than the model maximum (512 for BERT-family, 1024 for GPT-2, 2048 for mGPT) are truncated. Words that lose all their subwords, and words that produce zero subwords, get a **row of NaN** so that row counts still match the word table. Counts are recorded in `meta.json`. Consumers drop NaN rows (and the same rows from labels) before fitting.

## 4. Features (`features/{model}/{init}/{pooling}/{treebank}_{split}/`)
- `init`: `pretrained` or `random` (random weights from `load_model(random_weights=True, seed=42)`).
- One file per layer: `layer_{l:02d}.npy`, a float16 NumPy array of shape `[N, d]`, where `N` is the number of rows of the word table. Written with `numpy.lib.format.open_memmap` so that large models never hold all layers in RAM; read with `np.load(path, mmap_mode="r")`.
- `meta.json` in the same folder:

```json
{"model": "gpt2", "hf_id": "gpt2", "init": "pretrained", "pooling": "first",
 "treebank": "ewt", "split": "train", "num_layers": 12, "hidden_size": 768,
 "n_rows": 204576, "n_nan_rows": 3, "n_truncated_sentences": 1,
 "dtype_model": "float32", "dtype_saved": "float16", "seed": 42,
 "max_rows": null, "git_commit": "8fa6267", "created": "2026-11-15T10:00:00"}
```

- `max_rows` is non-null only if extraction was capped (allowed only for mGPT train, see issue X3); in that case the features cover the first `max_rows` rows of the word table.

Pooling needed per model (minimum set):

| Model | init | pooling | treebanks |
| :-- | :-- | :-- | :-- |
| bert, distilbert | pretrained, random | first | ewt (train/dev/test), pud_en |
| gpt2 | pretrained, random | first, last | ewt (train/dev/test), pud_en |
| bert_it | pretrained | first | isdt (train/dev/test), pud_it |
| geppetto | pretrained | first, last | isdt (train/dev/test), pud_it |
| mbert | pretrained | first | ewt, isdt (train/dev/test), pud_en, pud_it |
| mgpt | pretrained | first | ewt, isdt (train/dev/test), pud_en, pud_it |

`mean` pooling for bert and gpt2 on ewt is a nice-to-have ablation.

## 5. Control labels (`data/processed/control_{train_treebank}.json`)
Built from the train split of `ewt` or `isdt` (Hewitt and Liang, 2019):

```json
{"treebank": "ewt", "seed": 42,
 "distribution": {"NOUN": 0.17, "...": 0.0},
 "mapping": {"the": "ADV", "The": "PRON", "...": "..."}}
```

- `distribution`: empirical UPOS frequencies of the train split (word tokens, not types).
- `mapping`: every word type of the train split (case-sensitive form) gets one label sampled from `distribution` with `numpy.random.default_rng(seed)`, iterating types in sorted order.
- Word types not in `mapping` (dev/test/PUD only): label sampled from `distribution` with `default_rng(int(hashlib.md5(f"{seed}:{form}".encode()).hexdigest()[:8], 16))`. Never use Python's `hash()` (it changes between runs).
- Control label for `upos_next`: the control label of the next word form in the same sentence (null for the last word).

## 6. Probing results (`results/probing/{model}__{train_treebank}__{probe}__{init}__{pooling}.parquet`)
One row per (eval treebank, task, layer, seed).

| Column | Type | Note |
| :-- | :-- | :-- |
| `model` | str | registry key |
| `train_treebank` | str | `ewt` or `isdt` |
| `eval_treebank` | str | `ewt`, `isdt`, `pud_en` or `pud_it` (test split) |
| `lang` | str | language of the eval treebank |
| `kind` | str | `encoder` or `decoder` |
| `probe` | str | `linear` or `mlp` |
| `task` | str | `upos`, `control`, `upos_next` or `control_next` |
| `init` | str | `pretrained` or `random` |
| `pooling` | str | `first`, `mean` or `last` |
| `layer` | int | 0..L |
| `L` | int | number of layers |
| `tau` | float | `layer / L` |
| `acc` | float | accuracy on the eval treebank test split |
| `n_train` | int | number of training rows actually used |
| `n_eval` | int | number of eval rows actually used |
| `hparams` | str | JSON of the selected hyperparameters (for example `{"C": 1.0}`) |
| `seed` | int | |

Selectivity is not stored: compute it with `lib.metrics.selectivity` from the `upos`/`control` (or `upos_next`/`control_next`) rows.

## 7. Summary (`results/summary/cog.parquet`)
Produced by `lib.metrics.summarize_probing` from all probing results. One row per (model, train_treebank, eval_treebank, probe, init, pooling, task pair).
Columns: `model, train_treebank, eval_treebank, lang, probe, init, pooling, task` (`upos` or `upos_next`)`, L, cog, cog_norm, peak_layer, peak_tau, max_selectivity`.

## 8. CKA (`results/cka/{model_a}__vs__{model_b}__{treebank}.parquet`)
Long format, columns `layer_a, layer_b, cka, n_tokens, seed`. For `bert` vs `distilbert` on `ewt` test: 13 x 7 = 91 rows.

## 9. Lens conventions and results

### 9.1 Positions and targets
- Lenses work on subword positions of decoder models (`gpt2`, `geppetto`). The prediction at position `t` is compared with the actual token at `t + 1`. The last position of a sequence has no target and is skipped.
- On UD data (PUD) the input is the pre-split words from the word table, encoded with `lib.alignment.encode_words(..., add_bos=True)` exactly like feature extraction, so `word_ids` link positions to UD words. For the UPOS bridge, the position used for word `w` is the **last subword of `w`**, and the gold label is `next_upos` of `w`.
- On WikiText the text is concatenated and cut into non-overlapping windows of 256 tokens; use at most 200 windows (about 50k positions).

### 9.2 Lens definitions (decoder with final norm `ln_f` = `model.transformer.ln_f` and unembedding `W_U` = `model.lm_head`)
- `logit`: `logits_l = W_U(ln_f(h_l))` for `l < L`. At `l = L` use `W_U(h_L)` directly, because Hugging Face already applies `ln_f` to the last hidden state. Sanity check: `logits_L` equals `model(...).logits`.
- `tuned`: `logits_l = W_U(ln_f(h_l + A_l h_l + b_l))` with `A_l, b_l` initialised to zero; identity at `l = L`.
- `jacobian` (team definition, training-free): `J_l = mean over sampled positions t of d r_L(t) / d h_l(t)` (a `d x d` matrix, same position only, other positions held fixed), then `logits_l = W_U(ln_f(J_l h_l))`. Here `r_L` is the output of the last block **before** `ln_f`, captured with a forward hook on `model.transformer.h[-1]` (Hugging Face's `hidden_states[L]` is after `ln_f`, so it cannot be used). `J_L` is the identity. Estimated on the lens calibration split (ISDT train for GePpeTto, WikiText train for GPT-2), never on test.

### 9.3 Aggregate results (`results/lens/{model}__{dataset}.parquet`)
`dataset` is `pud_en`, `pud_it` or `wikitext`. One row per (lens, layer).
Columns: `model, dataset, lens, layer, L, tau, top1_acc, entropy_mean, kl_mean, upos_top1_acc, n_positions`. `upos_top1_acc` is null for WikiText.
Entropy and KL use natural log (nats). KL is `KL(P_L || P_l)`.

### 9.4 Per-token results (`results/lens/tokens/{model}__{dataset}.parquet`)
For the demo. One row per (sequence, position, lens, layer). Columns: `seq_id, pos, token, target_token, lens, layer, top1_token, top1_prob, entropy, kl`. Limit to 100 sequences per dataset.

### 9.5 Checkpoints
`checkpoints/tuned_lens/{model}.pt`: `state_dict` of `ws4_lens.tuned_lens.TunedLensTranslators`, plus `{"model": key, "train_data": "isdt_train", "epochs": ..., "seed": 42}` in `checkpoints/tuned_lens/{model}.json`.
`checkpoints/jacobian_lens/{model}.pt`: tensor `[L + 1, d, d]` (identity at `L`).

## 10. Python interfaces
Signatures that other tasks depend on. Owners may add private helpers but must not change these without a contract PR. NumPy arrays are `np.ndarray`, tensors are `torch.Tensor`.

### `lib/registry.py` (G0, Gianluca)
```python
def get_model_spec(key: str) -> dict          # keys: key, hf_id, kind, lang, num_layers, hidden_size, dtype
def list_models() -> list[str]
def get_treebank_spec(key: str) -> dict       # keys: key, repo, file_pattern, lang, splits, train_treebank
```

### `lib/data.py` (existing, Gianluca)
```python
UPOS_TAGS: tuple[str, ...]                    # section 1.3 order
def load_ud(treebank_dir, splits=("train", "dev", "test"), max_sentences=None) -> dict[str, list[dict]]
def load_model(checkpoint, *, kind, random_weights=False, seed=42, device=None,
               revision="main", tokenizer_kwargs=None, dtype=None) -> tuple[model, tokenizer, dict]
```

### `lib/paths.py` (D2, Filippo)
The repository root is found from the location of `paths.py`. The environment variable `HLT_FEATURES_DIR`, if set, replaces `features/` (so features can live on an external disk or shared storage). Functions only build paths; they never create or delete anything.
```python
def raw_dir(treebank: str) -> Path
def word_table_path(treebank: str, split: str) -> Path
def control_path(train_treebank: str) -> Path
def feature_dir(model: str, init: str, pooling: str, treebank: str, split: str) -> Path
def layer_path(model: str, init: str, pooling: str, treebank: str, split: str, layer: int) -> Path
def probing_result_path(model: str, train_treebank: str, probe: str, init: str, pooling: str) -> Path
def cka_result_path(model_a: str, model_b: str, treebank: str) -> Path
def lens_result_path(model: str, dataset: str, per_token: bool = False) -> Path
```

### `lib/word_table.py` (D2, Filippo)
```python
def build_word_table(sentences: list[dict], lang: str) -> pd.DataFrame       # input: one split from load_ud
def load_word_table(treebank: str, split: str) -> pd.DataFrame
def table_to_sentences(table: pd.DataFrame) -> list[tuple[str, list[str]]]   # (sent_id, words) in row order
```

### `lib/alignment.py` (E1, Marcin)
```python
def encode_words(tokenizer, sentences: list[list[str]], *, add_bos: bool, max_length: int | None = None) -> tuple[BatchEncoding, list[list[int | None]]]
def pool_words(hidden: torch.Tensor, word_ids: list[list[int | None]], n_words: list[int], pooling: str) -> list[torch.Tensor]
    # hidden: [B, T, d] for one layer; returns per sentence a [n_words_i, d] tensor, NaN rows for missing words
```

### `lib/features.py` (E2, Marcin)
```python
def extract_features(model_key: str, treebank: str, split: str, *, init: str = "pretrained", pooling: str = "first",
                     batch_size: int = 32, max_rows: int | None = None, overwrite: bool = False) -> Path
def load_layer(model: str, init: str, pooling: str, treebank: str, split: str, layer: int) -> np.ndarray   # mmap, float16 [N, d]
def load_meta(model: str, init: str, pooling: str, treebank: str, split: str) -> dict
```

### `lib/control_tasks.py` (E3, Marcin)
```python
def build_control_mapping(train_table: pd.DataFrame, seed: int = 42) -> dict      # the JSON object of section 5
def control_labels(table: pd.DataFrame, control: dict) -> pd.Series             # aligned with table rows
def control_next_labels(table: pd.DataFrame, control: dict) -> pd.Series        # null for last word
def load_control(train_treebank: str) -> dict
```

### `lib/metrics.py` (P1, Gianluca)
```python
def selectivity(acc_task: np.ndarray, acc_control: np.ndarray) -> np.ndarray
def relative_depth(layer: int | np.ndarray, num_layers: int) -> float | np.ndarray
def center_of_gravity(sel: np.ndarray, *, normalize: bool = True, clip_negative: bool = True) -> float
    # sel has L + 1 entries (layers 0..L); only layers 1..L are summed; returns nan if the clipped sum is 0
def peak_layer(sel: np.ndarray) -> int
def summarize_probing(results: pd.DataFrame) -> pd.DataFrame                    # section 7
```

### `ws1_probing/probes.py` (P2, P3, Gianluca)
```python
class Probe(Protocol):
    def fit(self, X_train, y_train, X_dev, y_dev) -> "Probe": ...
    def predict(self, X) -> np.ndarray: ...
    hparams: dict
class LinearProbe: ...   # __init__(self, Cs=(0.01, 0.1, 1.0, 10.0), max_iter=1000, seed=42)
class MLPProbe: ...      # __init__(self, hidden=256, lr=1e-3, weight_decay=0.0, batch_size=512, max_epochs=50, patience=5, seed=42, device=None)
```
Labels are integer class indices from `UPOS_TAGS`. Inputs are float32 arrays (cast from the float16 features).

### `ws1_probing/runner.py` (P4, P5, Gianluca)
```python
def run_probing(model: str, train_treebank: str, *, probe: str, init: str, pooling: str,
                tasks: tuple[str, ...] = ("upos", "control"), eval_treebanks: tuple[str, ...] | None = None,
                layers: list[int] | None = None, max_train_rows: int | None = 50000, seed: int = 42) -> pd.DataFrame
```
Writes `probing_result_path(...)` and returns the same frame. `eval_treebanks=None` means the train treebank plus its PUD.

### `ws2_compression/cka.py` (C1, C2, C3, Filippo)
```python
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float           # float64 internally
def sample_rows(valid_mask: np.ndarray, n: int, seed: int = 42) -> np.ndarray
def cka_matrix(model_a: str, model_b: str, treebank: str = "ewt", split: str = "test", *, pooling: str = "first",
               n_tokens: int = 20000, seed: int = 42) -> pd.DataFrame
```

### `ws4_lens/` (L1 to L6, Matteo)
```python
# metrics.py
def entropy(logits: torch.Tensor) -> torch.Tensor                                  # [..., V] -> [...]
def kl_to_final(logits_l: torch.Tensor, logits_L: torch.Tensor) -> torch.Tensor   # KL(P_L || P_l), [..., V] -> [...]
def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float
# logit_lens.py
def run_model(model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[torch.Tensor]
    # returns hidden_states as L + 1 tensors [B, T, d]; inputs come from encode_words (UD) or the WikiText windows
def logit_lens(model, hidden_states: list[torch.Tensor]) -> list[torch.Tensor]    # L + 1 tensors [B, T, V]
# tuned_lens.py
class TunedLensTranslators(torch.nn.Module): ...                                    # __init__(self, num_layers, hidden_size)
def tuned_lens(model, translators, hidden_states) -> list[torch.Tensor]
# jacobian_lens.py
def estimate_jacobians(model, batches: Iterable[tuple[torch.Tensor, torch.Tensor]], *, n_positions: int = 512, seed: int = 42) -> torch.Tensor  # [L + 1, d, d]; batches of (input_ids, attention_mask)
def jacobian_lens(model, jacobians: torch.Tensor, hidden_states) -> list[torch.Tensor]
# upos_bridge.py
def build_upos_lexicon(train_table: pd.DataFrame) -> dict[str, str]
def token_to_upos(token: str, lexicon: dict[str, str]) -> str
```

## 11. Conventions
- Seed 42 everywhere unless a task says otherwise; always stored in outputs.
- Test split for reported numbers, dev for hyperparameters, train for fitting. Never tune on test or PUD.
- Every output is reproducible from `configs/` plus a script in `scripts/`. Scripts take `--model`, `--treebank` and similar flags and print where they wrote files.
- Plots go to `results/figures/{workstream}/{name}.png` (and `.html` for plotly).
