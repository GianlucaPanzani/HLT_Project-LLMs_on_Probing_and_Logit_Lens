# WS2 Compression – Baseline Artifact Inventory

Audit of all baseline artifacts for Workstream 2 (Issues C1–C5), updated with measured data from the team's private Hugging Face repositories (`hlt-group11/features` and `hlt-group11/datasets`).
Generated for: `Closes #81`.

> **Status legend**
> - **MEASURED** – file confirmed to exist on shared HF storage (`hlt-group11/features` or `hlt-group11/datasets`) and locally verified against the referenced scripts and hyperparameters.
> - **MISSING** – artifact is in scope, but has not been generated or uploaded as of this audit (27 November 2026).
> - **N/A** – not in scope for this workstream.
> - **LITERATURE** – qualitative/numerical reference from a published paper; not a locally computed artifact.

---

## 1. Provenance & Shared Storage Snapshots

- **CKA results**: `hlt-group11/datasets` at commit `de35fa8df313bb9497add1b65797f345e9b65ef4` (`results/cka/bert__vs__distilbert__ewt.parquet`, SHA-256 `ceb7aa39d31d6e3b5d393a6ee8bf239620cc8cc8e06ee0322d19145a8b1a9eb4`, matches local file byte-for-byte).
- **Probing results & feature metadata**: `hlt-group11/features` at commit `7587960472da41ccea0ebab41e1249137c3988ff` (pulled via `PYTHONPATH=src python scripts/14_pull_probing_results.py`).
- **Feature metadata** (`bert/{pretrained,random}/first/ewt_test/meta.json` and `distilbert/{pretrained,random}/first/ewt_test/meta.json`): `n_rows=25094`, `n_nan_rows=0`, `n_truncated_sentences=0`, `dtype_model=float32`, `dtype_saved=float16`, `seed=42`, `max_rows=null`, extraction git commit `ce55eaf`.

---

## 2. CKA Matrix Artifacts (Issue C3)

| Artifact | Model A | Model B | Treebank | Split | Pooling | N tokens | Seed | Script | Result Path | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| CKA matrix (primary) | `bert` | `distilbert` | `ewt` | test | first | 20,000 | 42 | `scripts/20_cka.py` | `results/cka/bert__vs__distilbert__ewt.parquet` | **MEASURED** |
| CKA heatmap PNG | `bert` | `distilbert` | `ewt` | test | first | 20,000 | 42 | `scripts/20_cka.py` | `results/figures/ws2/bert__vs__distilbert_cka_heatmap.png` | **MEASURED** |
| CKA heatmap HTML | `bert` | `distilbert` | `ewt` | test | first | 20,000 | 42 | `scripts/20_cka.py` | `results/figures/ws2/bert__vs__distilbert_cka_heatmap.html` | **MEASURED** |
| CKA matrix (random-init) | `bert` (random) | `distilbert` (random/pretrained) | `ewt` | test | first | 20,000 | 42 | — | — | **MISSING** |

**Parquet schema** (`CONTRACTS.md` §8): columns `layer_a` (int), `layer_b` (int), `cka` (float64), `n_tokens` (int), `seed` (int).  
**Measured shape**: 13 × 7 = **91 rows** (BERT layers 0–12 × DistilBERT layers 0–6), `n_tokens=20000`, `seed=42`, CKA range `[0.3606, 0.9470]`.

**Retrieval command** (requires authenticated HF account with `hlt-group11` access):
```bash
hf download hlt-group11/datasets --repo-type dataset \
  --revision de35fa8df313bb9497add1b65797f345e9b65ef4 \
  --include 'results/cka/bert__vs__distilbert__ewt.parquet' --local-dir .
```

---

## 3. Probing Artifacts (Issues P1–P4, used by C4)

Each probing parquet follows `CONTRACTS.md` §6 (`results/probing/{model}__{train_treebank}__{probe}__{init}__{pooling}.parquet`) and stores paired `upos` and `control` tasks (and `eval_treebank` for `ewt` and `pud_en` where evaluated).

| Artifact | Model | Init | Pooling | Probe | Tasks | Eval TB | Rows | Script | Result Path | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| BERT linear pretrained | `bert` | pretrained | first | linear | `upos`, `control`, `upos_next`, `control_next` | `ewt`, `pud_en` | 104 | `scripts/10_run_probing.py` | `results/probing/bert__ewt__linear__pretrained__first.parquet` | **MEASURED** |
| DistilBERT linear pretrained | `distilbert` | pretrained | first | linear | `upos`, `control` | `ewt`, `pud_en` | 28 | `scripts/10_run_probing.py` | `results/probing/distilbert__ewt__linear__pretrained__first.parquet` | **MEASURED** |
| BERT linear random-init | `bert` | random | first | linear | `upos`, `control` | `ewt` | 26 | `scripts/10_run_probing.py` | `results/probing/bert__ewt__linear__random__first.parquet` | **MEASURED** |
| DistilBERT linear random-init | `distilbert` | random | first | linear | `upos`, `control` | `ewt` | 14 | `scripts/10_run_probing.py` | `results/probing/distilbert__ewt__linear__random__first.parquet` | **MEASURED** |
| BERT MLP pretrained | `bert` | pretrained | first | mlp | `upos`, `control` | `ewt`, `pud_en` | 52 | `scripts/10_run_probing.py` | `results/probing/bert__ewt__mlp__pretrained__first.parquet` | **MEASURED** (secondary) |
| DistilBERT MLP pretrained | `distilbert` | pretrained | first | mlp | `upos`, `control` | `ewt`, `pud_en` | 28 | `scripts/10_run_probing.py` | `results/probing/distilbert__ewt__mlp__pretrained__first.parquet` | **MEASURED** (secondary) |
| BERT MLP random-init | `bert` | random | first | mlp | `upos`, `control` | `ewt` | 26 | `scripts/10_run_probing.py` | `results/probing/bert__ewt__mlp__random__first.parquet` | **MEASURED** (secondary) |
| DistilBERT MLP random-init | `distilbert` | random | first | mlp | `upos`, `control` | `ewt` | 14 | `scripts/10_run_probing.py` | `results/probing/distilbert__ewt__mlp__random__first.parquet` | **MEASURED** (secondary) |
| Random-init evaluation on PUD | `bert`, `distilbert` | random | first | linear/mlp | `upos`, `control` | `pud_en` | — | — | — | **MISSING** |

### Measured Probing Summary (seed 42, `first` pooling)

| Model | Probe | Init | Eval TB | Layer 0 Acc | Best UPOS Acc (Layer) | Max Selectivity S (Layer) | `CoG_norm` |
|---|---|---|---|---:|---:|---:|---:|
| `bert` | linear | pretrained | `ewt` | 0.8723 | 0.9519 (6) | 0.3818 (11) | 0.5977 |
| `distilbert` | linear | pretrained | `ewt` | 0.8726 | 0.9477 (3) | 0.3371 (5) | 0.6263 |
| `bert` | linear | pretrained | `pud_en` | 0.8867 | 0.9604 (6) | 0.4230 (11) | 0.5794 |
| `distilbert` | linear | pretrained | `pud_en` | 0.8863 | 0.9587 (3) | 0.4006 (5) | 0.6147 |
| `bert` | linear | random | `ewt` | 0.7445 | 0.7445 (0) | 0.0173 (11) | — |
| `distilbert` | linear | random | `ewt` | 0.7455 | 0.7455 (0) | 0.0105 (6) | — |
| `bert` | mlp | pretrained | `ewt` | 0.8750 | 0.9580 (8) | 0.2579 (11) | — |
| `distilbert` | mlp | pretrained | `ewt` | 0.8707 | 0.9521 (4) | 0.2214 (5) | — |
| `bert` | mlp | random | `ewt` | 0.8014 | 0.8014 (0) | -0.0058 (11) | — |
| `distilbert` | mlp | random | `ewt` | 0.8000 | 0.8003 (1) | -0.0136 (3) | — |

---

## 4. Layer-Mapping Artifacts (Issue C4)

| Artifact | Produced by | Input | Output | Status |
|---|---|---|---|---|
| Layer mapping table (rigid + CKA-best) | `scripts/21_compression_report.py` | `bert__vs__distilbert__ewt.parquet` + probing parquets | `docs/results/ws2_compression.md` | **MEASURED** |
| Selectivity comparison figure | `scripts/21_compression_report.py` | probing parquets | `results/figures/ws2/bert_vs_distilbert_selectivity.png` | **MEASURED** |

---

## 5. Implementation Sanity Checks (not experimental baselines)

These are deterministic unit tests in `tests/test_cka.py`. They verify the correctness of the implementation, not the empirical properties of real models.

| Check | Expected result | Location | Status |
|---|---|---|---|
| CKA of identical matrices | 1.0 | `tests/test_cka.py::test_linear_cka_identical` | **PASSING** |
| CKA of zero-variance matrix | raises or returns 0 | `tests/test_cka.py::test_linear_cka_zero_variance` | **PASSING** |
| CKA symmetry (A vs B ≈ B vs A) | \|CKA(A,B) − CKA(B,A)\| < ε | `tests/test_cka.py` | **PASSING** |
| Sample-row reproducibility | same rows with same seed | `tests/test_cka.py` | **PASSING** |

> **Important**: These sanity checks confirm the _implementation_ is correct. They do **not** constitute evidence about BERT or DistilBERT representations.

---

## 6. Literature-Only Baselines

These are reference values from published papers. They are not locally computed; they cannot be reproduced by running our scripts.

| Reference | Metric | Architecture | Dataset | Value | Source |
|---|---|---|---|---|---|
| CKA within-network (same layer) | Linear CKA | ResNet-50 (CV) | ImageNet | ~1.0 | Kornblith et al. 2019 (ICML), Fig. 2 |
| CKA across same architecture | Linear CKA | BERT-base ↔ BERT-base | — | >0.9 (diagonal) | Expected from Kornblith et al. 2019 method |
| DistilBERT POS accuracy | Accuracy | `distilbert-base-uncased` | Penn Treebank WSJ | 97.0% | Sanh et al. 2019, Table 2 |
| BERT-base POS accuracy | Accuracy | `bert-base-uncased` | Penn Treebank WSJ | 97.5% | Sanh et al. 2019, Table 2 |

> **Important**: The Sanh et al. POS numbers are on Penn Treebank (PTB) WSJ with a fine-tuned head; our task uses UD EWT with a frozen-weight linear probe. These are **not** directly comparable. See `EVIDENCE.md` §1–3.

---

## 7. Remaining Missing-Baseline Summary

The primary CKA matrix, pretrained/random probing curves, and PUD pretrained evaluations are all **MEASURED**. Only the following secondary baselines remain uncomputed:

1. **Random-init CKA matrix** (`bert` random vs `distilbert` random/pretrained): random-init features exist in `hlt-group11/features`, but `ws2_compression.cka.cka_matrix()` currently hardcodes `init="pretrained"`, so no random-init CKA parquet has been computed. Without this, we cannot empirically separate initialization-inherited geometry from distillation-trained geometry.
2. **Permutation significance tests for CKA**: needed to quantify whether individual off-diagonal CKA values differ significantly from chance (Kornblith et al. 2019, §4.2).
3. **Random-backbone evaluation on PUD**: `bert` and `distilbert` random-init probes were only evaluated on `ewt`, not `pud_en`.

---

## 8. Hyperparameter Registry

Default hyperparameters used by all WS2 scripts (from `lib.registry` and script defaults):

| Parameter | Value | Set in |
|---|---|---|
| `n_tokens` | 20,000 | `scripts/20_cka.py --n-tokens` |
| `seed` | 42 | `scripts/20_cka.py --seed` |
| `pooling` | `first` (first subword token) | `scripts/20_cka.py --pooling` |
| `treebank` | `ewt` | `scripts/20_cka.py --treebank` |
| `split` | `test` | `scripts/20_cka.py --split` |
| `probe` | `linear` | `scripts/21_compression_report.py --probe` |
| `init` | `pretrained` (primary), `random` (baseline) | `scripts/21_compression_report.py --init` |

All weights remain **frozen** during probing (`AGENTS.md` constraint: `load_model` with frozen parameters).
