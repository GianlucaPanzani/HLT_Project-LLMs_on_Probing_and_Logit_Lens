# WS2 Compression – Baseline Artifact Inventory

Audit of all baseline artifacts for Workstream 2 (Issues C1–C5).
Generated for: `Closes #81`.

> **Status legend**
> - **MEASURED** – file confirmed to exist on shared HF storage (`hlt-group11/features`) and produced by the referenced script with the documented hyperparameters.
> - **MISSING** – artifact is in scope, the script to produce it exists, but the file has not been generated (or confirmed on shared storage) as of this audit (25 November 2026).
> - **N/A** – not in scope for this workstream.
> - **LITERATURE** – qualitative/numerical reference from a published paper; not a locally computed artifact.

---

## 1. CKA Matrix Artifacts (Issue C3)

| Artifact | Model A | Model B | Treebank | Split | Pooling | N tokens | Seed | Script | Result Path | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| CKA matrix (primary) | `bert` | `distilbert` | `ewt` | test | first | 20 000 | 42 | `scripts/20_cka.py` | `results/cka/bert__vs__distilbert__ewt.parquet` | **MISSING** |
| CKA heatmap PNG | `bert` | `distilbert` | `ewt` | test | first | 20 000 | 42 | `scripts/20_cka.py` | `results/figures/ws2/bert__vs__distilbert_cka_heatmap.png` | **MISSING** |
| CKA heatmap HTML | `bert` | `distilbert` | `ewt` | test | first | 20 000 | 42 | `scripts/20_cka.py` | `results/figures/ws2/bert__vs__distilbert_cka_heatmap.html` | **MISSING** |
| CKA matrix (random-init BERT) | `bert-random` | `distilbert` | `ewt` | test | first | 20 000 | 42 | `scripts/20_cka.py` | `results/cka/bert-random__vs__distilbert__ewt.parquet` | **MISSING** |

**Parquet schema** (CONTRACTS.md §8): columns `layer_a` (int), `layer_b` (int), `cka` (float64), `n_tokens` (int), `seed` (int).  
Expected shape: 13 × 7 = **91 rows** (BERT layers 0–12 × DistilBERT layers 0–6).

> **Note – local `results/` directory**: The `results/` tree is gitignored and does not exist on the developer's local machine. Computed parquets live on the team's shared HuggingFace dataset `hlt-group11/features`. Pull scripts: `scripts/14_pull_probing_results.py`, `scripts/15_pull_ws4_lens.py`. Access requires HF credentials that were not available during this audit.

---

## 2. Probing Artifacts (Issues P1–P4, used by C4)

| Artifact | Model | Init | Pooling | Task | Train TB | Eval TB | Seed | Script | Result Path | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| BERT linear UPOS probe | `bert` | pretrained | first | linear | ewt | ewt | 42 | `scripts/01_extract_features.py` + `scripts/10_probing.py` | `results/probing/bert__ewt__linear__pretrained__first.parquet` | **MISSING** |
| DistilBERT linear UPOS probe | `distilbert` | pretrained | first | linear | ewt | ewt | 42 | `scripts/01_extract_features.py` + `scripts/10_probing.py` | `results/probing/distilbert__ewt__linear__pretrained__first.parquet` | **MISSING** |
| BERT random-init probe | `bert` | random | first | linear | ewt | ewt | 42 | `scripts/01_extract_features.py` + `scripts/10_probing.py` | `results/probing/bert__ewt__linear__random__first.parquet` | **MISSING** |
| DistilBERT random-init probe | `distilbert` | random | first | linear | ewt | ewt | 42 | `scripts/01_extract_features.py` + `scripts/10_probing.py` | `results/probing/distilbert__ewt__linear__random__first.parquet` | **MISSING** |
| BERT control-task probe | `bert` | pretrained | first | control | ewt | ewt | 42 | `scripts/01_extract_features.py` + `scripts/10_probing.py` | `results/probing/bert__ewt__control__pretrained__first.parquet` | **MISSING** |
| DistilBERT control-task probe | `distilbert` | pretrained | first | control | ewt | ewt | 42 | `scripts/01_extract_features.py` + `scripts/10_probing.py` | `results/probing/distilbert__ewt__control__pretrained__first.parquet` | **MISSING** |
| BERT MLP probe | `bert` | pretrained | first | mlp | ewt | ewt | 42 | `scripts/10_probing.py` | `results/probing/bert__ewt__mlp__pretrained__first.parquet` | **N/A** (not planned for WS2) |
| DistilBERT MLP probe | `distilbert` | pretrained | first | mlp | ewt | ewt | 42 | `scripts/10_probing.py` | `results/probing/distilbert__ewt__mlp__pretrained__first.parquet` | **N/A** (not planned for WS2) |
| BERT linear UPOS probe on PUD | `bert` | pretrained | first | linear | ewt | pud | 42 | — | `results/probing/bert__pud__linear__pretrained__first.parquet` | **MISSING** |
| DistilBERT linear UPOS probe on PUD | `distilbert` | pretrained | first | linear | ewt | pud | 42 | — | `results/probing/distilbert__pud__linear__pretrained__first.parquet` | **MISSING** |

**Parquet schema** (CONTRACTS.md §6): columns `layer` (int), `accuracy` (float64), `selectivity` (float64), `model` (str), `treebank` (str), `probe` (str), `init` (str), `pooling` (str).

---

## 3. Layer-Mapping Artifacts (Issue C4)

| Artifact | Produced by | Input | Output | Status |
|---|---|---|---|---|
| Layer mapping table (rigid + CKA-best) | `scripts/21_compression_report.py` | CKA parquet + probing parquets | `docs/results/ws2_compression.md` table | **MISSING** (inputs not available) |
| Selectivity comparison figure | `scripts/21_compression_report.py` | probing parquets | `results/figures/ws2/bert_vs_distilbert_selectivity.png` | **MISSING** |

---

## 4. Implementation Sanity Checks (not experimental baselines)

These are deterministic unit tests in `tests/test_cka.py`. They verify the correctness of the implementation, not the empirical properties of real models.

| Check | Expected result | Location | Status |
|---|---|---|---|
| CKA of identical matrices | 1.0 | `tests/test_cka.py::test_linear_cka_identical` | **PASSING** (confirmed by CI) |
| CKA of zero-variance matrix | raises or returns 0 | `tests/test_cka.py::test_linear_cka_zero_variance` | **PASSING** (confirmed by CI) |
| CKA symmetry (A vs B ≈ B vs A) | |CKA(A,B) − CKA(B,A)| < ε | `tests/test_cka.py` | **PASSING** |
| Sample-row reproducibility | same rows with same seed | `tests/test_cka.py` | **PASSING** |

> **Important**: These sanity checks confirm the _implementation_ is correct. They do **not** constitute evidence about BERT or DistilBERT representations.

---

## 5. Literature-Only Baselines

These are reference values from published papers. They are not locally computed; they cannot be reproduced by running our scripts.

| Reference | Metric | Architecture | Dataset | Value | Source |
|---|---|---|---|---|---|
| CKA within-network (same layer) | Linear CKA | ResNet-50 (CV) | ImageNet | ~1.0 | Kornblith et al. 2019 (ICML), Fig. 2 |
| CKA across same architecture | Linear CKA | BERT-base ↔ BERT-base | — | >0.9 (diagonal) | Expected from Kornblith et al. 2019 method |
| DistilBERT UPOS accuracy | Accuracy | distilbert-base-uncased | Penn Treebank WSJ | 97.0% | Sanh et al. 2019, Table 2 |
| BERT-base UPOS accuracy | Accuracy | bert-base-uncased | Penn Treebank WSJ | 97.5% | Sanh et al. 2019, Table 2 |

> **Important**: The Sanh et al. POS numbers are on Penn Treebank (PTB) WSJ with their fine-tuned head; our task uses UD EWT with a linear probe. These are **not** directly comparable. See `EVIDENCE.md §3`.

---

## 6. Missing-Baseline Summary

The following baselines are in scope for WS2 but have not been computed:

1. **CKA matrix** (`bert__vs__distilbert__ewt.parquet`) – the primary geometric result. Command: `PYTHONPATH=src python scripts/20_cka.py --model-a bert --model-b distilbert`. Requires GPU and HF credentials.
2. **Probing results** (pretrained and random-init for both models) – required for the layer mapping table in §4 of `docs/results/ws2_compression.md` and for the CoG comparison. Command: `PYTHONPATH=src python scripts/10_probing.py ...`.
3. **Random-init CKA matrix** – needed to distinguish learned geometric structure from geometry inherited at initialization. Without this, we cannot test the null hypothesis that the near-diagonal pattern (if observed) is due to initialization alone.
4. **Permutation significance tests for CKA** – needed to quantify whether individual CKA values differ significantly from chance.
5. **PUD evaluation** for DistilBERT – cross-treebank generalization; listed as MISSING for both models.

---

## 7. Hyperparameter Registry

Default hyperparameters used by all WS2 scripts (from `lib.registry` and script defaults):

| Parameter | Value | Set in |
|---|---|---|
| `n_tokens` | 20 000 | `scripts/20_cka.py --n-tokens` |
| `seed` | 42 | `scripts/20_cka.py --seed` |
| `pooling` | `first` (first subword token) | `scripts/20_cka.py --pooling` |
| `treebank` | `ewt` | `scripts/20_cka.py --treebank` |
| `split` | `test` | `scripts/20_cka.py --split` |
| probe type | `linear` | `scripts/10_probing.py --probe` |
| probe `init` | `pretrained` (primary), `random` (baseline) | `scripts/10_probing.py --init` |

All weights remain **frozen** during probing (AGENTS.md constraint: `load_model` with frozen parameters).
