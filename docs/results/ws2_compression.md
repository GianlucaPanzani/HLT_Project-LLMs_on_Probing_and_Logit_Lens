# Workstream 2: Model Compression & Linguistic Hierarchy (Issues C1–C5)

Investigating how knowledge distillation affects internal representations: comparing `bert-base-cased` ($L=12$) and `distilbert-base-cased` ($L=6$) on Universal Dependencies part-of-speech tagging (`UD_English-EWT`).

> **Baseline audit and evidence register (Issue C5)**
> See [`../../src/ws2_compression/BASELINES.md`](../../src/ws2_compression/BASELINES.md) for the full artifact inventory (what is measured, what is missing, and what is N/A).
> See [`../../src/ws2_compression/EVIDENCE.md`](../../src/ws2_compression/EVIDENCE.md) for literature comparisons, interpretation checks, and evidence-qualified findings for R1.

---

## 1. Geometric Correspondence: Linear CKA (Issue C3)

Centered Kernel Alignment (CKA) measures representation similarity between BERT layers (0..12) and DistilBERT layers (0..6) across aligned test tokens from the EWT test split.

![CKA Heatmap](../../results/figures/ws2/bert__vs__distilbert_cka_heatmap.png)

> ⚪ **Figure not yet available** — `results/figures/ws2/bert__vs__distilbert_cka_heatmap.png` is MISSING (the CKA parquet has not been computed yet). Run `PYTHONPATH=src python scripts/20_cka.py --model-a bert --model-b distilbert` to generate it. See [`../../src/ws2_compression/BASELINES.md §1`](../../src/ws2_compression/BASELINES.md).

### Summary of Geometric Alignment
- If distillation preserved intermediate geometry linearly, the highest CKA value in DistilBERT column $l$ would fall exactly at BERT row $2l$ (the 1:2 diagonal marked in red).
- Off-diagonal peaks reveal whether DistilBERT compresses layers uniformly or if lower/higher layers shift their representational role.

---

## 2. Functional Correspondence: 1:2 Layer Mapping (Issue C4)

Comparing UPOS probing accuracy, selectivity, and Center of Gravity ($\text{CoG}_{\text{norm}}$) under the rigid 1:2 mapping ($l \leftrightarrow 2l$) and the CKA-best mapping.

### Layer Mapping Comparison Table

| DistilBERT Layer | Rigid BERT Layer | DistilBERT Acc | Rigid BERT Acc | Rigid Acc Diff | DistilBERT Sel | Rigid BERT Sel | Rigid Sel Diff | CKA-Best BERT | CKA-Best CKA | CKA-Best Sel |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0 | — | — | — | — | — | — | ⚪ | ⚪ | ⚪ |
| 1 | 2 | — | — | — | — | — | — | ⚪ | ⚪ | ⚪ |
| 2 | 4 | — | — | — | — | — | — | ⚪ | ⚪ | ⚪ |
| 3 | 6 | — | — | — | — | — | — | ⚪ | ⚪ | ⚪ |
| 4 | 8 | — | — | — | — | — | — | ⚪ | ⚪ | ⚪ |
| 5 | 10 | — | — | — | — | — | — | ⚪ | ⚪ | ⚪ |
| 6 | 12 | — | — | — | — | — | — | ⚪ | ⚪ | ⚪ |

*Note: Table values are populated automatically upon running `scripts/21_compression_report.py` once P4 probing results are generated.*

> ⚪ **All entries are placeholders** — probing results and CKA matrix are MISSING. The placeholder `1.000` for layer-0 CKA that appeared in earlier versions was a synthetic default, **not** a measurement; it has been removed. See [`../../src/ws2_compression/BASELINES.md`](../../src/ws2_compression/BASELINES.md) for the full status of each required artifact.

---

## 3. Linguistic Hierarchy & Depth Comparison

Layer-wise selectivity trajectories plotted against relative depth $\tau = l / L \in [0, 1]$:

![Selectivity Comparison](../../results/figures/ws2/bert_vs_distilbert_selectivity.png)

> ⚪ **Figure not yet available** — requires probing parquets (MISSING). See [`../../src/ws2_compression/BASELINES.md §2`](../../src/ws2_compression/BASELINES.md).

- **BERT $\text{CoG}_{\text{norm}}$**: (computed from linear UPOS probe)
- **DistilBERT $\text{CoG}_{\text{norm}}$**: (computed from linear UPOS probe)
- **$\Delta \text{CoG}_{\text{norm}}$**: $\text{CoG}_{\text{norm}}(\text{DistilBERT}) - \text{CoG}_{\text{norm}}(\text{BERT})$

---

## 4. Interpretation & Research Questions

1. **Does the 1:2 layer mapping hold geometrically (via CKA)?**
   - DistilBERT is trained with a **multi-term** distillation loss: soft cross-entropy on softmax-temperature teacher logits **plus** a cosine embedding loss $\mathcal{L}_{cos}$ on hidden states, plus hard cross-entropy on MLM labels (Sanh et al. 2019, §3.2). It is therefore **incorrect** to say "distillation only matches output logits". Furthermore, DistilBERT layer $l$ is initialised from BERT layer $2l$ (Sanh et al. 2019, §3.3), so near-diagonal CKA is expected at initialisation. CKA evaluates whether this alignment persists — or strengthens — through training.

2. **Does the mapping hold functionally (via UPOS probing)?**
   - Comparing probing accuracy and selectivity reveals whether DistilBERT layer $l$ decodes morphosyntax with comparable strength to BERT layer $2l$.

3. **Is the UPOS hierarchy compressed or shifted?**
   - Comparing $\text{CoG}_{\text{norm}}$ shows whether syntactic information concentrates at the same relative depth ($\tau$) in DistilBERT as in BERT, or whether compression forces morphosyntax to peak earlier to free upper layers for task-level distillation.
