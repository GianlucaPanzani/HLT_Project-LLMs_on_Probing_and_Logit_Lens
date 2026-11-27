# Workstream 2: Model Compression & Linguistic Hierarchy

Comparing `bert-base-cased` ($L=12$, teacher) and `distilbert-base-cased` ($L=6$, student) on `ewt` (`test` split, `first`-subword pooling, `linear` probe, `pretrained` weights, seed 42, frozen backbones).

C5 audit (27 November 2026): [baseline inventory and provenance](../../src/ws2_compression/BASELINES.md); [literature evidence and interpretation checks](../../src/ws2_compression/EVIDENCE.md).

## BERT (L=12) vs DistilBERT (L=6)

### CKA Heatmap

![CKA Heatmap](../../results/figures/ws2/bert__vs__distilbert_cka_heatmap.png)

### 1:2 Layer Mapping Table

For each DistilBERT layer *l*, the rigid mapping assumes BERT layer *2l*.
The CKA-best column shows the BERT layer with highest geometric similarity.

| DistilBERT | Distil Acc | Distil Sel | Rigid BERT | Rigid Acc | Rigid Sel | CKA-Best BERT | CKA-Best Acc | CKA-Best Sel | CKA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8726 | 0.1116 | 0 | 0.8723 | 0.1148 | 0 | 0.8723 | 0.1148 | 0.9470 |
| 1 | 0.9318 | 0.2099 | 2 | 0.9356 | 0.2309 | 1 | 0.9062 | 0.1736 | 0.8774 |
| 2 | 0.9438 | 0.2574 | 4 | 0.9486 | 0.2784 | 4 | 0.9486 | 0.2784 | 0.8294 |
| 3 | 0.9477 | 0.2894 | 6 | 0.9519 | 0.3181 | 5 | 0.9507 | 0.2966 | 0.8068 |
| 4 | 0.9464 | 0.3151 | 8 | 0.9512 | 0.3531 | 8 | 0.9512 | 0.3531 | 0.7731 |
| 5 | 0.9418 | 0.3371 | 10 | 0.9437 | 0.3772 | 11 | 0.9433 | 0.3818 | 0.7285 |
| 6 | 0.9339 | 0.3367 | 12 | 0.9366 | 0.3730 | 12 | 0.9366 | 0.3730 | 0.6208 |

### Selectivity Comparison

![Selectivity](../../results/figures/ws2/bert_vs_distilbert_selectivity.png)

- **BERT CoG_norm**: 0.5977
- **DistilBERT CoG_norm**: 0.6263
- **Delta CoG_norm (DistilBERT - BERT)**: +0.0286

### Interpretation

1. **Geometric correspondence (CKA)**: DistilBERT's distillation objective includes a hidden-state cosine embedding loss $\mathcal{L}_{\text{cos}}$ and initializes student layer *l* from teacher layer *2l* (Sanh et al., 2019, §§3.2–3.3), so near-diagonal alignment is expected rather than purely emergent. Empirically, CKA-best matches the 1:2 diagonal at layers 0, 2, 4, and 6 (CKA 0.9470, 0.8294, 0.7731, 0.6208) and shifts by one layer at layers 1 (BERT 1, 0.8774), 3 (BERT 5, 0.8068), and 5 (BERT 11, 0.7285). High CKA reflects geometric similarity under this sample and pooling, not causal or functional equivalence (Kornblith et al., 2019, §6.1).

2. **Functional correspondence (Probing)**: DistilBERT closely tracks BERT's UPOS accuracy (peak 0.9477 at layer 3 vs BERT 0.9519 at layer 6; rigid 1:2 differences within 0.0049), while its selectivity is consistently lower across contextual layers (peak 0.3371 at layer 5 vs BERT 0.3818 at layer 11; rigid 1:2 gaps of -0.0210 to -0.0400).

3. **Is the hierarchy compressed or shifted?** DistilBERT's $\text{CoG}_{\text{norm}}$ (0.6263) is slightly higher than BERT's (0.5977, $\Delta = +0.0286$), indicating that morphosyntactic selectivity concentrates at a similar—slightly deeper—relative depth within the 6-layer student.
