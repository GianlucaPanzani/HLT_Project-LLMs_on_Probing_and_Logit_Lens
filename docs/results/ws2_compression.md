# Workstream 2: Model Compression & Linguistic Hierarchy

## BERT (L=12) vs DistilBERT (L=6)

### CKA Heatmap

![CKA Heatmap](../../results/figures/ws2/bert__vs__distilbert_cka_heatmap.png)

### 1:2 Layer Mapping Table

For each DistilBERT layer *l*, the rigid mapping assumes BERT layer *2l*.
The CKA-best column shows the BERT layer with highest geometric similarity.

| DistilBERT | Rigid BERT | Rigid Acc | Rigid Sel | CKA-Best BERT | CKA-Best Acc | CKA-Best Sel | CKA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.0000 | 0.8723 | 0.1148 | 0.0000 | 0.8723 | 0.1148 | 0.9470 |
| 1.0000 | 2.0000 | 0.9356 | 0.2309 | 1.0000 | 0.9062 | 0.1736 | 0.8774 |
| 2.0000 | 4.0000 | 0.9486 | 0.2784 | 4.0000 | 0.9486 | 0.2784 | 0.8294 |
| 3.0000 | 6.0000 | 0.9519 | 0.3181 | 5.0000 | 0.9507 | 0.2966 | 0.8068 |
| 4.0000 | 8.0000 | 0.9512 | 0.3531 | 8.0000 | 0.9512 | 0.3531 | 0.7731 |
| 5.0000 | 10.0000 | 0.9437 | 0.3772 | 11.0000 | 0.9433 | 0.3818 | 0.7285 |
| 6.0000 | 12.0000 | 0.9366 | 0.3730 | 12.0000 | 0.9366 | 0.3730 | 0.6208 |

### Selectivity Comparison

![Selectivity](../../results/figures/ws2/bert_vs_distilbert_selectivity.png)

- **BERT CoG_norm**: 0.5977
- **DistilBERT CoG_norm**: 0.6263

### Interpretation

1. **Geometric correspondence (CKA)**: The heatmap shows whether the
   assumed 1:2 layer mapping (DistilBERT layer *l* ↔ BERT layer *2l*)
   holds geometrically. Cells on the 1:2 diagonal should have the highest
   CKA in their column if the mapping is accurate.

2. **Functional correspondence (Probing)**: The accuracy and selectivity
   columns quantify whether paired layers encode the same amount of UPOS
   information. Similar selectivity under the mapping suggests the
   linguistic hierarchy is preserved during distillation.

3. **Is the hierarchy compressed or shifted?** The CoG_norm comparison
   indicates whether DistilBERT distributes morphosyntactic information
   across its depth in the same relative proportions as BERT.
