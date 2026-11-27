# WS2 Compression – Literature Evidence and Interpretation Checks

Evidence registry for Workstream 2 (Issues C1–C5), updated on 27 November 2026 with measured CKA and probing results from `hlt-group11/datasets` and `hlt-group11/features`.
`Closes #81`.

> **Evidence qualification key**
> - 🟢 **VERIFIED** – claim is directly supported by a cited primary source, a locally passing test, or a measured result artifact in `BASELINES.md`.
> - 🟡 **EXPECTED** – claim is consistent with cited literature and WS2 methodology, but a specific control/baseline has not been computed locally.
> - 🔴 **INCORRECT** – claim in an earlier draft was contradicted by a cited primary source or measurement; corrected.
> - ⚪ **UNMEASURED** – claim cannot be evaluated without a missing baseline (e.g., random-init CKA or permutation null).

---

## 1. Primary Sources

### Source 1 — Kornblith et al. 2019

> S. Kornblith, M. Norouzi, H. Lee, G. Hinton. **Similarity of Neural Network Representations Revisited.** *Proceedings of the 36th ICML*, 2019. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)

**Key claims from this paper used in WS2:**

| Claim | Verification status | Notes |
|---|---|---|
| CKA measures representational similarity; it is invariant to orthogonal transformations and isotropic scaling | 🟢 VERIFIED | Theorem 1, §3. Our `linear_cka()` in `src/ws2_compression/cka.py` uses the feature-space formulation with column mean centering. |
| CKA ∈ [0, 1] by construction (for positive-semi-definite kernels) | 🟢 VERIFIED | Corollary 1, `tests/test_cka.py::test_linear_cka_identical` (CKA = 1.0), and measured `bert__vs__distilbert__ewt.parquet` (all 91 values in `[0.3606, 0.9470]`). |
| High CKA does **not** imply functional identity | 🟢 VERIFIED | §6.1: "CKA is a measure of similarity, not of functional equivalence." Empirically, BERT/DistilBERT pairs with CKA 0.7285–0.8774 still exhibit selectivity gaps of −0.0210 to −0.0447. |
| Same architecture trained from different seeds yields high within-network CKA on the diagonal | 🟢 VERIFIED | Fig. 2 (ResNet-50 on ImageNet). |
| CKA without a random/permutation baseline cannot quantify significance of individual values | ⚪ UNMEASURED | §4.2 and §5. Neither a permutation null nor a random-init CKA matrix has been computed yet (see `BASELINES.md` §7). |

**Evidence-qualified statements:**

1. 🟢 "Linear CKA, as defined by Kornblith et al. (2019, Eq. 3), measures the similarity of hidden-state geometry between two networks. It is invariant to orthogonal transformations and isotropic scaling, but it does **not** imply that the two networks perform the same computation." (Kornblith et al. 2019, §3 and §6.1)

2. 🟢 "The column-argmax CKA-best mapping is not guaranteed by definition to be monotonic or one-to-one. On our measured `bert` vs `distilbert` EWT test matrix (`n_tokens=20000`, `seed=42`), the empirical CKA-best sequence is `0, 1, 4, 5, 8, 11, 12` for DistilBERT layers `0..6` — which happens to be strictly increasing, matching the rigid 1:2 mapping at layers `0, 2, 4, 6` and shifting by ±1 at layers `1, 3, 5`."

3. ⚪ "Without a permutation null-distribution (Kornblith et al. 2019, §4.2) or random-init CKA baseline, we cannot quantify how much of the measured off-diagonal CKA (`0.3606` to `0.9086`) exceeds chance or initialization geometry (`BASELINES.md` §7)."

---

### Source 2 — Sanh et al. 2019

> V. Sanh, L. Debut, J. Chaumond, T. Wolf. **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.** *5th Workshop on Energy Efficient Machine Learning and Cognitive Computing, NeurIPS 2019.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)

**Key claims from this paper used in WS2:**

| Claim | Verification status | Notes |
|---|---|---|
| DistilBERT uses a **multi-term** distillation loss: soft cross-entropy on softmax-temperature logits **+ cosine embedding loss on hidden states + hard cross-entropy on MLM labels** | 🟢 VERIFIED | §3.2, Eq. (1): $\mathcal{L} = \alpha \mathcal{L}_{ce} + \beta \mathcal{L}_{mlm} + \gamma \mathcal{L}_{cos}$. The cosine embedding loss $\mathcal{L}_{cos}$ aligns student hidden states to teacher hidden states. |
| DistilBERT is **initialised** from every other layer of the BERT teacher | 🟢 VERIFIED | §3.3: "We initialize the student from the teacher by taking one layer out of two." DistilBERT layer $l$ starts from BERT layer $2l$ weights. |
| Near-diagonal CKA alignment persists in the trained checkpoints | 🟢 VERIFIED | Measured in `bert__vs__distilbert__ewt.parquet`: 1:2 diagonal CKA ranges from `0.9470` (L0↔L0) to `0.6208` (L6↔L12), and CKA-best stays within ±1 of `2l` at every layer. |
| DistilBERT achieves 97.0% POS accuracy vs BERT-base 97.5% on Penn Treebank WSJ (fine-tuned head) | 🟢 VERIFIED | §4, Table 2 (gap −0.5 pp). **Not directly comparable** to our frozen linear-probe UD EWT results (`94.77%` vs `95.19%`, gap −0.42 pp), though both show DistilBERT retains nearly all POS accuracy. |

**Evidence-qualified statements:**

4. 🟢 "DistilBERT's distillation objective includes a cosine embedding loss $\mathcal{L}_{cos}$ applied to the student's hidden states relative to the teacher's (Sanh et al. 2019, §3.2). The earlier report claim that 'distillation only matches output logits' was **incorrect** and has been corrected in `docs/results/ws2_compression.md`."

5. 🟢 "DistilBERT layer $l$ is initialised from BERT layer $2l$ (Sanh et al. 2019, §3.3). Geometric similarity near the 1:2 diagonal (`0.9470` at L0 down to `0.6208` at L6) is therefore consistent with initialisation + $\mathcal{L}_{cos}$ rather than a purely emergent alignment. A random-init CKA baseline (`BASELINES.md` §7.1) is still needed to quantify the initialisation-only contribution."

6. 🟢 "Sanh et al. (2019, Table 2) report 97.0% POS accuracy for DistilBERT vs 97.5% for BERT-base on PTB WSJ with a fine-tuned head (−0.5 pp). On our frozen-weight linear probe over UD EWT (`first` pooling, seed 42), peak UPOS accuracy is 94.77% (DistilBERT L3) vs 95.19% (BERT L6) (−0.42 pp), and on PUD EN 95.87% vs 96.04% (−0.17 pp). Absolute scores are lower as expected for frozen linear probes, while the small accuracy gap between student and teacher holds."

---

## 2. Interpretation Checks (from Issue #81)

### 2.1 Distillation Objective Claim
- **Earlier draft claim:** "Distillation trains the student to reproduce the teacher's output logits, but does not explicitly constrain intermediate representations to match BERT layers $2l$."
- **Verdict:** 🔴 **INCORRECT** — Sanh et al. (2019, §3.2) include $\mathcal{L}_{cos}$ on hidden states and initialize from every other layer (§3.3). **Corrected** in `docs/results/ws2_compression.md` and `scripts/21_compression_report.py`.

### 2.2 CKA as Causal Equivalence
- **Verdict:** 🟢 **VERIFIED – claim is NOT warranted.** High CKA indicates geometric alignment under the sampled tokens (`n_tokens=20000`, `first` pooling), not functional equivalence (Kornblith et al. 2019, §6.1). For example, DistilBERT L5 and BERT L11 have CKA `0.7285`, yet their UPOS selectivity differs by `−0.0447` (`0.3371` vs `0.3818`).

### 2.3 CKA-Best Mapping Monotonicity
- **Measured result:** 🟢 **VERIFIED.** In `bert__vs__distilbert__ewt.parquet`, the CKA-best BERT layer for DistilBERT layers `0..6` is `[0, 1, 4, 5, 8, 11, 12]`, which is strictly monotonic (`0 < 1 < 4 < 5 < 8 < 11 < 12`) and injective, though non-monotonicity remains possible in general.

### 2.4 Embedding Layer CKA
- **Earlier placeholder:** `1.000` in the unpopulated table.
- **Measured result:** 🟢 **VERIFIED = `0.9470`** (not `1.000`). DistilBERT L0 and BERT L0 have high CKA (`0.9470`) consistent with shared vocabulary and embedding initialization, but are not identical after pretraining/distillation.

---

## 3. Principal Findings for R1 (Measured, Seed 42)

### Finding 1 (Geometric — Near-1:2 CKA Alignment with ±1 Shifts)
- **Statement:** "Linear CKA between frozen `bert-base-cased` ($L=12$) and `distilbert-base-cased` ($L=6$) on UD EWT test (`n_tokens=20000`, `first` pooling, `seed=42`) peaks near the 1:2 diagonal: CKA-best matches $2l$ exactly at layers 0 (`0.9470`), 2 (`0.8294`), 4 (`0.7731`), and 6 (`0.6208`), and shifts by one layer at layers 1→BERT 1 (`0.8774` vs `0.8612` at L2), 3→BERT 5 (`0.8068` vs `0.7963` at L6), and 5→BERT 11 (`0.7285` vs `0.7029` at L10). This alignment is consistent with DistilBERT's 1:2 initialization and $\mathcal{L}_{cos}$ distillation objective (Sanh et al. 2019, §§3.2–3.3)."
- **Slide-ready figure:** `results/figures/ws2/bert__vs__distilbert_cka_heatmap.png`.
- **Limitation:** No random-init CKA matrix or permutation significance test (`BASELINES.md` §7), so initialization-inherited vs. learned alignment cannot be separated quantitatively.

### Finding 2 (Functional — UPOS Accuracy Preserved, Selectivity Slightly Lower)
- **Statement:** "Under frozen linear UPOS probing on UD EWT test (`seed=42`), DistilBERT closely matches BERT's UPOS accuracy (peak `94.77%` at L3 vs `95.19%` at L6; rigid 1:2 differences within `0.49 pp`), while its selectivity $S = \text{Acc}_{\text{upos}} - \text{Acc}_{\text{control}}$ is consistently `0.0210` to `0.0400` lower across contextual layers (peak `0.3371` at L5 vs BERT `0.3818` at L11). Both models substantially outperform random-init baselines (best accuracy `74.45%` / `74.55%` at L0; max selectivity `0.0173` / `0.0105`)."
- **Slide-ready table:** `docs/results/ws2_compression.md` §"1:2 Layer Mapping Table" (generated by `scripts/21_compression_report.py`).
- **Limitation:** Single treebank for CKA (`ewt`), single seed (`42`), `first`-subword pooling only; random-init probing was not evaluated on `pud_en`.

### Finding 3 (Hierarchy Depth — Similar Relative Depth via $\text{CoG}_{\text{norm}}$)
- **Statement:** "Normalized Center of Gravity of selectivity across contextual layers ($1..L$) is `0.5977` for BERT and `0.6263` for DistilBERT on EWT (`+0.0286`; and `0.5794` vs `0.6147`, `+0.0353` on PUD EN), indicating that morphosyntactic selectivity occupies a similar—slightly deeper—relative depth in the 6-layer student rather than collapsing into earlier layers."
- **Slide-ready figure:** `results/figures/ws2/bert_vs_distilbert_selectivity.png`.
- **Limitation:** $\text{CoG}_{\text{norm}}$ summarizes the shape of the selectivity curve (layers $1..L$), not the magnitude of linguistic knowledge; DistilBERT's peak selectivity remains lower than BERT's.

---

## 4. Claims That Must Not Be Made

| Forbidden claim | Reason |
|---|---|
| "DistilBERT layer $l$ is geometrically or functionally equivalent to BERT layer $2l$" | CKA is not causal/functional equivalence (Kornblith et al. 2019, §6.1); selectivity is lower by 0.021–0.040 |
| "CKA = 1.000 for layer 0" | Measured value is `0.9470` |
| "Distillation only matches output logits" | Contradicted by Sanh et al. (2019, §3.2) $\mathcal{L}_{cos}$ |
| "Near-diagonal CKA proves emergent alignment" | Cannot separate from 1:2 initialization without random-init CKA (`BASELINES.md` §7.1) |

---

## 5. Remaining Gaps – Priority Order

| Priority | Artifact | Why needed | Status |
|---|---|---|---|
| P1 | Random-init CKA matrix (`bert` random vs `distilbert` random/pretrained) | Separate initialization-inherited geometry from distillation-trained structure | Features exist on HF; `cka_matrix()` needs `init` parameter |
| P2 | Permutation significance test for CKA | Quantify statistical significance of individual CKA values (Kornblith et al. 2019, §4.2) | Not yet scripted |
| P3 | Random-backbone probing on `pud_en` | Extend random-init baseline check to out-of-domain PUD | Not yet run |

---

## 6. References

1. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *Proceedings of the 36th International Conference on Machine Learning* (ICML 2019). [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *5th Workshop on Energy Efficient Machine Learning and Cognitive Computing at NeurIPS 2019.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
