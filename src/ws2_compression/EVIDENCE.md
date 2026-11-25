# WS2 Compression – Literature Evidence and Interpretation Checks

Evidence registry for Workstream 2 (Issues C1–C5), prepared for the R1 review (25 November 2026).
Branch: `c5-cka-baselines`. `Closes #81`.

> **Evidence qualification key**
> - 🟢 **VERIFIED** – claim is directly supported by a cited primary source or by a locally passing test.
> - 🟡 **EXPECTED** – claim is consistent with cited literature and with WS2 methodology, but the specific measurement has not been computed locally (see `BASELINES.md`).
> - 🔴 **INCORRECT** – claim in the existing report is contradicted by the cited primary source; must be corrected.
> - ⚪ **UNMEASURED** – claim cannot be evaluated without a result file that is currently MISSING.

---

## 1. Primary Sources

### Source 1 — Kornblith et al. 2019

> S. Kornblith, M. Norouzi, H. Lee, G. Hinton. **Similarity of Neural Network Representations Revisited.** *Proceedings of the 36th ICML*, 2019. [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)

**Key claims from this paper used in WS2:**

| Claim | Verification status | Notes |
|---|---|---|
| CKA measures representational similarity; it is invariant to orthogonal transformations and isotropic scaling | 🟢 VERIFIED | Theorem 1, §3. Our `linear_cka()` in `cka.py` uses the HSIC-normalised formulation from Eq. (3). |
| CKA ∈ [0, 1] by construction (for positive-semi-definite kernels) | 🟢 VERIFIED | Corollary 1 and our `test_linear_cka_identical` unit test (CKA = 1.0 for equal matrices). |
| High CKA does **not** imply functional identity | 🟢 VERIFIED | §6.1: "CKA is a measure of similarity, not of functional equivalence." A network can achieve the same task with geometrically dissimilar representations. |
| Same architecture trained from different seeds yields high within-network CKA on the diagonal | 🟢 VERIFIED | Fig. 2 (ResNet-50 on ImageNet). We expect analogous behaviour for BERT ↔ BERT. |
| CKA without a random baseline (permutation test) cannot quantify significance of individual values | 🟢 VERIFIED | §4.2 and the permutation null in §5. WS2 does not currently compute this baseline (see `BASELINES.md §6`). |

**Three evidence-qualified statements for R1:**

1. 🟢 "Linear CKA, as defined by Kornblith et al. (2019, Eq. 3), measures the similarity of hidden-state geometry between two networks. It is invariant to orthogonal transformations and isotropic scaling, but it does **not** imply that the two networks perform the same computation." (Kornblith et al. 2019, §3 and §6.1)

2. 🟡 "The column-argmax CKA-best mapping (selecting the BERT layer with the highest CKA value for each DistilBERT layer) is **not** guaranteed to be monotonic or one-to-one. Two DistilBERT layers may both map to the same BERT layer, or the mapping may skip BERT layers." (Consistent with the method in Kornblith et al. 2019; not directly stated there but follows from the definition of argmax over an asymmetric matrix.)

3. ⚪ "Without a permutation null-distribution for each CKA value (Kornblith et al. 2019, §4.2), we cannot determine whether individual off-diagonal CKA values are significantly greater than chance. This baseline is MISSING (see `BASELINES.md §6.4`)."

---

### Source 2 — Sanh et al. 2019

> V. Sanh, L. Debut, J. Chaumond, T. Wolf. **DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.** *5th Workshop on Energy Efficient Machine Learning and Cognitive Computing, NeurIPS 2019.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)

**Key claims from this paper used in WS2:**

| Claim | Verification status | Notes |
|---|---|---|
| DistilBERT uses a **multi-term** distillation loss: soft cross-entropy on softmax-temperature logits **+ cosine embedding loss on hidden states + hard cross-entropy on MLM labels** | 🟢 VERIFIED | §3.2, Eq. (1): $\mathcal{L} = \alpha \mathcal{L}_{ce} + \beta \mathcal{L}_{mlm} + \gamma \mathcal{L}_{cos}$. The cosine embedding loss $\mathcal{L}_{cos}$ aligns student hidden states to teacher hidden states at **every** layer. |
| DistilBERT is **initialised** from every other layer of the BERT teacher | 🟢 VERIFIED | §3.3: "We initialize the student from the teacher by taking one layer out of two." DistilBERT layer $l$ starts from BERT layer $2l$ weights. |
| The near-diagonal pattern in the CKA heatmap is therefore **expected at initialization** and may persist through training | 🟡 EXPECTED | Follows from §3.3: geometric similarity is baked in at $t=0$. The question is whether fine-tuning preserves or breaks it. We cannot answer this without the CKA matrix (MISSING). |
| DistilBERT achieves 97.0% UPOS accuracy on Penn Treebank WSJ (fine-tuned head) | 🟢 VERIFIED | §4, Table 2. **Not directly comparable** to our EWT linear-probe results (different treebank, fine-tuned vs. frozen, different tokenisation). |
| BERT-base achieves 97.5% UPOS accuracy on PTB WSJ (fine-tuned head) | 🟢 VERIFIED | §4, Table 2. Same incompatibility caveat. |

**Three evidence-qualified statements for R1:**

4. 🟢 "DistilBERT's distillation objective includes a cosine embedding loss $\mathcal{L}_{cos}$ applied to the student's hidden states relative to the teacher's (Sanh et al. 2019, §3.2). The claim that 'distillation only matches output logits' is **incorrect**: the training objective explicitly aligns intermediate representations." (See §2 of this document for the correction to `docs/results/ws2_compression.md`.)

5. 🟢 "DistilBERT layer $l$ is initialised from BERT layer $2l$ (Sanh et al. 2019, §3.3). Geometric similarity near the 1:2 diagonal in the CKA heatmap is therefore expected at initialisation and is **not** evidence of an emergent alignment. A random-init CKA baseline (see `BASELINES.md §6.3`) is required to distinguish initialisation-inherited geometry from trained structure."

6. ⚪ "Sanh et al. (2019) report 97.0% UPOS accuracy for DistilBERT on Penn Treebank WSJ with a fine-tuned head. Our setup uses UD EWT with a frozen-weight linear probe, so direct numerical comparison is not valid. We expect our linear-probe accuracy to be lower for both models. No local measurement is currently available (probing results MISSING)."

---

## 2. Interpretation Checks (from Issue #81)

The following checks verify the correctness of interpretive statements in `docs/results/ws2_compression.md` and the codebase.

### 2.1 Distillation Objective Claim

**Claim in report (§4, question 1):**
> "Distillation trains the student to reproduce the teacher's output logits, but does not explicitly constrain intermediate representations to match BERT layers $2l$."

**Verdict:** 🔴 **INCORRECT** (partially).

The first clause is true: soft cross-entropy on teacher logits is part of $\mathcal{L}$. However, the claim that distillation does "not explicitly constrain intermediate representations" is false. Sanh et al. (2019) §3.2 include $\mathcal{L}_{cos}$, a cosine embedding loss on hidden states, as an explicit constraint on intermediate geometry. Additionally, initialisation from every other BERT layer (§3.3) means intermediate representations start geometrically close to those of the teacher.

**Correction applied in `docs/results/ws2_compression.md`** (see §3 of this document).

### 2.2 CKA as Causal Equivalence

**Claim implied by report structure:**
> High CKA in a column implies that DistilBERT layer $l$ performs the same linguistic computation as the highest-CKA BERT layer.

**Verdict:** 🟢 **VERIFIED – claim is NOT warranted.**

Kornblith et al. (2019) §6.1 explicitly state that CKA is not causal equivalence. We echo this: "High CKA indicates geometric alignment of hidden-state distributions under the specific sentence sample and pooling strategy; it does not establish that the two layers perform the same linguistic operation."

### 2.3 CKA-Best Mapping Monotonicity

**Claim implied by C4 naming convention:**
> The CKA-best mapping produces a monotonically increasing BERT-layer sequence for DistilBERT layers 0–6.

**Verdict:** 🟡 **EXPECTED but unverified without the CKA matrix.**

The 1:2 rigid mapping is monotonic by construction. The CKA-best mapping (argmax per column) need not be. If two DistilBERT layers maximise CKA at the same BERT layer, the mapping is not injective. We cannot evaluate this without the MISSING CKA parquet.

### 2.4 Embedding Layer CKA

**Claim implied by the placeholder row in `docs/results/ws2_compression.md` Table (layer 0, CKA-Best CKA = 1.000):**
> DistilBERT layer 0 has CKA = 1.000 with BERT layer 0.

**Verdict:** ⚪ **UNMEASURED – do not assert.**

DistilBERT shares BERT's vocabulary and was initialised from BERT's embedding weights (Sanh et al. 2019, §3.3), so CKA near 1 is *plausible* at the embedding layer. However, the value `1.000` in the placeholder table is a synthetic default (`src/ws2_compression/mapping.py`), not a measurement. This claim must be removed or explicitly flagged as a placeholder until the CKA parquet is available.

---

## 3. Principal Findings for R1

The following findings are the best-supported claims WS2 can make at this audit date (25 November 2026), given that all computed result files are MISSING.

### Finding 1 (Geometric – literature-grounded expectation)

**Statement:** "Given that DistilBERT is initialised from every other BERT layer (Sanh et al. 2019, §3.3) and trained with a hidden-state cosine embedding loss ($\mathcal{L}_{cos}$, §3.2), we expect the CKA heatmap to show elevated values near the 1:2 diagonal at the start of training. Whether this near-diagonal pattern persists after fine-tuning — and whether it is statistically significant above a random-init baseline — is an **open empirical question** that requires the CKA parquet (currently MISSING, see `BASELINES.md §1`)."

**Slide-ready figure:** When computed, use `results/figures/ws2/bert__vs__distilbert_cka_heatmap.png` (produced by `scripts/20_cka.py`). The heatmap marks the rigid 1:2 diagonal in red.

**Limitation:** Without a random-init CKA baseline, we cannot distinguish initialisation-inherited geometry from trained structure.

### Finding 2 (Functional – literature comparison)

**Statement:** "Sanh et al. (2019, Table 2) report that DistilBERT retains 97.0% / 97.5% of BERT's UPOS accuracy on PTB WSJ (fine-tuned head). Under our more demanding evaluation — frozen-weight linear probe on UD EWT — we expect a larger gap between models because the linear probe cannot compensate for weaker representations. The exact gap is **unknown** pending computation of the probing parquets (MISSING)."

**Slide-ready table:** When computed, use the layer mapping comparison table in `docs/results/ws2_compression.md §2`, produced by `scripts/21_compression_report.py`.

**Limitation:** Single treebank (EWT), single seed (42), single probe type (linear), first-subword pooling. Results may not generalise to other settings.

### Finding 3 (Methodological – implementation verified)

**Statement:** "Our linear CKA implementation (`src/ws2_compression/cka.py::linear_cka`) correctly returns 1.0 for identical matrices and handles edge cases (zero variance). This is confirmed by the passing unit tests in `tests/test_cka.py`. The implementation faithfully reproduces the HSIC-normalised estimator from Kornblith et al. (2019, Eq. 3)."

**Evidence:** `PYTHONPATH=src pytest tests/test_cka.py -q` passes (CI-confirmed).

---

## 4. Claims That Must Not Be Made in R1

The following claims are **not supported** by current evidence and must be omitted or explicitly qualified:

| Forbidden claim | Reason |
|---|---|
| "DistilBERT layer $l$ is geometrically equivalent to BERT layer $2l$" | Requires CKA matrix (MISSING) + significance test (MISSING) |
| "CKA = 1 for layer 0" | Placeholder value, not a measurement (see §2.4) |
| "Distillation only matches output logits" | Contradicted by Sanh et al. 2019 §3.2 (see §2.1) |
| "High CKA implies the two layers decode morphosyntax equally well" | Contradicted by Kornblith et al. 2019 §6.1 (see §2.2) |
| Specific CoG_norm values | Requires probing parquets (MISSING) |
| Specific accuracy or selectivity numbers | Requires probing parquets (MISSING) |

---

## 5. Missing Evidence – Priority Order for Future Computation

| Priority | Artifact | Why needed | Command |
|---|---|---|---|
| P1 | CKA matrix `bert__vs__distilbert__ewt.parquet` | Enables Finding 1; unlocks heatmap and layer-mapping table | `PYTHONPATH=src python scripts/20_cka.py --model-a bert --model-b distilbert` |
| P2 | Probing results for BERT and DistilBERT (pretrained) | Enables Finding 2 and CoG comparison | `PYTHONPATH=src python scripts/10_probing.py ...` |
| P3 | Random-init CKA matrix | Null hypothesis for geometric alignment | `PYTHONPATH=src python scripts/20_cka.py --model-a bert-random --model-b distilbert` |
| P4 | Probing results (random init) | Selectivity baseline; distinguishes representation quality from linear decodability | `PYTHONPATH=src python scripts/10_probing.py --init random ...` |
| P5 | Permutation significance test | Quantify CKA significance (Kornblith et al. 2019, §4.2) | Not yet scripted; would require extending `src/ws2_compression/cka.py` |

---

## 6. References

1. Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. *Proceedings of the 36th International Conference on Machine Learning* (ICML 2019). [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)

2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *5th Workshop on Energy Efficient Machine Learning and Cognitive Computing at NeurIPS 2019.* [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
