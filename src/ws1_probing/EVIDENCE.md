# WS1 probing: literature evidence (P7 / #80)

Checked on 24 November 2026. This note evaluates the claims in the [WS1 report](../../docs/results/ws1_probing.md); implementation, full inventory, shared snapshot and limitations are in [BASELINES.md](BASELINES.md).

**Assessment:** the literature supports our choice of complementary controls and several observed trends. Published BERT/GPT-2 values provide useful numerical context, but none of the reviewed experiments matches our checkpoint, data/splits, pooling, sample size and probe procedure sufficiently to count as a numerical replication. Similar scores alone cannot validate an implementation.

## 1. Primary sources and what they establish

Links lead to paper pages, not directly to PDFs. Page references below are printed proceedings pages.

| Source | Relevant location | Evidence for this project | Boundary |
| :-- | :-- | :-- | :-- |
| Hewitt and Liang (2019), [Designing and Interpreting Probes with Control Tasks](https://aclanthology.org/D19-1275/) | §2, pp. 2734–2735; §3 | Fixed word-type controls and task-minus-control accuracy; POS labels sampled from the empirical token distribution. Our construction follows this principle. | Experiments use ELMo and Penn Treebank POS, not our BERT/GPT-2 UPOS benchmark. |
| Cao, Sanh and Rush (2021), [Low-Complexity Probing via Finding Subnetworks](https://aclanthology.org/2021.naacl-main.74/) | §§4–5 and Table 1, p. 962 | Tests pretrained BERT, reset encoder, and reset encoder plus embeddings. POS MLP-1 accuracy is 90.25%, 86.53% and 69.16%, respectively. | `bert-base-uncased`, UD POS and a different probe. The 95.69% fine-tuning row is not a frozen-probe baseline. |
| Conia and Navigli (2022), [Probing for Predicate Argument Structures in Pretrained Language Models](https://aclanthology.org/2022.acl-long.316/) | §3.1 and Table 1, pp. 4624–4625 | Uses Random and Static references with trained probes and frozen language models, including BERT-base-cased. | Predicate senses/rolesets on CoNLL-2009, not UPOS. Mean subword pooling and layer combinations differ. Its accuracy/F1 cannot confirm our POS numbers. |
| Zhang and Bowman (2018), [Language Modeling Teaches You More than Translation Does: Lessons Learned Through Auxiliary Syntactic Task Analysis](https://aclanthology.org/W18-5448/) | §3, “Untrained Baseline” and “Word Identity”, pp. 359–360 | Untrained representations can support syntactic classifiers through recoverable lexical identity. | LSTM encoders and different tasks/protocols; mechanism-level support, not a Transformer numerical comparison. |
| Alammar (2021), [Ecco: An Open Source Library for the Explainability of Transformer Language Models](https://aclanthology.org/2021.acl-demo.30/) | §§5.1–5.2, p. 253; Table 1, p. 254 | BERT numerical comparison; see section 2. | Use **Hidden States**, not FFNN activations. |
| Li, Cotterell and Sachan (2022), [Probing via Prompting](https://aclanthology.org/2022.naacl-main.84/) | §§3.1–3.5, pp. 1146–1149; Table 3, p. 1148 | GPT-2 numerical comparison and control limitations; see section 3. | Different POS protocol. |
| Ács et al. (online 2023), [Morphosyntactic probing of multilingual BERT models](https://www.cambridge.org/core/journals/natural-language-engineering/article/morphosyntactic-probing-of-multilingual-bert-models/8C0D539D3F11FB188AB73228BA7F5805) | §8.4, Table 8 and Figure 21 | Tests random mBERT/XLM-R, with and without pretrained embeddings. Context perturbations scarcely affect random-model performance; the authors consequently do not adopt randomized models as their baselines. | Relevant contrary evidence about interpretation; multilingual morphological tasks differ from our English UPOS setup. |

The last source limits, rather than invalidates, our choice: a random backbone is useful as an untrained-feature reference, but a weak sole reference for contextual linguistic structure. Our layer-0 and word-type controls provide additional checks; we have not performed the perturbation tests needed to directly establish reliance on context.

## 2. Closest BERT comparison: Ecco

Local protocol: `bert-base-cased`, frozen, linear probe, `first`, UD r2.19 EWT train/dev/test, 50,000 sampled training words, 25,094 test words, seed 42. Artifact: [bert__ewt__linear__pretrained__first.parquet](../../results/probing/bert__ewt__linear__pretrained__first.parquet), `eval_treebank=ewt`, paired `upos`/`control`.

Ecco uses English UD v2, 10,000 tokens split 67%/33%, linear Adam probes for 50 epochs (learning rate 0.001), and averages five trials. Matching checkpoint variant, treebank, pooling and control implementation are not established. Its block IDs 0..11 map to our 1..12; `Embed` maps to our 0. Below are published means, with selectivity divided by 100. [Source: §5 and Table 1](https://aclanthology.org/2021.acl-demo.30/).

| Representation | Our accuracy | Published accuracy | Our S | Published S |
| :-- | --: | --: | --: | --: |
| Embeddings: our 0 / paper Embed | 87.23% | 87.6% | 0.1148 | 0.061 |
| Our 5 / paper 4 | 95.07% | 94.9% | 0.2966 | 0.275 |
| Our 11 / paper 10 | 94.33% | 90.6% | 0.3818 | 0.373 |
| Our 12 / paper 11 | 93.66% | 89.0% | 0.3730 | 0.367 |

**Verdict: qualitative support, with numerical differences.** Both curves separate an earlier accuracy optimum from later selectivity. Embedding selectivity and late-layer accuracy differ substantially; the latter is over four percentage points apart. These discrepancies remain unexplained by a controlled experiment. Different protocols could contribute, but similarity at selected layers is not replication or statistical agreement.

## 3. Closest GPT-2 comparison: Probing via Prompting

Local protocol: `gpt2`, frozen, `first`, UD r2.19 EWT train/dev/test, 50,000 training words, 25,094 test words, seed 42. Compare layer 12 of the [linear](../../results/probing/gpt2__ewt__linear__pretrained__first.parquet) and [MLP](../../results/probing/gpt2__ewt__mlp__pretrained__first.parquet) files, `eval_treebank=ewt`.

The paper uses GPT-2 Small and OntoNotes 5.0 POS, with train/dev/test splits. DP(LR) uses the last layer; DP(MLP) mixes layers. Span pooling and one-epoch Adam training differ from our procedure. [Source: §§3.1–3.3](https://aclanthology.org/2022.naacl-main.84/). These feature, label and fitting differences prevent a controlled comparison with our per-layer UPOS probes.

| Configuration | POS/UPOS accuracy | Control accuracy | S |
| :-- | --: | --: | --: |
| Local pretrained linear, layer 12 | 91.44% | 55.35% | 0.3609 |
| Paper pretrained DP(LR) | 89.56% | 48.75% | 0.4081 |
| Local pretrained MLP, layer 12 | 92.29% | 67.23% | 0.2506 |
| Paper pretrained DP(MLP), layer mixture | 94.01% | 69.58% | 0.2443 |

Published values: [Table 3, p. 1148](https://aclanthology.org/2022.naacl-main.84/), converting the reported difference in percentage points to our scale. **Verdict: qualitative support.** MLP accuracy gains accompany larger control gains and lower selectivity in both setups. The numerical resemblance of MLP selectivity does not validate our value.

A critical caveat from §3.5: representation-dependent control scores can rank methods differently from random-backbone comparisons. We therefore interpret task accuracy, control accuracy and selectivity jointly, without treating control accuracy as pure probe capacity.

## 4. Audit of the report's main findings

All local values below use EWT test, seed 42. Artifacts are the measured parquets in [BASELINES.md, section 2](BASELINES.md#2-measured-inventory-and-provenance); links here select the exact files. “No comparable evidence found” is limited to the sources reviewed, not a claim about the entire literature. No comparison qualifies as a matched numerical replication.

| Report finding and local evidence | Published comparison / matching conditions | Differences and verdict |
| :-- | :-- | :-- |
| **Current-UPOS accuracy peaks before selectivity.** BERT: accuracy 95.19% at 6, S 0.3818 at 11; GPT-2: 93.44% at 8, S 0.3609 at 12. Sources: [BERT linear pretrained first](../../results/probing/bert__ewt__linear__pretrained__first.parquet), [GPT-2 linear pretrained first](../../results/probing/gpt2__ewt__linear__pretrained__first.parquet). | Ecco, Table 1: BERT's accuracy and selectivity favor different depths. Same task family, model family and metrics; section 2 gives aligned values. | **Qualitative support for BERT.** No matching GPT-2 layerwise protocol found. Li et al.'s last-layer/mixture results do not verify our GPT-2 peak layer. |
| **Pretrained features improve on the random reference.** Linear best accuracy: BERT 95.19% vs 74.45%; GPT-2 93.44% vs 74.28%. Random maximum S: 0.0173 / 0.0205. Sources: pretrained files above plus [BERT random](../../results/probing/bert__ewt__linear__random__first.parquet), [GPT-2 random](../../results/probing/gpt2__ewt__linear__random__first.parquet). | Cao et al., Table 1, BERT POS: pretrained MLP-1 90.25%, reset-all 69.16%. Li et al., Table 3, random GPT-2: DP(LR) S 0.0308, DP(MLP) S 0.0218. | **Qualitative support.** Cao uses uncased BERT and a different MLP; Li uses OntoNotes and different probes. These comparisons do not establish a universal expected random score. |
| **Contextual representations beat the pretrained layer-0 reference.** BERT 87.23% → 95.19%; GPT-2 88.39% → 93.44%, gains 7.97 / 5.05 pp. Sources: pretrained linear first files above. | Conia and Navigli, §3.1: Random/Static/contextual comparison, including BERT-base-cased; Ecco gives a POS embedding reference. | **Qualitative support.** Conia and Navigli study SRL-related tasks, not POS; their Static and our embedding output are conceptually related, not proven identical tensors. Improvement cannot be attributed solely to contextual syntax. |
| **MLP does not improve control-based selectivity.** At BERT 11: linear S 0.3818 vs MLP 0.2579; at GPT-2 12: 0.3609 vs 0.2506. MLP S is lower at every layer. Sources: linear files above plus [BERT MLP](../../results/probing/bert__ewt__mlp__pretrained__first.parquet), [GPT-2 MLP](../../results/probing/gpt2__ewt__mlp__pretrained__first.parquet). | Hewitt and Liang motivate the capacity/control comparison; Li et al., Table 3, shows the same ordering for GPT-2. | **Qualitative support.** Different capacity/optimization; control depends on representations too. No conclusion that nonlinear linguistic information is absent. |
| **BERT next-UPOS exceeds GPT-2; selectivity peaks earlier than for current UPOS.** Best accuracy 82.96% at BERT 6 vs 52.73% at GPT-2 11; S peaks 0.4432 at 6 vs 0.2459 at 9. Sources: [BERT first](../../results/probing/bert__ewt__linear__pretrained__first.parquet), [GPT-2 last](../../results/probing/gpt2__ewt__linear__pretrained__last.parquet), `upos_next`/`control_next`. | **No comparable evidence found** for this exact next-word setup in the reviewed papers. | A local result with unequal context: BERT sees the target word and GPT-2 does not. Control subtraction does not equalize context. No claim of superior forecasting, and random/MLP next-word controls are absent. |
| **CoG summarizes depth, not information amount.** Linear current-UPOS CoG_norm: 0.5977 / 0.5943; MLP: 0.6383 / 0.6534 despite lower S throughout. Sources: the four pretrained first files and `lib.metrics.summarize_probing`. | Li et al., §4.1 / Figure 3, also reports CoG, but over a distribution of selected attention heads. | **No numerically comparable evidence found.** Our weights are clipped selectivity over layers 1..L; theirs are head counts. Higher CoG does not establish greater knowledge or an exact peak. |

BERT's current-UPOS advantage over GPT-2 is also only descriptive: attention direction, checkpoint training, tokenizer and architecture vary together. None of these sources isolates the causal effect of the attention mask for our measurements. DistilBERT/PUD/Italian artifacts are inventoried for completeness; this note makes no new comparative performance claim for them.

## 5. Presentation-ready table and three statements

Use this table with the label **“UD EWT test, frozen backbone, first-subword pooling, seed 42; selectivity = task accuracy − control accuracy.”** These are layer-matched values, not comparisons between different maxima. Sources: the BERT/GPT-2 pretrained `first` linear/MLP files linked above; `upos` and `control` rows.

| Model / layer | Probe | UPOS accuracy | Control accuracy | Selectivity |
| :-- | :-- | --: | --: | --: |
| BERT / 11 | Linear | 94.33% | 56.15% | 0.3818 |
| BERT / 11 | MLP | 95.18% | 69.39% | 0.2579 |
| GPT-2 / 12 | Linear | 91.44% | 55.35% | 0.3609 |
| GPT-2 / 12 | MLP | 92.29% | 67.23% | 0.2506 |

1. **Finding:** pretrained features support stronger UPOS decoding. **Baseline:** random-backbone linear probes reach only 0.0173 / 0.0205 maximum selectivity. **Limit:** random features still carry lexical cues; this does not isolate learned contextual syntax.
2. **Finding:** the MLP's small task-accuracy gains accompany larger control gains. **Baseline:** matched linear/MLP word-type controls at every layer. **Limit:** the conclusion is specific to these two fitting procedures and this control, whose difficulty depends on the representation.
3. **Finding:** accuracy and selectivity identify different preferred depths. **Baseline:** show task/control curves and layer 0 together. **Limit:** one seed, no uncertainty estimates, and no exact literature replication; CoG is only a curve summary.

For the presentation, say **“consistent with published trends under different protocols”**, not “the papers prove our values are correct.” Literature supports the design and helps interpret the results; artifact checks support internal consistency. The two forms of evidence answer different questions.
