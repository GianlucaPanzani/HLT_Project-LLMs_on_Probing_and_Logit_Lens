# BERT vs GPT-2 probing results (issue P6 / #14)

Question: how does UPOS information emerge across layers when a model sees only the left context (GPT-2) versus both sides (BERT), for the current word and the next word? Do pretrained representations improve on lexical lookup and random weights, and does a nonlinear probe improve selectivity?

Data: UD r2.19, English-EWT train/dev/test. Models: `bert` (`bert-base-cased`) and `gpt2`, both with 12 Transformer blocks and hidden size 768. All backbone weights are frozen. Results below use seed 42 only and EWT test; PUD results are not included in these figures. Metric and alignment definitions follow [CONTRACTS.md](../CONTRACTS.md), sections 3, 6, 7 and 10; the research questions follow [PROPOSAL.md](../../PROPOSAL.md), workstream 1.

## Summary

- **Current-word selectivity peaks late in both models.** With linear probes, BERT peaks at layer 11 (`tau=0.917`, selectivity 0.3818); GPT-2 peaks at layer 12 (`tau=1.000`, 0.3609). Their normalized centers of gravity are close: 0.5977 and 0.5943. Accuracy peaks earlier, at layers 6 and 8, respectively. The late rise in selectivity partly reflects declining control accuracy.
- **Next-word UPOS is much more decodable from BERT.** BERT reaches 82.96% accuracy and 0.4432 selectivity at layer 6. GPT-2 reaches 52.73% accuracy at layer 11 and 0.2459 selectivity at layer 9. BERT can attend to the target word; GPT-2 cannot. This is a comparison of decodability under different context access, not an equal-context forecasting benchmark.
- **Pretraining and contextual layers both help.** Pretrained layer-0 UPOS accuracy is already 87.23% for BERT and 88.39% for GPT-2. Contextual layers raise the best accuracy by 7.97 and 5.05 percentage points. Random-weight models reach their highest accuracy at layer 0 and lose accuracy with depth; their maximum selectivity is only 0.0173 and 0.0205.
- **The MLP does not improve selectivity.** It slightly improves the best UPOS accuracy, but raises control accuracy much more. Selectivity is lower than the linear probe at every layer for both models. These runs support the linear probe as the more selective readout under the project's control task.

## 1. Experimental setup and reading the figures

Probes use 50,000 training rows per task. Task and control share the selected rows within each pair. Hyperparameters and MLP early stopping use dev; reported accuracies use test. The linear probe is regularized multinomial logistic regression. The MLP has one hidden layer of 256 units with ReLU and early stopping. Both standardize features using training statistics.

| Comparison | Probe | Initialization | BERT pooling | GPT-2 pooling | Test rows per task/control |
| :-- | :-- | :-- | :-- | :-- | --: |
| Current-word `upos` | linear | pretrained | first | first | 25,094 |
| Next-word `upos_next` | linear | pretrained | first | last | 23,017 |
| UPOS baselines | linear | pretrained, random | first | first | 25,094 |
| UPOS probe capacity | linear, MLP | pretrained | first | first | 25,094 |

For `upos_next`, the representation of word `w` predicts the UPOS of the next word in the same sentence. Sentence-final words have no next-word label and are excluded. GPT-2 uses the last subword of `w`, as required by the contract; for current-word UPOS it uses the first subword, which need not have seen the rest of a split word.

The control labels are a fixed random mapping from word types to UPOS classes, sampled using the training tag distribution. For the next-word task, the control label belongs to the next word. The mapping uses seed 42 and is shared across models. Random backbone initialization also uses seed 42; the probes are trained on its frozen features.

All horizontal axes use `tau = layer / 12`, including the embedding output at layer 0. Selectivity is task accuracy minus its matched control accuracy. Accuracy values in the tables are percentages; selectivity and `CoG_norm` are reported on their original scales. Changes in accuracy or selectivity expressed as **pp** are percentage-point differences.

`CoG_norm` summarizes the whole selectivity curve over layers 1..12, excluding layer 0 and clipping negative selectivity to zero for this calculation only. Negative values are retained when reporting selectivity itself. CoG is neither the peak layer nor a measure of how much information is present; section 6 reports it alongside the peak and its magnitude.

## 2. Current-word UPOS: where does selective information emerge?

![BERT and GPT-2 current-word UPOS accuracy, control accuracy and selectivity](../../results/figures/ws1/bert_vs_gpt2__ewt__ewt__seed42__upos_curves.png)

[Interactive figure](../../results/figures/ws1/bert_vs_gpt2__ewt__ewt__seed42__upos_curves.html).

| Model | Best accuracy | Accuracy peak layer | Selectivity peak layer (`tau`) | Accuracy at selectivity peak | Control at selectivity peak | Maximum selectivity |
| :-- | --: | --: | :-- | --: | --: | --: |
| BERT | 95.19% | 6 | 11 (0.917) | 94.33% | 56.15% | 0.3818 |
| GPT-2 | 93.44% | 8 | 12 (1.000) | 91.44% | 55.35% | 0.3609 |

Both models support strong UPOS decoding before the last layer. BERT has higher task accuracy and selectivity from layer 2 onward, while GPT-2 starts slightly ahead at layers 0 and 1. BERT's advantage is consistent with access to right context, but these two models also differ in tokenization, pretraining and architecture; this experiment does not isolate the attention mask as the only cause.

The answer depends on the metric. BERT's accuracy peaks at layer 6 and GPT-2's at layer 8, whereas their selectivity peaks are at layers 11 and 12. From BERT layer 6 to 11, task accuracy falls from 95.19% to 94.33%, but control accuracy falls more, from 63.38% to 56.15%. GPT-2 shows the same effect from layer 8 to 12: accuracy falls from 93.44% to 91.44%, while control accuracy falls from 61.98% to 55.35%.

Thus, the late layers separate the linguistic task from the arbitrary type-label control more strongly, even though they do not maximize raw UPOS accuracy. The nearly equal `CoG_norm` values (0.5977 vs 0.5943) describe similar overall depth profiles; they do not imply equal selectivity or identical peak locations.

## 3. Next-word UPOS: what changes with causal versus bidirectional context?

![BERT and GPT-2 next-word UPOS accuracy, control accuracy and selectivity](../../results/figures/ws1/bert_vs_gpt2__ewt__ewt__seed42__upos_next_curves.png)

[Interactive figure](../../results/figures/ws1/bert_vs_gpt2__ewt__ewt__seed42__upos_next_curves.html).

| Model | Best accuracy | Accuracy peak layer | Selectivity peak layer (`tau`) | Accuracy at selectivity peak | Control at selectivity peak | Maximum selectivity |
| :-- | --: | --: | :-- | --: | --: | --: |
| BERT | 82.96% | 6 | 6 (0.500) | 82.96% | 38.64% | 0.4432 |
| GPT-2 | 52.73% | 11 | 9 (0.750) | 51.94% | 27.35% | 0.2459 |

BERT gains next-word decodability rapidly: accuracy rises from 36.81% at layer 0 to 67.83% at layer 1 and 80.56% at layer 2. It reaches a broad middle-layer selectivity plateau: layers 6 and 7 differ by only 0.000087 selectivity (0.0087 pp). Layer 6 is the numerical maximum, not evidence for a sharply localized optimum. After the middle layers, both task accuracy and selectivity decline.

GPT-2 improves more gradually, from 37.59% accuracy at layer 0 to 52.73% at layer 11. Its selectivity peaks at layer 9; subsequent control gains outpace task gains. BERT's accuracy exceeds GPT-2's at every contextual layer. At the same layer 6, the comparison is 82.96% vs 48.42% accuracy and 0.4432 vs 0.2356 selectivity.

This difference answers the context-access question: BERT's representation at the current word can incorporate the next word itself, while GPT-2's last-subword representation contains only the prefix through the current word. BERT's higher control accuracy is consistent with access to the next word's identity as well. Its higher selectivity shows that the advantage persists after subtracting this control, but does not remove the difference in available context.

Relative to current-word probing, the selectivity peak moves earlier in both models: BERT from layer 11 to 6, GPT-2 from 12 to 9. GPT-2's next-word curve is the appropriate probing input for the later lens comparison, using the contracted `last` pooling. These results concern the next **word's UPOS**, not next-subword vocabulary accuracy or generated-text quality. No lens comparison is made here.

## 4. Baselines: how much comes from lexical lookup and pretraining?

![Pretrained and random-weight UPOS accuracy with layer-0 references](../../results/figures/ws1/bert_vs_gpt2__ewt__ewt__seed42__upos_baselines.png)

[Interactive figure](../../results/figures/ws1/bert_vs_gpt2__ewt__ewt__seed42__upos_baselines.html).

| Model | Pretrained layer 0 | Best pretrained accuracy (layer) | Gain over pretrained layer 0 | Random layer 0 | Random layer 12 |
| :-- | --: | :-- | --: | --: | --: |
| BERT | 87.23% | 95.19% (6) | +7.97 pp | 74.45% | 70.80% |
| GPT-2 | 88.39% | 93.44% (8) | +5.05 pp | 74.28% | 62.15% |

The horizontal lines are the separately measured layer-0 accuracies for each initialization. Pretrained embeddings already support substantial UPOS decoding, so high accuracy in isolation is insufficient evidence of contextual syntax. The improvement above the pretrained layer-0 reference shows the additional benefit of the contextual representations under the same probe setup.

With random weights, accuracy decreases from layer 0 through layer 12 in both models. Random representations still support substantial decoding: a trained probe can exploit word identity in fixed embeddings. Their control results, calculated from the same parquet pairs although not displayed in this accuracy-only figure, are correspondingly strong. The maximum random-model selectivity is just 0.0173 for BERT and 0.0205 for GPT-2, compared with 0.3818 and 0.3609 for pretrained models.

The random baselines therefore support a role for learned representations beyond the untrained architecture and lexical lookup. They do not justify interpreting roughly 74% random layer-0 accuracy as learned contextual linguistic knowledge. The `CoG_norm` of a nearly zero-selectivity curve is also a weak basis for claims about where linguistic information is concentrated.

## 5. Linear versus MLP: more linguistic selectivity or more control fitting?

![Linear and MLP UPOS accuracy, control accuracy and selectivity](../../results/figures/ws1/bert_vs_gpt2__ewt__ewt__seed42__upos_linear_vs_mlp.png)

[Interactive figure](../../results/figures/ws1/bert_vs_gpt2__ewt__ewt__seed42__upos_linear_vs_mlp.html).

The MLP increases the best task accuracy from 95.19% to 95.80% for BERT and from 93.44% to 93.85% for GPT-2. Those are comparisons of each probe's own maximum; BERT's accuracy peak shifts from layer 6 to 8, while GPT-2's remains at layer 8. The layer-matched comparison at each model's shared selectivity peak is:

| Model / layer | Probe | UPOS accuracy | Control accuracy | Selectivity |
| :-- | :-- | --: | --: | --: |
| BERT / 11 | linear | 94.33% | 56.15% | 0.3818 |
| BERT / 11 | MLP | 95.18% | 69.39% | 0.2579 |
| GPT-2 / 12 | linear | 91.44% | 55.35% | 0.3609 |
| GPT-2 / 12 | MLP | 92.29% | 67.23% | 0.2506 |

At BERT layer 11 the MLP gains 0.85 pp in task accuracy but 13.24 pp in control accuracy, reducing selectivity by 12.39 pp. At GPT-2 layer 12 it gains 0.85 pp in task accuracy but 11.88 pp in control accuracy, reducing selectivity by 11.03 pp. The MLP has lower selectivity at **all 13 layers** in both models. Its task accuracy is not uniformly higher: GPT-2 has small decreases at several early layers.

The observed benefit is therefore a small improvement in the best raw accuracy, accompanied by a much larger ability to fit the arbitrary word-type control. This is consistent with increased memorization capacity; the experiment does not show an improvement in selective linguistic decoding. It also does not establish that nonlinear linguistic information is absent: the conclusion concerns this MLP configuration and this control task.

The MLP's `CoG_norm` is higher than the linear probe's (0.6383 vs 0.5977 for BERT; 0.6534 vs 0.5943 for GPT-2), even though selectivity is lower everywhere. This reflects the distribution of the remaining selectivity across depth, not a gain in information. The selectivity peaks remain at layers 11 and 12.

## 6. Normalized centers of gravity

These rows are recomputed with `lib.metrics.summarize_probing` from the selected seed-42 results and match the corresponding rows in [cog.parquet](../../results/summary/cog.parquet). Only configurations used in the four figures are listed. Random-baseline CoGs are included for completeness; their selectivity magnitudes are small.

| Task | Model | Probe | Init | Pooling | CoG_norm | Peak layer | Peak tau | Max selectivity |
| :-- | :-- | :-- | :-- | :-- | --: | --: | --: | --: |
| upos | BERT | linear | pretrained | first | 0.5977 | 11 | 0.9167 | 0.3818 |
| upos | GPT-2 | linear | pretrained | first | 0.5943 | 12 | 1.0000 | 0.3609 |
| upos_next | BERT | linear | pretrained | first | 0.5462 | 6 | 0.5000 | 0.4432 |
| upos_next | GPT-2 | linear | pretrained | last | 0.5683 | 9 | 0.7500 | 0.2459 |
| upos | BERT | linear | random | first | 0.7050 | 11 | 0.9167 | 0.0173 |
| upos | GPT-2 | linear | random | first | 0.5352 | 2 | 0.1667 | 0.0205 |
| upos | BERT | MLP | pretrained | first | 0.6383 | 11 | 0.9167 | 0.2579 |
| upos | GPT-2 | MLP | pretrained | first | 0.6534 | 12 | 1.0000 | 0.2506 |

## 7. Scope and reproduction

These are descriptive results for one seed, one English test treebank and the recorded probe configurations. No error bars, confidence intervals or significance claims are implied. Layer maxima are descriptions of the test curves, not choices used to retune the probes. The four figures do not evaluate MLP or random-init next-word probing, cross-lingual transfer, or the Streamlit demo.

The inputs are the contracted files in [probing/](../../results/probing/), for BERT and GPT-2 trained on EWT: linear pretrained `first`, linear random `first`, MLP pretrained `first`, plus GPT-2 linear pretrained `last`. Task and control are paired by configuration, layer and seed; all curves contain layers 0..12. The figures were visually checked against the parquet values, and the CoG and peak values were checked against the saved summary. The linear runs used here contain no `converged=false` flags.

From the repository root, with the existing probing results available:

```bash
PYTHONPATH=src python scripts/11_summarize_probing.py
PYTHONPATH=src python scripts/12_plot_probing.py --models bert gpt2 --train-treebank ewt --eval-treebank ewt
```

The first command updates `results/summary/cog.parquet`; the plotting script computes its own summary in memory from the selected seed. The second command writes the four PNG files and four self-contained HTML files linked above. Neither command retrains probes or backbone models.

The PNG of the linear/MLP comparison currently has overlapping CoG annotations. The exact values are given in section 6; the annotation placement should be adjusted before the final presentation.

Artifacts under `results/` are gitignored; the version-controlled report is `docs/results/ws1_probing.md`. Share the generated figures through the team's storage and record their location in the P6 PR. The relative figure links resolve after the artifacts have been downloaded or regenerated locally; the generated files must not be committed. `scripts/13_push_probing_results.py` uploads only probing parquets, so it does not publish these figures.
