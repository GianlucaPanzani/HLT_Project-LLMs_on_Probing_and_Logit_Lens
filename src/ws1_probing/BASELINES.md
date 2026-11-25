# WS1 probing baseline audit (P7 / #80)

Audit date: 24 November 2026. Owner: Gianluca. Read this with [EVIDENCE.md](EVIDENCE.md) and the [WS1 result report](../../docs/results/ws1_probing.md). Definitions follow [CONTRACTS.md](../../docs/CONTRACTS.md), sections 3–7, and [PROPOSAL.md](../../PROPOSAL.md), section 5.

**Conclusion:** the implemented random backbone, layer-0 reference and word-type control are defensible, complementary probing controls. They support conclusions about the accessibility of UPOS labels under this experimental setup. They do not establish that selectivity measures a pure quantity of syntactic knowledge, or that the backbone uses that knowledge when generating text.

## 1. What each baseline tests

| Comparison | What is actually implemented | Strength | Limit |
| :-- | :-- | :-- | :-- |
| Pretrained vs random backbone | Same checkpoint configuration and tokenizer; the random model is constructed with `from_config`, seed 42, including newly initialized embeddings. Both backbones remain frozen; separate probes are trained on each. | Tests whether pretrained weights provide more useful features than an untrained architecture under the same probing procedure. | The tokenizer, architecture, positional structure and recoverable lexical identity remain. It is not a chance classifier. Only one initialization was measured. |
| Contextual layers vs layer 0 | Probe `hidden_states[0]`, the embedding output before any Transformer block, for both initializations. | Provides a within-model reference for decoding without contextual Transformer processing. | This is not necessarily a bare word-embedding lookup: position embeddings and model-specific embedding processing can contribute. Improvement over it does not isolate contextual syntax from all other transformations. |
| UPOS vs word-type control | A fixed, case-sensitive word-type-to-label mapping; labels are sampled from the complete training UPOS distribution with seed 42. Separate task/control probes use the same selected rows. | Reveals when high task accuracy accompanies strong learning of an arbitrary lexical mapping. | Control difficulty also depends on the representation, vocabulary overlap and label frequencies. Subtracting control accuracy does not exactly remove memorization. |
| Linear vs MLP | Compare the same features and labels using logistic regression and a one-hidden-layer MLP. Both also solve the control task. | Checks whether increased probe capacity improves accuracy without a larger control gain. | The training procedures and regularization differ too. This is a comparison of two concrete probes, not an exhaustive test of linear versus nonlinear information. |

The first two controls have direct precedents in [Conia and Navigli (2022), §3.1](https://aclanthology.org/2022.acl-long.316/) and [Cao et al. (2021), §§4–5](https://aclanthology.org/2021.naacl-main.74/). Word-type controls and accuracy-difference selectivity follow [Hewitt and Liang (2019), §2](https://aclanthology.org/D19-1275/). The task and protocol differences, and contrary evidence, are detailed in [EVIDENCE.md](EVIDENCE.md).

Three uses of randomness must be kept separate:

- **Random backbone initialization** is an experimental baseline. Its features are fixed throughout probe training; ordinary initialization can include constant biases/normalization parameters, so not every scalar weight is a random draw.
- **Probe initialization** is part of fitting a classifier. The MLP starts with freshly initialized weights and is trained; a new linear classifier is also fitted for each task/layer. This is not the `init=random` baseline, and probes are never intentionally left untrained.
- **Random control targets** define another supervised task. A word type retains its assigned label across occurrences and splits; labels are not redrawn per token or epoch. Unseen types receive a deterministic, hash-seeded draw from the training distribution. `control_next` uses the next word's assigned label.

Implementation references: [data.py](../lib/data.py), `load_model`; [features.py](../lib/features.py), extraction; [control_tasks.py](../lib/control_tasks.py); [runner.py](runner.py), `_build_control_mapping`, `_select_valid_data`, `run_probing`; [probes.py](probes.py), `LinearProbe` and `MLPProbe`. The control module's introductory wording that selectivity “measures how much” accuracy is memorization should be read as motivation, not as an established decomposition; the qualified interpretation here is the one used in the report.

## 2. Measured inventory and provenance

The audit read the actual shared repository, not just [configs/ws1.yaml](../../configs/ws1.yaml):

- Shared dataset: [hlt-group11/features, immutable snapshot](https://huggingface.co/datasets/hlt-group11/features/tree/7587960472da41ccea0ebab41e1249137c3988ff/results/probing), commit `7587960472da41ccea0ebab41e1249137c3988ff` (team access required).
- All **15 local result files, 708 rows**, matched the remote content identifiers byte-for-byte at this snapshot. Files reside under `results/probing/` both locally and remotely. Every row has seed 42 and `n_train=50000`.
- All **33 associated feature metadata files** were read at that snapshot. They report `max_rows=null`, zero NaN rows and zero truncated sentences; model computation is float32 and saved features are float16. The large feature matrices were not downloaded or independently rescanned.
- Source code inspected at Git commit `2cb85bc31f52cb0ff28d21386bc70f07ee19228d`. This is the audit's code revision, not an asserted historical training revision.

In the table, **U** means both `upos` and `control`; **N** means both `upos_next` and `control_next`. All listed files have status **measured**. `C` lists the selected values actually recorded; epochs are the observed MLP training range, including epochs after its best dev checkpoint. Filenames are relative to the exact `results/probing/` directory above.

| Result file | Tasks | Test treebanks | Rows | Recorded fit settings |
| :-- | :-- | :-- | --: | :-- |
| `bert__ewt__linear__pretrained__first.parquet` | U, N | ewt, pud_en | 104 | C: 0.01, 0.1, 1, 10 |
| `bert__ewt__linear__random__first.parquet` | U | ewt | 26 | C: 0.1, 1, 10 |
| `bert__ewt__mlp__pretrained__first.parquet` | U | ewt, pud_en | 52 | 9–31 epochs |
| `bert__ewt__mlp__random__first.parquet` | U | ewt | 26 | 13–26 epochs |
| `distilbert__ewt__linear__pretrained__first.parquet` | U | ewt, pud_en | 28 | C: 0.01, 0.1, 1 |
| `distilbert__ewt__linear__random__first.parquet` | U | ewt | 14 | C: 0.01, 0.1, 1, 10 |
| `distilbert__ewt__mlp__pretrained__first.parquet` | U | ewt, pud_en | 28 | 6–26 epochs |
| `distilbert__ewt__mlp__random__first.parquet` | U | ewt | 14 | 13–23 epochs |
| `gpt2__ewt__linear__pretrained__first.parquet` | U | ewt, pud_en | 52 | C: 0.01, 0.1, 1, 10 |
| `gpt2__ewt__linear__random__first.parquet` | U | ewt | 26 | C: 0.01, 0.1, 1, 10 |
| `gpt2__ewt__mlp__pretrained__first.parquet` | U | ewt, pud_en | 52 | 6–25 epochs |
| `gpt2__ewt__mlp__random__first.parquet` | U | ewt | 26 | 13–30 epochs |
| `gpt2__ewt__linear__pretrained__last.parquet` | U, N | ewt, pud_en | 104 | C: 0.01, 0.1, 1, 10 |
| `bert_it__isdt__linear__pretrained__first.parquet` | N | isdt, pud_it | 52 | C: 0.01, 0.1 |
| `geppetto__isdt__linear__pretrained__last.parquet` | U, N | isdt, pud_it | 104 | C: 0.01, 0.1, 1, 10 |

The [model registry](../../configs/models.yaml) and remote `meta.json` agree on these checkpoint identities:

| Key | Checkpoint | Blocks / hidden size |
| :-- | :-- | :-- |
| bert | `bert-base-cased` | 12 / 768 |
| distilbert | `distilbert-base-cased` | 6 / 768 |
| gpt2 | `gpt2` | 12 / 768 |
| bert_it | `dbmdz/bert-base-italian-cased` | 12 / 768 |
| geppetto | `LorenzoDeMattei/GePpeTto` | 12 / 768 |

Each measured curve contains layers `0..L`. GPT-2/GePpeTto's final hidden state includes the model's final normalization, as recorded by `final_norm_on_last_layer=true`. Layer 0 is an embedding output, not a network with zero weights.

### Splits, sample sizes and caps

Data are UD r2.19, configured in [datasets.yaml](../../configs/datasets.yaml); the 17 UPOS labels are defined in [data.py](../lib/data.py). Train and dev use the training treebank in the filename; evaluation uses its test split or the same-language PUD test split. PUD has no training/dev role here.

| Split | Available U rows | Available N rows | Probe use |
| :-- | --: | --: | :-- |
| EWT train | 204,578 | 192,034 | Sample 50,000 valid rows per task/control pair |
| EWT dev | 25,148 | 23,147 | All eligible rows; hyperparameter/early-stopping selection |
| EWT test | 25,094 | 23,017 | All eligible rows; reported evaluation |
| PUD EN test | 21,180 | 20,180 | Evaluation only, where listed above |
| ISDT train | 276,014 | 262,893 | Sample 50,000 valid rows per task/control pair |
| ISDT dev | 11,907 | 11,343 | All eligible rows; selection |
| ISDT test | 10,417 | 9,935 | All eligible rows; evaluation |
| PUD IT test | 23,783 | 22,783 | Evaluation only, where listed above |

EWT/ISDT counts were checked from the shared word tables; PUD counts from metadata and result rows. N excludes sentence-final words. The **50,000-row probe training cap** is separate from feature extraction: these 33 feature folders are uncapped. Seed-42 sampling is shared within a task/control pair and reused across layers. Current- and next-word tasks can select different rows. Equal scalar counts alone do not prove row identity: pairing follows the inspected runner, but selected row IDs and per-word predictions are not stored in these result files.

All probes use a training-only `StandardScaler`. Linear probes search `C=(0.01, 0.1, 1, 10)` with L-BFGS, `max_iter=1000`; the best dev classifier is retained without refitting on train+dev. MLPs use hidden size 256, ReLU, AdamW, learning rate 0.001, weight decay 0, batch size 512, at most 50 epochs and patience 5 on dev accuracy; the best weights are restored. Their saved device is `mps`. Probe/sampling seed and backbone initialization seed are 42; the independent control mapping also uses its fixed seed 42. Task and control select their own best fit under the same procedure.

### Checks and remaining provenance limits

- All 708 result rows have bounded accuracies, unique configuration/task/layer/seed keys, complete task/control pairs, equal paired `n_train`/`n_eval` and complete layer coverage.
- All saved linear fits have `converged=true`. This means the selected fit emitted no captured convergence warning; it does not certify the quality of every candidate or rule out every numerical problem.
- MLP runs completed 6–31 epochs. A linear-style convergence flag is **not applicable**; dev-selected early stopping is recorded instead. No optimizer/seed sweep or uncertainty estimate is available.
- The shared metadata records extraction code revision `ce55eaf` for English folders except `gpt2/pretrained/last/ewt_test` (`6d57fcf`); the eight Italian folders record `0a06fdd`. These are recorded abbreviated code hashes, not checkpoint revisions. Mixed extraction revisions are documented rather than silently treated as identical software.
- The current snapshot preserves the results and associated feature metadata together. However, result rows do not record the original feature commit, training-code commit, library versions, row IDs or immutable model/tokenizer revision. The model registry uses `revision: main`, and metadata does not resolve it to a checkpoint hash. The snapshot therefore establishes artifact identity, not a complete reconstruction of historical execution.

## 3. Missing or inapplicable comparisons

| Item | Status in the audited snapshot | Consequence |
| :-- | :-- | :-- |
| Layer 0, pretrained and random, linear and MLP, English models | Measured in the files above | No separate experiment/file is needed for layer 0. |
| Random-backbone evaluation on PUD | Missing | Do not extend EWT random-baseline conclusions to out-of-domain PUD. |
| Next-word random-backbone or MLP results | Missing | Current-word baseline findings do not validate next-word selectivity automatically. |
| Random Transformer blocks retaining pretrained embeddings | Literature-only; absent locally | Cannot separate the contributions of pretrained embeddings and pretrained contextual blocks as in Cao et al. |
| Majority-class classifier and explicit word-type lookup classifier | Missing as measured probing baselines | Layer 0 is an embedding reference, not a measured lookup classifier; `1/17` is not an empirical majority baseline. |
| `mean` pooling, and a complete first/last/mean baseline grid | Missing | Only GPT-2 pretrained linear has both first and last. Do not claim a general pooling winner. |
| Random probe evaluated without fitting | Not applicable | It would test an untrained classifier, not the baseline implemented here. |
| Additional seeds for these 15 files | Missing | Every claim in this audit is descriptive for seed 42. Other workstream notes about multiple seeds do not change these files. |

## 4. Measured baseline comparison

EWT test, current-word UPOS, `first` pooling, seed 42. Accuracy is a percentage; selectivity is `S = accuracy_UPOS - accuracy_control` on the 0–1 accuracy scale. Maxima are descriptive test-curve summaries, not dev-selected model rankings. The accuracy and selectivity columns can refer to different layers.

| Model / probe | Backbone | Best UPOS accuracy (layer) | Maximum S (layer) |
| :-- | :-- | :-- | :-- |
| BERT / linear | pretrained | 95.19% (6) | 0.3818 (11) |
| BERT / linear | random | 74.45% (0) | 0.0173 (11) |
| BERT / MLP | pretrained | 95.80% (8) | 0.2579 (11) |
| BERT / MLP | random | 80.14% (0) | -0.0058 (11) |
| GPT-2 / linear | pretrained | 93.44% (8) | 0.3609 (12) |
| GPT-2 / linear | random | 74.28% (0) | 0.0205 (2) |
| GPT-2 / MLP | pretrained | 93.85% (8) | 0.2506 (12) |
| GPT-2 / MLP | random | 79.90% (0) | 0.0104 (9) |

Sources are the eight `bert`/`gpt2`, `first` files in section 2, filtered to `eval_treebank=ewt`. Selectivity is recomputed from paired rows; it is not a stored accuracy column. BERT's random MLP has negative selectivity at every layer: the control is easier for that configuration, not evidence of “negative linguistic knowledge.” Its CoG is undefined because all contextual-layer selectivities become zero after the CoG-only clipping rule. Negative selectivity remains visible in the reported curves.

Random features can preserve distinguishable word identities on which a trained probe learns frequent UPOS associations. Thus, a random backbone is useful precisely because it need not score near chance. Our lexical explanation is consistent with the controls and literature; no direct word-identity decoding or context-perturbation experiment was performed locally. [EVIDENCE.md](EVIDENCE.md) distinguishes this inference from a measured mechanism.

## 5. Retrieval and verification

With an authenticated Hugging Face account, this retrieves the exact audited result files into a separate directory; it does not replace the working results:

```bash
hf download hlt-group11/features --repo-type dataset \
  --revision 7587960472da41ccea0ebab41e1249137c3988ff \
  --include 'results/probing/*.parquet' --local-dir /tmp/p7-audited-snapshot
```

Associated metadata paths are `{model}/{init}/{pooling}/{treebank}_{split}/meta.json`; tables are `data/processed/{treebank}_{split}.parquet`, at that same commit. Read only these small files for the metadata audit; full hidden-state arrays are not needed. [14_pull_probing_results.py](../../scripts/14_pull_probing_results.py) instead resolves the latest `main` and replaces local files, so it does not by itself reproduce this historical snapshot.

For the paper comparisons and presentation values, filter the downloaded parquets by the exact configuration in [EVIDENCE.md](EVIDENCE.md). The [result report](../../docs/results/ws1_probing.md) supplies the existing summary/plot commands. CoG uses layers 1..L, clips negative selectivity only for its weights, and is not an accuracy peak or an information quantity.

The documentation audit checked the saved values and metadata without retraining. Remaining validation priorities are to recover missing run provenance, repeat selected comparisons across seeds if time allows, and add lexical/majority or context-perturbation controls before making stronger claims about contextual syntax. None is silently reported as already measured.
