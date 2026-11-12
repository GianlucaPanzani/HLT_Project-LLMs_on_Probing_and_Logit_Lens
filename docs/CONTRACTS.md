# Data and file contracts

These formats are frozen after Nov 14. All paths are relative to the repository root. `data/`, `features/`, `results/` and `checkpoints/` are gitignored.
Model keys: `bert`, `distilbert`, `gpt2`, `bert_it`, `geppetto`, `mbert`, `mgpt` (Hugging Face ids in `configs/models.yaml`).
Treebank keys: `ewt`, `isdt`, `pud_en`, `pud_it`.

## 1. Word table
`data/processed/{treebank}_{split}.parquet`, one row per UD word (multiword token ranges are skipped, only syntactic words).

| Column | Type | Note |
| :-- | :-- | :-- |
| sent_id | str | |
| word_idx | int | position within the sentence, starts at 0 |
| form | str | |
| upos | str | one of the 17 UPOS tags |
| next_upos | str or null | null for the last word of a sentence |
| lang | str | `en` or `it` |

## 2. Features
`features/{model}/{init}/{treebank}_{split}/layer_{l:02d}.pt`
- `init` is `pretrained` or `random`.
- Tensor of shape `[N, d]`, fp16. Row `i` corresponds to row `i` of the word table for that treebank and split.
- A `meta.json` in the same folder records: model id, pooling (`first` or `mean`), dtype, number of layers L, dimension d, number of truncated sentences, git commit.
- Layers go from 0 (embeddings) to L.

## 3. Control labels
`data/processed/control_{treebank}.json`: mapping from word form to UPOS label, with the seed. Shared by every model and layer.

## 4. Probing results
`results/probing/{model}__{treebank}__{probe}__{init}.parquet`, one row per layer and seed.

| Column | Type | Note |
| :-- | :-- | :-- |
| model, treebank | str | |
| lang | str | |
| arch | str | `encoder` or `decoder` |
| probe | str | `linear` or `mlp` |
| task | str | `upos`, `control` or `upos_next` |
| init | str | `pretrained` or `random` |
| layer | int | 0..L |
| L | int | number of layers |
| tau | float | layer / L |
| acc | float | accuracy on test split |
| seed | int | |

Selectivity is not stored. Compute it with `src/lib/metrics.py` (Gianluca) as `acc(upos) - acc(control)`.

## 5. CKA
`results/cka/{model_a}__vs__{model_b}.parquet`, long format with columns `layer_a`, `layer_b`, `cka`. For BERT against DistilBERT: 13 x 7 rows.

## 6. Lens results
Aggregate: `results/lens/{model}__{dataset}.parquet`, columns `lens` (`logit`, `tuned`, `jacobian`), `layer`, `L`, `tau`, `top1_acc`, `entropy_mean`, `kl_mean`, `upos_top1_acc`.

Per token (used by the demo): `results/lens/tokens/{model}__{dataset}.parquet`, columns `sent_id`, `pos`, `token`, `lens`, `layer`, `top1_token`, `entropy`, `kl`.

## 7. Summary
`results/summary/cog.parquet`: columns `model`, `treebank`, `probe`, `init`, `cog`, `cog_norm`, `peak_layer`.

## 8. Checkpoints
`checkpoints/tuned_lens/{model}.pt`, produced by Matteo.

## 9. Conventions
- Seeds: default 42, always stored in the results.
- Use the test split for all reported numbers, dev for choosing hyperparameters, train for fitting.
- Every result file written by a script should be reproducible from `configs/` plus the script.
