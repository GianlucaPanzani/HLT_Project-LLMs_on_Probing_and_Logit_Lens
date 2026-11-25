# HLT Group 11: Inside a Language Model, Probing and the Logit Lens

Layer-by-layer analysis of frozen Transformers with supervised probing (UPOS) and the logit/tuned/Jacobian lens.

- How to work on an issue (people and coding agents): [AGENTS.md](AGENTS.md)
- Specification: [PROPOSAL.md](PROPOSAL.md)
- Who does what and when: [docs/TODO.md](docs/TODO.md)
- File and data formats: [docs/CONTRACTS.md](docs/CONTRACTS.md)

## Layout

```
src/lib/             shared code (loaders, alignment, features, metrics); changes via PR
src/ws1_probing/     WS1 probes and baselines
src/ws2_compression/ WS2 BERT vs DistilBERT, CKA
src/ws3_crossling/   WS3 English vs Italian
src/ws4_lens/        WS4 lens spectrum
src/viz/             shared plots
scripts/             runnable entry points
configs/             model, dataset and experiment settings
app/                 Streamlit demo
notebooks/           personal exploration, prefix the file with your name
tests/               unit tests
docs/                plans and contracts
data/ features/ results/ checkpoints/   generated, not committed
```

Run code and notebooks with `PYTHONPATH=src` so that `from lib.data import ...` works.

## Setup
```
pip install -r requirements.txt
```

## Interactive demo

The Streamlit app uses sidebar multipage navigation (issue A5). Launch it with:

```bash
PYTHONPATH=src streamlit run app/app.py
```

Pages are discovered automatically from `app/pages/[0-9][0-9]_*.py`. Add a new page by creating a file there; no edit to the router is needed. See [app/PAGES.md](app/PAGES.md) for page ownership and the UI/helper separation rules.

## Workflow
Tasks are GitHub issues grouped by milestone; the overview is in [docs/TODO.md](docs/TODO.md). Open one PR per issue (`Closes #N`). CI (lint and tests) runs on PRs only.
