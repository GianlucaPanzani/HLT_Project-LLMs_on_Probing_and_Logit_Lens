# Team split, timeline and tasks

Spec: [PROPOSAL.md](../PROPOSAL.md). Formats and interfaces: [CONTRACTS.md](CONTRACTS.md). How to work on an issue: [AGENTS.md](../AGENTS.md).
Every task below is a GitHub issue (same ID in the title, for example `[P3]`). Each issue is self-contained: context, dependencies, files, interface, steps, edge cases and acceptance criteria. This file is the overview.

Dependency order: G0 -> D1 -> D2 -> E1/E2/E3 -> E4/E6 -> probing, CKA, cross-linguistic. The lens (L1, L2) only needs G0 and can start on day 1.

Rules:
- Each person edits only their own folders (section 1). Changes to `src/lib/` go through a small PR reviewed by one other member.
- Open one PR per task, with `Closes #N`. CI runs on PRs only (see section 5).
- When you finish your tasks, take a task labelled `help wanted` (section 3) and tell the owner in the issue.

## 1. Ownership

| Person | Workstream | Owns | Availability |
| :-- | :-- | :-- | :-- |
| Marcin Czarnecki | E: shared extraction (English models) | `src/lib/alignment.py`, `src/lib/features.py`, `src/lib/control_tasks.py`, `scripts/01_extract_features.py` | Nov 13-16 only |
| Gianluca Panzani | G0 registry, then P: probes, probing metrics, GPT-2 vs BERT | `src/lib/registry.py`, `src/lib/data.py`, `src/lib/metrics.py`, `src/ws1_probing/`, `configs/models.yaml`, `configs/datasets.yaml` | Full |
| Filippo Gigante | C: CKA and compression, plus data preparation | `src/ws2_compression/`, `src/lib/paths.py`, `src/lib/word_table.py`, `scripts/00_download_data.py`, `scripts/02_build_word_tables.py` | Full |
| Matteo Salvini | L: lens spectrum and probing bridge | `src/ws4_lens/` | Full |
| Arcangelo Chine | X: cross-linguistic, plus Italian and multilingual extraction | `src/ws3_crossling/` | Full |
| All | `src/viz/`, `app/`, `docs/` | Shared, through PRs | Phase 2 and 3 |

Why:
- Marcin leaves after Nov 16, so he only gets the pipeline everyone is blocked on. He ends with a handoff (E5).
- Filippo's CKA task is the lightest early on, so he also prepares the data (D1, D2), which is needed first.
- Matteo has the lens, the heaviest workstream, split into small tasks so that others can join.

## 2. Timeline (GitHub milestones)

| Milestone | Due | Contents |
| :-- | :-- | :-- |
| M0 Foundation | Nov 14 | G0 (Nov 13 morning), D1, D2, E1, E2, E3, P1, L1, X1 |
| M1 Features ready | Nov 16 | E4, E5, X1 (all extraction runs done) |
| M2 Phase 1 results | Nov 18 | P2-P6, C1-C4, L2-L6, X2-X4 first results |
| M3 Phase 2 plots | Nov 25 | cross-analysis plots, remaining runs, X5, L7, L8 |
| M4 Demo and report | Dec 3 | A1-A5, R1 |

## 3. Tasks

`HW` marks long tasks that are split on purpose and where a free member should join.

### Foundation and data (first two days)
| ID | Task | Owner | Due |
| :-- | :-- | :-- | :-- |
| G0 | Model and treebank registry, `UPOS_TAGS`, `requirements.txt` (blocks everything, merge first) | Gianluca | Nov 13 |
| D1 | Download script for EWT, ISDT, PUD EN and IT (`scripts/00_download_data.py`) | Filippo | Nov 13 |
| D2 | Word tables (parquet) and `lib/paths.py`, per CONTRACTS.md sections 2 and 10 | Filippo | Nov 14 |

### Extraction (Marcin, with Arcangelo for the rest)
| ID | Task | Owner | Due |
| :-- | :-- | :-- | :-- |
| E1 | Subword to word alignment, first and mean pooling, edge cases | Marcin | Nov 14 |
| E2 | Feature extraction engine (batching, layers 0..L, one file per layer, `meta.json`) | Marcin | Nov 14 |
| E3 | Control task labels (fixed seed, shared across models) | Marcin | Nov 14 |
| E4 | Run extraction for BERT, DistilBERT, GPT-2, pretrained and random (HW: Filippo can run part) | Marcin | Nov 16 |
| E5 | Handoff note `docs/EXTRACTION.md` and merge everything | Marcin | Nov 16 |
| E6 | Run extraction for Italian BERT, GePpeTto, mBERT, mGPT (fp16) (HW) | Arcangelo | Nov 17 |

### Probes (Gianluca)
| ID | Task | Owner | Due |
| :-- | :-- | :-- | :-- |
| P1 | Probing metrics: `sel_acc`, Macro-F1 `sel_f1`, `sel_weighted`, unchanged accuracy-based CoG, with tests | Gianluca | Nov 14 |
| P2 | Linear probe (scikit-learn, C chosen on dev) | Gianluca | Nov 16 |
| P3 | MLP probe, early stopping (HW) | Gianluca | Nov 17 |
| P4 | Probe runner: all layers, UPOS/control and baselines; dev/test scores, paired selectivity outputs and dev weighted selection per CONTRACTS (HW) | Gianluca | Nov 17 |
| P5 | Next-token UPOS task (`upos_next`) | Gianluca | Nov 17 |
| P6 | GPT-2 vs BERT comparison and plots | Gianluca | Nov 18 |

### CKA and compression (Filippo)
| ID | Task | Owner | Due |
| :-- | :-- | :-- | :-- |
| C1 | Linear CKA function (float64, centred) with tests | Filippo | Nov 16 |
| C2 | Aligned token sampling for BERT and DistilBERT | Filippo | Nov 16 |
| C3 | Run the 13x7 CKA matrix and heatmap | Filippo | Nov 17 |
| C4 | 1:2 layer mapping comparison of accuracy, selectivity and CoG (needs P4) | Filippo | Nov 18 |

### Lens (Matteo)
| ID | Task | Owner | Due |
| :-- | :-- | :-- | :-- |
| L1 | Standard logit lens for GPT-2 and GePpeTto, with the check that layer L equals the model logits | Matteo | Nov 14 |
| L2 | Lens metrics: entropy, KL, top-1 (stable numerics) with tests | Matteo | Nov 15 |
| L3 | Tuned lens GPT-2 from the official checkpoint | Matteo | Nov 16 |
| L4 | Tuned lens GePpeTto training on ISDT train (HW) | Matteo | Nov 18 |
| L5 | Jacobian lens (HW) | Matteo | Nov 18 |
| L6 | UPOS lexicon for the top-1 token to UPOS bridge | Matteo | Nov 17 |
| L7 | Run all lenses on PUD and WikiText, write aggregate and per-token results | Matteo | Nov 25 |
| L8 | Overlay of lens curves and next-token probing selectivity (needs P5) | Matteo | Nov 25 |

### Cross-linguistic (Arcangelo)
| ID | Task | Owner | Due |
| :-- | :-- | :-- | :-- |
| X1 | Italian specifics: contractions, apostrophes, multiword tokens | Arcangelo | Nov 15 |
| X2 | mBERT probing on EN and IT (needs E6, P4) | Arcangelo | Nov 18 |
| X3 | mGPT probing with normalisation by tau, CoG and selectivity (HW) | Arcangelo | Nov 18 |
| X4 | Paired models: BERT vs Italian BERT, GPT-2 vs GePpeTto | Arcangelo | Nov 18 |
| X5 | Delta CoG and cross-linguistic plots | Arcangelo | Nov 25 |

### Demo and report (everyone, Nov 26 to Dec 3)
| ID | Task | Owner | Due |
| :-- | :-- | :-- | :-- |
| A1 | Streamlit app shell and Tab 1 lens heatmap (live lens on user text) | Matteo | Dec 3 |
| A2 | Tab 2 probing explorer | Gianluca | Dec 3 |
| A3 | Tab 3 CKA explorer | Filippo | Dec 3 |
| A4 | Tab 4 cross-linguistic dashboard | Arcangelo | Dec 3 |
| R1 | Report and slides (sections by workstream) | All | Dec 3 |

### P1 Macro-F1 extension: implementation and dependent tasks

The revised schemas and interfaces in [CONTRACTS.md](CONTRACTS.md), sections 6, 7 and 10, are implementation targets. Document changes alone do not complete these tasks. Approve the contract update before integrating dependent code; retain the separate contract PR/review requirement.

- **P1:** Implement Macro-F1 over all 17 tags (`zero_division=0`), `sel_f1`, `sel_weighted` with fixed `alpha=0.5` by default, and task/control pairing. Keep `selectivity(...)` compatible and all existing CoG/peak statistics based on `sel_acc`. Add explicit split filtering to `summarize_probing`.
- **P2/P3:** Preserve the declared internal tuning/early-stopping criterion consistently for task and control (P2: dev accuracy). Supply predictions from the selected probes; weighted configuration selection belongs to P4 and must never choose a deliberately weaker control.
- **D2/P4:** Add the documented paths for derived selectivity and selection records. Persist raw `acc`, `f1_macro`, class support and `eval_split`; save paired `sel_acc`, `sel_f1`, `sel_weighted` and `alpha`. Rank configurations on dev by averaging the same seeds and all layers `1..L`, preserving candidate order on ties. Require complete coverage/common evaluation rows and record candidates, scores and the chosen configuration before test evaluation.
- **P5:** Apply the new outputs to `upos_next/control_next`, using identical valid-row masks for both tasks, including missing next-word targets.
- **P6/C4:** Read the new split-aware results, compare all three selectivities and raw scores, and keep CoG and the BERT/DistilBERT 1:2 mapping unchanged.
- **X2/X3/X4/X5:** Use the updated runner and regenerate evaluations where F1 is missing. Show tag supports and separate dataset/language comparisons; delta CoG remains based on `sel_acc`.
- **L8:** Overlay each next-token selectivity with lens curves for the matching dataset and split; identify the metric and `alpha` explicitly.
- **A2/A3/A4:** Add metric selection, display `alpha`, distinguish dev from test and handle unavailable legacy F1 values. CoG markers remain accuracy-based.
- **R1:** Explain the Macro-F1 extension, fixed weighting, dev-only selection and remaining class-distribution limitations; report the raw component scores alongside selectivity and CoG.

Feature extraction, control-label generation, CKA computation and lens metrics require no algorithmic change. Legacy accuracy-only results remain valid for `sel_acc`/CoG; F1 requires saved predictions/confusion matrices or reevaluation, never zero-filled placeholders.

Acceptance checks for the extension:

- [ ] Synthetic imbalanced data demonstrates different Macro-F1 values at equal accuracy; absent tags and all 17 labels follow the contract.
- [ ] Formula tests cover negative selectivity, alpha endpoints/default, invalid weights/shapes/nonfinite scores, and preserve existing CoG results for identical accuracy inputs.
- [ ] Pairing tests cover duplicate/missing rows, shared masks, split separation and legacy inputs; incomplete candidates cannot enter selection.
- [ ] Runner tests verify that changing test results cannot change dev selection, and that result/selection artifacts preserve the required metrics and metadata.
- [ ] Affected consumers and their tests use explicit metric names and splits; Ruff and offline tests pass.

## 3b. Who helps whom
- Marcin's E4 can be split by model, so Filippo (after D2) runs DistilBERT and the random-init runs while Marcin runs BERT and GPT-2.
- Gianluca's P3 and P4 are the longest probing tasks: Filippo (after C3) takes P3 or part of P4.
- Matteo's L4 and L5 are the longest lens tasks: Filippo takes L5 (Jacobian) once CKA is done.
- Arcangelo's X3 (mGPT) is memory heavy: whoever has a GPU takes the extraction.

## 4. Marcin handoff (E5, due Nov 16)
- [ ] All code merged to `main`; nothing left on a local branch.
- [ ] `docs/EXTRACTION.md`: how to run, how long it takes, what is done, known problems.
- [ ] Features for all English models are on the team's shared storage (say where).
- [ ] One command re-runs extraction for one model.

## 5. CI and GitHub setup
- CI runs on pull requests to `main` only, only when code or CI files change, cancels superseded runs and has a 5 minute limit. It runs `ruff` and `pytest -m "not slow"`, without installing torch.
- Tests that need torch must start with `pytest.importorskip("torch")`, so they are skipped in CI and run locally. Tests that download models or data are marked `@pytest.mark.slow` (see `pytest.ini`).
- Results write-ups live in `docs/results/` (one file per workstream) and feed the report (R1).
- Everyone runs `ruff check src tests scripts app` and `PYTHONPATH=src pytest -q` before opening a PR.

## 6. Decisions to make as a team
- Where do large files (features, checkpoints) live? They are gitignored, so we need a shared drive.
- First-subword or mean pooling for the main results? Default: first subword, mean as an ablation.
- Negative selectivity: preserve it in `sel_acc`, `sel_f1` and `sel_weighted`; keep the existing default clipping only for accuracy-based CoG. The function's optional unclipped mode remains available.
