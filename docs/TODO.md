# Team split, timeline and tasks

Spec: [PROPOSAL.md](../PROPOSAL.md). Formats and interfaces: [CONTRACTS.md](CONTRACTS.md). How to work on an issue: [AGENTS.md](../AGENTS.md).
Every task below is a GitHub issue (same ID in the title, for example `[P3]`). Each issue is self-contained: context, dependencies, files, interface, steps, edge cases and acceptance criteria. This file is the overview.

**Current execution plan:** [section 7: final sprint, 24 November to 3 December 2026](#7-final-sprint-24-november-to-3-december-2026). It supersedes the earlier dates and remaining-work assumptions below. Sections 1–6 preserve the original project plan.

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
| P1 | Probing metrics in `src/lib/metrics.py` with tests | Gianluca | Nov 14 |
| P2 | Linear probe (scikit-learn, C chosen on dev) | Gianluca | Nov 16 |
| P3 | MLP probe, early stopping (HW) | Gianluca | Nov 17 |
| P4 | Probe runner: all layers, UPOS and control, pretrained, layer 0 and random baselines, writes CONTRACTS results (HW) | Gianluca | Nov 17 |
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
- Negative selectivity in CoG: clip to zero (default) or leave as is.

## 7. Final sprint: 24 November to 3 December 2026

Presentation: **26 November**. Final demo and report: **3 December**. Dates use Europe/Rome. Freeze presentation content on **25 November evening**, finish required feature work by **1 December**, and reserve **2 December** for integration/rehearsal. On presentation/delivery days, do smoke checks and small fixes only.

This schedule was checked against repository revision `2cb85bc`, the contract, proposal, result notes, notebooks and GitHub issue history on 24 November. Estimates are active work, excluding transfers and model runs; benchmark missing runs before committing compute time. No new work is assigned to Marcin.

### 7.1 What is already done, and what needs finishing

- Gianluca owns probing: P1–P6 and A2 are closed. Linear/MLP and pretrained/random comparisons already have measured results, so the presentation can reuse them.
- Filippo owns CKA: C1–C4 are closed, but `app/tabs/cka.py` is still a placeholder and `docs/results/ws2_compression.md` contains placeholder values. Existing **A3 (#34)** is updated to finish this work, including accuracy/control/selectivity.
- Arcangelo owns multilingual analysis: X1–X5 are closed. Commit `45aceae` already implements the dashboard; existing **A4 (#35)** is updated for verification and multipage integration rather than duplicated.
- Matteo owns lens analysis and the new router. L1–L8 and A1 are closed; Arcangelo also contributed the existing A1 implementation. Existing plotting code can power the UPOS/token page without running notebooks.
- Existing **R1 (#36)** is rescheduled for both deadlines. Twelve new issues are added; A3/A4/R1 remain the original issues and are not closed merely because code exists.
- Locally, probing parquets are present, while CKA/lens results and lens checkpoints are absent. This is a local inventory, not evidence that shared artifacts do not exist. Owners must confirm their shared-storage locations before screenshots or inference.

### 7.2 Issue schedule and priority

P0 = start immediately for the presentation; P1 = finish the core presentation/demo; P2 = required after the presentation; P3 = optional and the first work to drop. “Required” applies to the stated milestone slice, not necessarily the issue's complete final scope.

| Issue | Owner | Priority | Due / useful deliverable | 26 Nov | 3 Dec | New dependencies |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| [A5 #79 — Sidebar multipage app](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/79) | Matteo | P0 | 24 Nov: one router and four existing pages | Required | Required | None |
| [P7 #80 — Probing baselines and literature](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/80) | Gianluca | P0 | 25 Nov: main evidence; 1 Dec: final audit | Required | Required | None |
| [C5 #81 — CKA baselines and literature](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/81) | Filippo | P0 | 25 Nov: main evidence; 1 Dec: final audit | Required | Required | A3 for final measured report only |
| [X6 #82 — Multilingual baselines and literature](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/82) | Arcangelo | P0 | 25 Nov: main evidence; 1 Dec: final audit | Required | Required | None to start; incorporate X7 later |
| [L9 #83 — Lens baselines and literature](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/83) | Matteo | P0 | 25 Nov: supported claims; 1 Dec: final audit | Required | Required | None |
| [P8 #84 — Linear versus MLP page](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/84) | Gianluca | P1 | 25 Nov: existing results on a dedicated page | Required | Required | A5 for navigation check only |
| [A3 #34 — CKA mappings and three metrics](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/34) | Filippo | P0 | 25 Nov: measured table/figure; 30 Nov: polish | Required | Required | A5 for navigation check only |
| [A4 #35 — Validate existing EN/IT dashboard](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/35) | Arcangelo | P1 | 25 Nov: accepted dashboard and evidence | Required | Required | A5 for navigation check only |
| [L10 #86 — UPOS/token commitment page](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/86) | Matteo | P1 | 25 Nov: existing overlays; 30 Nov: controls | Required | Required | A5 for navigation check only |
| [P9 #85 — Pooling comparison](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/85) | Gianluca | P2 | 30 Nov: BERT/GPT-2 first/last/mean results and page | Optional | Required | A5 for navigation; schedule after P8 |
| [G1 #87 — mGPT lens contract extension](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/87) | Arcangelo; Matteo reviews | P2 | 27 Nov: separate contract-only PR | Optional | Required for X7 | None |
| [X7 #88 — mGPT UPOS/token stabilization](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/88) | Arcangelo | P2 | 1 Dec: bounded EN/IT logit-lens diagnostic and page | Optional | Required | G1; A5 for navigation |
| [A6 #89 — Optional prediction lab](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/89) | All; Arcangelo UI | P3 | 2 Dec, only if required work is on track | Optional | Optional | A5; reference panels degrade independently |
| [A7 #90 — Artifact checks and demo rehearsal](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/90) | Filippo coordinates; all verify | P1 | Inventory by 25 Nov; full rehearsal 2 Dec | Optional as separate issue | Required | Required pages for final sign-off only |
| [R1 #36 — Presentation and final report](https://github.com/GianlucaPanzani/HLT_Project-LLMs_on_Probing_and_Logit_Lens/issues/36) | All; Gianluca assembles slides | P0/P1 | 25 Nov: slides; 3 Dec: final report | Required | Required | Evidence for slides; final analyses for report |

### 7.3 Parallel work by person

| Window | Gianluca | Matteo | Filippo | Arcangelo |
| :-- | :-- | :-- | :-- | :-- |
| 24 Nov | P7: inventory and main literature comparisons | A5 first; begin L9 | C5 and locate A3 CKA artifacts | X6; inspect existing A4 acceptance |
| 25 Nov | Finish P7 presentation slice; P8; assemble R1 outline | L9 presentation slice; L10 minimal page; R1 contribution | A3 measured mapping/metrics; C5 presentation slice; A7 inventory; R1 contribution | Finish X6 presentation slice and A4 verification; R1 contribution |
| 26 Nov | Presentation and smoke checks | Presentation and smoke checks | Presentation and smoke checks | Presentation and smoke checks |
| 27–30 Nov | P9: benchmark missing runs, complete matched pooling results/page | Finish L10; review G1; finish L9 audit | Polish A3; finish C5; prepare A7 runbook | G1 on 27 Nov, then X7 small EN/IT run and page |
| 1 Dec | P7 final audit; pooling findings into R1 | L9 final audit and R1 | C5 final audit and R1 | Finish X7/X6 and R1 |
| 2 Dec | Own-page verification; final report | Own-page verification; final report | Coordinate A7 rehearsal and backups | Own-page verification; final report; optional A6 only if on track |
| 3 Dec | Final smoke check and delivery | Final smoke check and delivery | Final smoke check and delivery | Final smoke check and delivery |

Before the presentation, allow roughly 5–7 active hours for Gianluca, 6–9 for Matteo, 6–8 for Filippo and 4–6 for Arcangelo, plus artifact transfers. These are planning estimates, not assumed availability. The 28–29 November weekend can be used for unattended runs if available; it has no required meeting or additional feature milestone. If time is tight, drop A6 and post-presentation polish first, retaining the evidence and minimal pages.

### 7.4 Interfaces and conflict avoidance

- A5 puts the UI body in `app/pages/NN_*.py`: Streamlit widgets, layout, captions, displayed errors, charts/tables and download buttons. Pages import reusable loading, validation, alignment and calculation functions from `app/tabs/`; these helpers return data, figures or status information (or raise explicit errors) without rendering the UI. Move existing UI bodies out of `tabs.render()` into their page files, preserving scientific behavior. Reusable scientific plotting stays in `src/viz/`.
- `app/app.py` keeps global page configuration, discovers sorted page filenames without importing every page, and registers file paths with `st.Page` / `st.navigation(position="sidebar")`. Only the selected page runs; each owner can add a page without editing the router. Helper imports must not trigger UI, model loading or data processing. The one launch command remains `PYTHONPATH=src streamlit run app/app.py`. See [app/PAGES.md](../app/PAGES.md) for ownership, loading rules and widget conventions.
- Test helper functions with synthetic inputs and test UI behavior by executing the page file. A5 covers lazy routing and missing-artifact behavior; the UI move does not change scientific APIs or result formats.
- Page ownership: `01_Lens_Heatmap.py` Matteo, `02_Probing.py` Gianluca, `03_Compression.py` Filippo, `04_Crossling.py` Arcangelo; new pages `05_Linear_vs_MLP.py` and `06_Pooling.py` Gianluca, `07_UPOS_and_Tokens.py` Matteo, `08_mGPT_UPOS_and_Tokens.py` Arcangelo, optional `09_Prediction_Lab.py` Arcangelo.
- The four baseline/evidence issues edit separate workstream notes. P7 owns probing interpretation; L9 owns lens interpretation; X6 owns crossling interpretation. Their page issues do not compete for the same result write-up.
- A3 generates the compression report before C5 edits its interpretation, since `scripts/21_compression_report.py` currently overwrites it. Gianluca sequences P8 then P9 if both need `src/viz/probing_plots.py`.
- G1 changes only `docs/CONTRACTS.md` in a separately reviewed PR; X7 starts after it merges. All other mandatory work reuses existing scientific signatures and formats.
- The optional A6 issue puts UI composition in `app/pages/09_Prediction_Lab.py`, reusable orchestration in `app/tabs/prediction_lab.py`, and the four contributions' supporting functions in owner-specific `app/tabs/lab_panels/` modules. Panel helpers return data, figures or status information; widgets and display code stay in the page. Private function signatures remain free because no other issue consumes them; existing cross-issue APIs are preserved. It has no required downstream consumer, and no new serialized probe format or training pipeline is assumed.

The only new scientific prerequisite chain is **G1 → X7**. **A5 → page smoke checks** is an integration dependency, not a reason to postpone page development. Evidence audits run in parallel. R1 can use reproducible static figures even when a page is not ready; A7 assembles the completed required pages at the end.

### 7.5 Required evidence and bounded scope

| Owner | Baseline inventory to create | Literature comparison to create |
| :-- | :-- | :-- |
| Gianluca | `src/ws1_probing/BASELINES.md` | `src/ws1_probing/EVIDENCE.md` |
| Filippo | `src/ws2_compression/BASELINES.md` | `src/ws2_compression/EVIDENCE.md` |
| Arcangelo | `src/ws3_crossling/BASELINES.md` | `src/ws3_crossling/EVIDENCE.md` |
| Matteo | `src/ws4_lens/BASELINES.md` | `src/ws4_lens/EVIDENCE.md` |

Each issue requires measured/missing/literature-only baseline status and an evidence table linking local artifacts to primary sources, with matching conditions and discrepancies. At least two sources and two principal findings are due on 25 November; the complete audit is due on 1 December. A paper using another architecture or task supplies qualitative context, not numerical replication.

Presentation safeguards follow the actual result notes: distinguish accuracy peaks from selectivity peaks and CoG; qualify BERT's access to future context; report EN/IT evaluation-set reversals; distinguish subword accuracy from whole-word accuracy and UPOS bridge coverage; treat 90%-of-maximum crossing as a descriptive proxy rather than persistent stability. L9 specifically audits the overly broad universal/causal conclusions and Jacobian attribution in the current lens write-up.

P9 is limited to pretrained BERT/GPT-2, a linear probe, current-word UPOS/control and first/last/mean pooling. X7 follows the user's clarified request for **UPOS/token stabilization across mGPT layers**, using a small reproducible EN/IT sample and standard logit lens only; a full tuned/Jacobian or next-UPOS probe campaign is not scheduled.

A6 proposes a final-token completion UI with a live lens heatmap, layer slider or agreement marker, and owner-supplied probing/compression/multilingual reference panels. Existing probing parquets contain scores, not a deployable bank of probes: dataset reference curves must not be presented as sentence-specific probing. Likewise, the BERT/DistilBERT CKA matrix is a named compression reference, not CKA of the selected GPT decoder. Middle-token infilling and new training are outside this optional slice.

The final artifact manifest and launch/rehearsal instructions belong to A7 (`docs/demo/artifacts.csv`, `docs/demo/RUNBOOK.md`). Exact acceptance checks, allowed files and any cross-issue interfaces are in the linked GitHub issues. Existing M3/M4 milestones are reused; the explicit two-deadline requirements in each issue govern partial presentation readiness versus final completion.
