# Working on an issue (for people and coding agents)

## Read first, in this order
1. The issue you are working on. It lists context, dependencies, files to create, steps, edge cases and acceptance criteria.
2. [docs/CONTRACTS.md](docs/CONTRACTS.md): registries, file formats, alignment rules and the Python signatures other tasks depend on. If the issue and the contract disagree, the contract wins.
3. [PROPOSAL.md](PROPOSAL.md): the research goals and the formulas (section 5).
4. [docs/TODO.md](docs/TODO.md): who owns what, and the timeline.

## Environment
- Python 3.11. `pip install -r requirements.txt`.
- Run everything from the repository root with `PYTHONPATH=src`, for example `PYTHONPATH=src python scripts/01_extract_features.py --model gpt2 --treebank ewt`.
- Shared modules are imported as `from lib.<module> import ...`; workstream modules as `from ws1_probing.<module> import ...` and so on.
- Models are loaded only through `lib.data.load_model` with the settings from `lib.registry.get_model_spec`. All weights stay frozen.
- Device: let `load_model` choose (CUDA, then MPS, then CPU). Do not hard-code `cuda`.

## Where code goes
- Only edit the files listed in your issue. Shared code is in `src/lib/`; each workstream has its own folder (`src/ws1_probing/`, `src/ws2_compression/`, `src/ws3_crossling/`, `src/ws4_lens/`). Plots go in `src/viz/`, the demo in `app/`.
- Runnable entry points go in `scripts/`, named `NN_short_name.py` (numbers as in the issue). Scripts use `argparse`, take the registry keys as flags, and print the paths they write.
- Library code never writes outside the paths returned by `lib.paths`.
- Notebooks (`notebooks/<name>_*.ipynb`) are for exploration only; nothing a deliverable depends on may live in a notebook.

## Code style
- Follow `src/lib/data.py`: type hints, docstrings with `Parameters` and `Returns` sections, explicit errors with clear messages (`ValueError`, `FileNotFoundError`).
- Imports at the top of the file. No global state other than constants.
- `ruff check src tests scripts app` must pass (configuration in `ruff.toml`).

## Tests
- Every issue lists the tests it needs. Put them in `tests/test_<module>.py`.
- Tests use small synthetic data and must not download anything. Tests that need torch start with `torch = pytest.importorskip("torch")`; CI does not install torch and skips them, so run them locally.
- Tests that need a real model or dataset download are marked `@pytest.mark.slow` and are not run by CI.
- Run `PYTHONPATH=src pytest -q` before opening a PR.

## Pull requests
- One PR per issue, branch name `<issue-id>-<short-name>` (for example `e1-alignment`), PR description starting with `Closes #<number>`.
- Tick the acceptance criteria in the issue (or list them in the PR) and paste the command output that proves them.
- Do not commit anything under `data/`, `features/`, `results/` or `checkpoints/`. Say in the PR where large outputs were uploaded.
- Changing `docs/CONTRACTS.md` needs its own PR approved by another member.

## When something is unclear
Do not guess silently. Write the question as a comment on the issue, choose the most conservative option, and record the choice in the PR description.
