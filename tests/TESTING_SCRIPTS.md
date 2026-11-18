# Testing scripts

Run these scripts from the **repository root**, with the project's Python environment activated and dependencies installed. Both scripts set `PYTHONPATH=src` for pytest.

## `tests/test.sh`

Runs one test file with `python -m pytest -q`, then runs `ruff check src tests scripts app`.

```bash
bash tests/test.sh tests/test_probes.py
```

- Requires exactly one argument: an existing test file.
- Runs Ruff even when pytest fails.
- Does not exclude tests marked `slow` in the selected file.
- Invalid arguments or a missing file produce exit status 1.
- After running both checks, the script returns **Ruff's exit status**. A successful exit therefore does not guarantee that pytest passed; inspect both outputs.

## `tests/test_all.sh`

Runs the automatically discovered pytest suite. Accepts at most one optional argument:

| Command | Behavior |
| :-- | :-- |
| `bash tests/test_all.sh` | Runs `python -m pytest -q`, including slow tests. |
| `bash tests/test_all.sh --slow` | Same as the default; runs all tests, not just slow tests. |
| `bash tests/test_all.sh --no-slow` | Runs `python -m pytest -q -m "not slow"`. |

- Slow tests may download models and require local datasets or generated word tables. These scripts do not prepare those prerequisites.
- Returns pytest's exit status; invalid arguments produce exit status 1.
- Does not run Ruff. Run `ruff check src tests scripts app` separately when needed.

## P2 benchmark

`tests/tests_probing_fit_benchmark.py` deliberately uses a `tests_` prefix, so it is not automatically discovered by `test_all.sh`, including with `--slow`. Run it explicitly to display the fit timing and environment details for the P2 PR:

```bash
PYTHONPATH=src python -m pytest -q -s tests/tests_probing_fit_benchmark.py
```

The benchmark generates synthetic data and times the full `LinearProbe.fit` search over its default C candidates. Data generation and prediction are excluded from the reported fit time.
