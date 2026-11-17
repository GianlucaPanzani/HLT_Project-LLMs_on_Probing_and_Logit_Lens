"""Status report of the feature folders expected by CONTRACTS.md section 4.

Used to follow the extraction runs of issue E6 (Italian and multilingual models) and to produce the
table that is pasted in the issue: rows, size on disk, time, NaN rows, truncated sentences and cap.
It only reads files; it never loads a model.
"""

from lib.features import load_meta
from lib.paths import feature_dir
from lib.registry import get_treebank_spec

# (model, treebank, poolings): every split of the treebank is expected (PUD only has "test").
E6_RUNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("bert_it", "isdt", ("first",)),
    ("bert_it", "pud_it", ("first",)),
    ("geppetto", "isdt", ("first", "last")),
    ("geppetto", "pud_it", ("first", "last")),
    ("mbert", "ewt", ("first",)),
    ("mbert", "isdt", ("first",)),
    ("mbert", "pud_en", ("first",)),
    ("mbert", "pud_it", ("first",)),
    ("mgpt", "ewt", ("first",)),
    ("mgpt", "isdt", ("first",)),
    ("mgpt", "pud_en", ("first",)),
    ("mgpt", "pud_it", ("first",)),
)

MGPT_TRAIN_CAP = 50_000
"""Recommended cap on mGPT train rows (docs/EXTRACTION.md section 5)."""

COLUMNS: tuple[str, ...] = (
    "run", "status", "rows", "size_gb", "seconds", "nan_rows", "truncated", "max_rows", "model_dtype", "device",
)


def _folder_size_gb(folder) -> float:
    '''
    Total size of the files of a feature folder.

    Parameters
    ----------
    - folder (Path): Feature folder.

    Returns
    -------
    float: Size in gigabytes (1e9 bytes).
    '''
    return sum(path.stat().st_size for path in folder.iterdir() if path.is_file()) / 1e9


def report_runs(
    models: tuple[str, ...] | None = None,
    runs: tuple[tuple[str, str, tuple[str, ...]], ...] = E6_RUNS,
) -> list[dict]:
    '''
    Describe the state of every expected feature folder.

    Parameters
    ----------
    - models (tuple[str, ...] | None): Keep only these registry keys. None
    keeps all of them. Unknown keys raise ValueError. Defaults to None.
    - runs (tuple): Expected runs as (model, treebank, poolings). Every split
    of the treebank is expected. Defaults to E6_RUNS.

    Returns
    -------
    list[dict]: One dict per (model, treebank, pooling, split) with the keys
    of COLUMNS. status is "done" (meta.json present), "missing" (no complete
    folder) or "check cap" (a cap that CONTRACTS.md section 4 does not allow:
    only mGPT train may be capped, and the recommended value is
    MGPT_TRAIN_CAP). The numeric columns are None for missing folders.
    '''
    known = {model for model, _, _ in runs}
    if models is not None:
        unknown = sorted(set(models) - known)
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}; the plan has {sorted(known)}.")

    report = []
    for model, treebank, poolings in runs:
        if models is not None and model not in models:
            continue
        for pooling in poolings:
            for split in get_treebank_spec(treebank)["splits"]:
                row = dict.fromkeys(COLUMNS)
                row["run"] = f"{model} {treebank}/{split} {pooling}"
                try:
                    meta = load_meta(model, "pretrained", pooling, treebank, split)
                except FileNotFoundError:
                    row["status"] = "missing"
                    report.append(row)
                    continue
                allowed_cap = model == "mgpt" and split == "train"
                capped = meta.get("max_rows") is not None
                row.update(
                    status="check cap" if capped and not allowed_cap else "done",
                    rows=meta["n_rows"],
                    size_gb=round(_folder_size_gb(feature_dir(model, "pretrained", pooling, treebank, split)), 2),
                    seconds=round(meta["seconds"]) if meta.get("seconds") is not None else None,
                    nan_rows=meta["n_nan_rows"],
                    truncated=meta["n_truncated_sentences"],
                    max_rows=meta.get("max_rows"),
                    model_dtype=meta.get("dtype_model"),
                    device=meta.get("device"),
                )
                report.append(row)
    return report


def format_markdown(report: list[dict]) -> str:
    '''
    Format a report as a Markdown table plus a one-line summary.

    Parameters
    ----------
    - report (list[dict]): Output of report_runs.

    Returns
    -------
    str: The table (header, separator, one line per run) and a final line
    with the number of complete runs, the total size and the number of NaN rows.
    '''
    def cell(value) -> str:
        return "" if value is None else str(value)

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + " --- |" * len(COLUMNS)]
    lines += ["| " + " | ".join(cell(row[column]) for column in COLUMNS) + " |" for row in report]
    done = [row for row in report if row["status"] != "missing"]
    total = sum(row["size_gb"] for row in done)
    nan_rows = sum(row["nan_rows"] for row in done)
    lines.append("")
    lines.append(f"{len(done)} of {len(report)} runs have a meta.json, {total:.1f} GB, {nan_rows} NaN rows in total.")
    return "\n".join(lines)
