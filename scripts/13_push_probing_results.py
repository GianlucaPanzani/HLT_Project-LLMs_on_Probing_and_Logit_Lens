"""Upload run_probing result files to the team's Hugging Face dataset.

Examples:
    PYTHONPATH=src python scripts/13_push_probing_results.py
    PYTHONPATH=src python scripts/13_push_probing_results.py --results-dir /path/to/results

Reads <results-dir>/probing/*.parquet and uploads them in one commit to
hlt-group11/features, under results/probing/, following CONTRACTS.md section 6.
Existing files with the same name are updated; other remote files are kept.
The summary produced by 11_summarize_probing.py is not part of this upload.
Authentication uses the saved Hugging Face login or HF_TOKEN, with write access.
"""

import argparse
import io
from pathlib import Path

import pandas as pd

from lib.paths import ROOT, probing_result_path

try:
    from huggingface_hub import CommitOperationAdd, HfApi
except ImportError:
    CommitOperationAdd = HfApi = None


HF_REPO_ID = "hlt-group11/features"
REMOTE_PROBING_DIR = "results/probing"
JOB_COLUMNS = ("model", "train_treebank", "probe", "init", "pooling")
RESULT_COLUMNS = (
    "model", "train_treebank", "eval_treebank", "lang", "kind", "probe", "task",
    "init", "pooling", "layer", "L", "tau", "acc", "n_train", "n_eval", "hparams", "seed",
)


def _read_results(results_dir: Path) -> list[tuple[str, bytes]]:
    """Read a fixed snapshot of the local probing parquets before uploading.

    Parameters
    ----------
    results_dir : Path
        Results root containing the probing subdirectory.

    Returns
    -------
    list[tuple[str, bytes]]
        Remote paths and original parquet bytes, sorted by filename.
        Reading the bytes once ensures validation and upload use the same data.

    Raises
    ------
    FileNotFoundError
        If the results directory or its probing parquet files are missing.
    ValueError
        If a parquet is unreadable, empty, lacks result columns, contains
        multiple jobs, or has a filename inconsistent with its job metadata.
    """
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}.")

    # Only collect runner outputs, excluding summaries, figures and other tasks.
    paths = sorted((results_dir / "probing").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No probing parquet files in {results_dir / 'probing'}.")

    results = []
    for path in paths:
        content = path.read_bytes()
        try:
            frame = pd.read_parquet(io.BytesIO(content))
        except (ValueError, OSError) as error:
            raise ValueError(f"Cannot read probing result {path}; it may still be being written.") from error

        # Validate the result structure and reuse the contracted filename helper.
        missing = sorted(set(RESULT_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing probing result columns in {path}: {missing}.")
        if frame.empty:
            raise ValueError(f"Empty probing result: {path}.")
        jobs = frame[list(JOB_COLUMNS)].drop_duplicates()
        if len(jobs) != 1 or jobs.isna().any().any():
            raise ValueError(f"Expected one job with non-null metadata in {path}.")
        expected_name = probing_result_path(*jobs.iloc[0].tolist()).name
        if path.name != expected_name:
            raise ValueError(f"Unexpected result filename {path.name}; expected {expected_name}.")

        results.append((f"{REMOTE_PROBING_DIR}/{path.name}", content))

    return results


def main(argv: list[str] | None = None) -> None:
    """Validate local results, upload them together and print their commit URL.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments; None reads them from the current process.

    Returns
    -------
    None
        Prints the uploaded paths and the immutable commit reference to share.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results",
        help="Results root containing probing/ (default: the repository's results/).",
    )
    args = parser.parse_args(argv)

    # Finish all local reads before contacting HF, so malformed inputs fail early.
    results = _read_results(args.results_dir.expanduser())
    if HfApi is None or CommitOperationAdd is None:
        raise ImportError("huggingface_hub is required. Install it with: pip install huggingface_hub")

    # A single commit publishes all selected files; no deletion operations are sent.
    operations = [
        CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)
        for path, content in results
    ]
    print(f"Uploading {len(results)} probing result files to {HF_REPO_ID}...", flush=True)
    commit = HfApi().create_commit(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision="main",
        operations=operations,
        commit_message=f"Upload {len(results)} probing result files",
    )

    for path, _ in results:
        print(f"Uploaded: {path}")
    print(f"Commit: {commit.oid}")
    print(f"Share: {commit.commit_url}")


if __name__ == "__main__":
    main()
