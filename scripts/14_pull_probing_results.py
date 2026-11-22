"""Download probing results from the team's Hugging Face dataset.

Examples:
    PYTHONPATH=src python scripts/14_pull_probing_results.py
    PYTHONPATH=src python scripts/14_pull_probing_results.py --results-dir /path/to/results

Reads results/probing/*.parquet from hlt-group11/features and writes them to
<results-dir>/probing/, following CONTRACTS.md section 6. The latest main
revision is resolved once to a commit, shared by the listing and every read.
Existing files with the same name are replaced; other local files are kept.
The summary is not downloaded: regenerate it with 11_summarize_probing.py
after pulling into the default results directory.
Authentication uses the saved Hugging Face login or HF_TOKEN.
"""

import argparse
import io
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import pandas as pd

from lib.paths import ROOT, probing_result_path
from lib.remote_features import HfApi, HfRangeSource

HF_REPO_ID = "hlt-group11/features"
REMOTE_PROBING_DIR = "results/probing"
JOB_COLUMNS = ("model", "train_treebank", "probe", "init", "pooling")
RESULT_COLUMNS = (
    "model", "train_treebank", "eval_treebank", "lang", "kind", "probe", "task",
    "init", "pooling", "layer", "L", "tau", "acc", "n_train", "n_eval", "hparams", "seed",
)


def _read_results(source: HfRangeSource, paths: list[str]) -> list[tuple[str, bytes]]:
    """Download and validate the result files before changing local parquets.

    Parameters
    ----------
    source : HfRangeSource
        Authenticated transport pinned to the commit used to list the files.
    paths : list[str]
        Repository paths returned by the HF listing at that commit.

    Returns
    -------
    list[tuple[str, bytes]]
        Validated filenames and original parquet bytes, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If there are no probing parquet files or a selected file is missing.
    ValueError
        If a parquet is incomplete, unreadable, empty, lacks result columns,
        contains multiple jobs, or has a filename inconsistent with its metadata.
    """
    # Match the upload script's flat probing folder, excluding all other outputs.
    selected = sorted(
        path for path in paths
        if PurePosixPath(path).parent == PurePosixPath(REMOTE_PROBING_DIR)
        and PurePosixPath(path).suffix == ".parquet"
    )
    if not selected:
        raise FileNotFoundError(f"No probing parquet files in {HF_REPO_ID}/{REMOTE_PROBING_DIR}.")

    results = []
    for path in selected:
        size = source.size(path)
        content = source.read(path, 0, size)
        if len(content) != size:
            raise ValueError(f"Incomplete download of probing result {path}.")
        try:
            frame = pd.read_parquet(io.BytesIO(content))
        except (ValueError, OSError) as error:
            raise ValueError(f"Cannot read probing result {path}.") from error

        # Use the contracted filename helper to keep remote and local layouts aligned.
        missing = sorted(set(RESULT_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Missing probing result columns in {path}: {missing}.")
        if frame.empty:
            raise ValueError(f"Empty probing result: {path}.")
        jobs = frame[list(JOB_COLUMNS)].drop_duplicates()
        if len(jobs) != 1 or jobs.isna().any().any():
            raise ValueError(f"Expected one job with non-null metadata in {path}.")
        name = PurePosixPath(path).name
        expected_name = probing_result_path(*jobs.iloc[0].tolist()).name
        if name != expected_name:
            raise ValueError(f"Unexpected result filename {name}; expected {expected_name}.")
        results.append((name, content))

    return results


def main(argv: list[str] | None = None) -> None:
    """Download a pinned snapshot and replace local result files after validation.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments; None reads them from the current process.

    Returns
    -------
    None
        Prints the downloaded paths and the source commit reference.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results",
        help="Results root containing probing/ (default: the repository's results/).",
    )
    args = parser.parse_args(argv)
    results_dir = args.results_dir.expanduser()
    if results_dir.exists() and not results_dir.is_dir():
        raise ValueError(f"Results path is not a directory: {results_dir}.")
    if HfApi is None:
        raise ImportError("huggingface_hub is required. Install it with: pip install huggingface_hub")

    # Reuse the existing transport for login, revision resolution and remote reads.
    api = HfApi()
    source = HfRangeSource(HF_REPO_ID, "main", api=api)
    paths = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", revision=source.revision)
    print(f"Reading probing results from {HF_REPO_ID}@{source.revision}...", flush=True)
    results = _read_results(source, paths)

    # Stage every validated file on the destination filesystem before replacing any.
    # Each rename is atomic, so readers never see a partially written parquet.
    output_dir = results_dir / "probing"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, _ in results:
        if (output_dir / name).is_dir():
            raise ValueError(f"Result destination is a directory: {output_dir / name}.")
    with TemporaryDirectory(prefix=".pull-probing-", dir=results_dir) as temporary:
        staging = Path(temporary)
        for name, content in results:
            (staging / name).write_bytes(content)
        for name, _ in results:
            destination = output_dir / name
            (staging / name).replace(destination)
            print(f"Downloaded: {destination}")

    print(f"Commit: {source.revision}")
    print(f"Source: https://huggingface.co/datasets/{HF_REPO_ID}/commit/{source.revision}")


if __name__ == "__main__":
    main()
