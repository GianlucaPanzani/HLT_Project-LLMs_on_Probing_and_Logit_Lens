"""Run one probing experiment or the jobs in configs/ws1.yaml.

Examples:
    PYTHONPATH=src python scripts/10_run_probing.py --model bert --train-treebank ewt --probe linear --init pretrained --pooling first
    PYTHONPATH=src python scripts/10_run_probing.py --config configs/ws1.yaml

Set HLT_HF_REVISION to the feature commit and authenticate with Hugging Face.
HLT_HF_REPO defaults to hlt-group11/features. Results overwrite the job's parquet.
"""

import argparse
from pathlib import Path

import yaml

from lib.paths import probing_result_path
from ws1_probing.runner import run_probing

REQUIRED_KEYS = ("model", "train_treebank", "probe", "init", "pooling")
OPTIONAL_KEYS = ("tasks", "eval_treebanks", "layers", "max_train_rows", "seed")


def _row_limit(value: str) -> int | None:
    """Parse a positive row limit, or 'none' to use every training row."""
    if value.lower() == "none":
        return None
    try:
        limit = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Use a positive integer or 'none'.") from None
    if limit <= 0:
        raise argparse.ArgumentTypeError("Use a positive integer or 'none'.")
    return limit


def _load_jobs(path: Path) -> list[dict]:
    """Load YAML defaults and jobs, rejecting unknown or incomplete settings.

    Parameters
    ----------
    path : Path
        YAML file with a jobs list and optional defaults mapping.

    Returns
    -------
    list[dict]
        Job arguments, with each job taking precedence over defaults.
    """
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or set(config) - {"defaults", "jobs"}:
        raise ValueError("Config must be a mapping with only 'defaults' and 'jobs'.")
    defaults = config.get("defaults", {})
    jobs = config.get("jobs")
    if not isinstance(defaults, dict):
        raise ValueError("Config defaults must be a mapping.")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Config jobs must be a nonempty list.")
    allowed = set(REQUIRED_KEYS + OPTIONAL_KEYS)
    if set(defaults) - allowed:
        raise ValueError(f"Unknown default keys: {set(defaults) - allowed}.")
    merged_jobs = []
    outputs = set()
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict) or set(job) - allowed:
            raise ValueError(f"Job {index} must be a mapping using supported runner arguments.")
        merged = {**defaults, **job}
        missing = [key for key in REQUIRED_KEYS if not isinstance(merged.get(key), str) or not merged[key]]
        if missing:
            raise ValueError(f"Job {index} requires string values for: {', '.join(missing)}.")
        for key in ("tasks", "eval_treebanks", "layers"):
            if key in merged and not (key != "tasks" and merged[key] is None):
                if not isinstance(merged[key], list):
                    raise ValueError(f"Job {index}: {key} must be a list.")
        # The filename does not include tasks, layers or seed: avoid silent data loss.
        output = tuple(merged[key] for key in REQUIRED_KEYS)
        if output in outputs:
            raise ValueError(f"Job {index} would overwrite another job's result file.")
        outputs.add(output)
        merged_jobs.append(merged)
    return merged_jobs


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments, run experiments sequentially and print output paths."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="YAML jobs; cannot be combined with individual job flags.")
    parser.add_argument("--model")
    parser.add_argument("--train-treebank")
    parser.add_argument("--probe", choices=("linear", "mlp"))
    parser.add_argument("--init", choices=("pretrained", "random"))
    parser.add_argument("--pooling", choices=("first", "mean", "last"))
    parser.add_argument("--tasks", nargs="+", choices=("upos", "control"))
    parser.add_argument("--eval-treebanks", nargs="+")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--max-train-rows", type=_row_limit, default=argparse.SUPPRESS,
                        help="Positive integer (default: 50000), or 'none' for all rows.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quiet", action="store_true", help="Only print job progress and output paths.")
    args = parser.parse_args(argv)
    options = {key: value for key, value in vars(args).items()
               if key in REQUIRED_KEYS + OPTIONAL_KEYS and (value is not None or key == "max_train_rows")}
    if args.config:
        if options:
            parser.error("--config cannot be combined with individual job flags.")
        try:
            jobs = _load_jobs(args.config)
        except (OSError, ValueError, yaml.YAMLError) as error:
            parser.error(str(error))
    else:
        missing = [f"--{key.replace('_', '-')}" for key in REQUIRED_KEYS if key not in options]
        if missing:
            parser.error(f"Required for a single job: {', '.join(missing)}.")
        jobs = [options]

    for index, job in enumerate(jobs, start=1):
        print(f"Job {index}/{len(jobs)}: {job}", flush=True)
        run_probing(**job, verbose=0 if args.quiet else 1)
        path = probing_result_path(*(job[key] for key in REQUIRED_KEYS))
        print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
