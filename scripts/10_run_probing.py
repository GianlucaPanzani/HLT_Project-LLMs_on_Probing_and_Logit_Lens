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


def _load_jobs_from_yaml(path: Path) -> list[dict]:
    """Load YAML file with defaults and jobs defined inside, rejecting unknown or incomplete settings.

    Parameters
    ----------
    path : Path
        YAML file with a jobs list and optional defaults mapping.

    Returns
    -------
    list[dict]
        Job arguments, with each job taking precedence over defaults.
    """
    # Save the content of the YAML file in config
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Case of bad format of YAML file
    if not isinstance(config, dict) or set(config) - {"defaults", "jobs"}:
        raise ValueError("Config must be a mapping with only 'defaults' and 'jobs'.")
    
    # Initializations
    jobs_params = config.get("jobs")
    default_params = config.get("defaults", {})
    allowed_params = set(REQUIRED_KEYS + OPTIONAL_KEYS)

    # Validation of the data
    if not isinstance(default_params, dict):
        raise ValueError("Config defaults must be a mapping.")
    if not isinstance(jobs_params, list) or not jobs_params:
        raise ValueError("Config jobs must be a nonempty list.")
    if set(default_params) - allowed_params:
        raise ValueError(f"Unknown default keys: {set(default_params) - allowed_params}.")
    
    # Each iteration creates the set of parameters of each job (= set of model's parameters)
    merged_jobs = []
    outputs = set()
    for index, job_params in enumerate(jobs_params, start=1):
        if not isinstance(job_params, dict) or set(job_params) - allowed_params:
            raise ValueError(f"Job {index} must be a mapping using supported runner arguments.")
        
        merged_params = {**default_params, **job_params}

        # Check of missing parameters
        missing = [key for key in REQUIRED_KEYS if not isinstance(merged_params.get(key), str) or not merged_params[key]]
        if missing:
            raise ValueError(f"Job {index} requires string values for: {', '.join(missing)}.")
        
        # Checks of params that should have a list as value
        for param in ("tasks", "eval_treebanks", "layers"):
            if (
                param in merged_params 
                and not (param != "tasks" and merged_params[param] is None) 
                and not isinstance(merged_params[param], list)
            ):
                raise ValueError(f"Job {index}: {param} must be a list.")
        
        # Check the overlapping of filepaths (based on the values associated with the required parameters)
        output = tuple(merged_params[key] for key in REQUIRED_KEYS)
        if output in outputs:
            raise ValueError(f"Job {index} would overwrite another job's result file.")
        outputs.add(output)

        merged_jobs.append(merged_params)

    return merged_jobs


############################################################
########################## MAIN ############################
############################################################

def main(argv: list[str] | None = None) -> None:

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
    parser.add_argument("--max-train-rows", type=_row_limit, default=argparse.SUPPRESS, help="Positive integer (default: 50000), or 'none' for all rows.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--quiet", action="store_true", help="Only print job progress and output paths.")
    args = parser.parse_args(argv)

    options = {
        arg: value for arg, value in vars(args).items()
        if arg in REQUIRED_KEYS + OPTIONAL_KEYS and (value is not None or arg == "max_train_rows")
    }

    if args.config:
        if options:
            parser.error("--config cannot be combined with individual job flags.")
        try:
            jobs = _load_jobs_from_yaml(args.config)
        except (OSError, ValueError, yaml.YAMLError) as error:
            parser.error(str(error))
    else:
        missing = [f"--{key.replace('_', '-')}" for key in REQUIRED_KEYS if key not in options]
        if missing:
            parser.error(f"Required for a single job: {', '.join(missing)}.")
        jobs = [options]

    n_jobs = len(jobs)
    for i, job in enumerate(jobs, start=1):
        print(f"Job {i}/{n_jobs}: {job}", end=" -> " if args.quiet else "\n", flush=True)
        run_probing(**job, verbose=0 if args.quiet else 1)
        path = probing_result_path(*(job[key] for key in REQUIRED_KEYS))
        print(f"Wrote to {path}", flush=True)
    
    return


if __name__ == "__main__":
    main()
