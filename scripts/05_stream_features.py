"""Stream one layer of extracted features from a private Hugging Face repository (issue DS).

Reads the batches one by one and keeps nothing: it prints rows, shapes and the
bytes transferred. See docs/REMOTE_FEATURES.md.

Example:
    PYTHONPATH=src python scripts/05_stream_features.py --repo-id hlt-group11/features --revision main \
        --model gpt2 --treebank ewt --split test --layer 6 --start-row 0 --stop-row 2000
"""

import argparse
import sys

import numpy as np

from lib.remote_features import HfRangeSource, iter_remote_feature_batches

try:
    import resource  # not available on Windows
except ImportError:
    resource = None


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MB, or None where it cannot be measured."""
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="private Hugging Face repository, for example team/features")
    parser.add_argument("--revision", required=True, help="branch, tag or commit (resolved to a commit hash)")
    parser.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--features-prefix", default="", help="folder of the feature folders in the repository (default: root)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--init", default="pretrained", choices=["pretrained", "random"])
    parser.add_argument("--pooling", default="first", choices=["first", "mean", "last"])
    parser.add_argument("--treebank", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--stop-row", type=int, default=None)
    parser.add_argument("--print-every", type=int, default=1, help="print one line every this many batches")
    args = parser.parse_args()

    source = HfRangeSource(args.repo_id, args.revision, args.repo_type)
    print(f"Repository {args.repo_id} at commit {source.revision}")
    batches = iter_remote_feature_batches(
        args.repo_id, args.model, args.init, args.pooling, args.treebank, args.split, args.layer,
        revision=args.revision, batch_size=args.batch_size, start_row=args.start_row, stop_row=args.stop_row,
        repo_type=args.repo_type, features_prefix=args.features_prefix, source=source,
    )
    n_batches = n_rows = n_nan_rows = 0
    for batch in batches:
        n_batches += 1
        n_rows += len(batch.rows)
        n_nan_rows += int(np.isnan(batch.features).any(axis=1).sum())
        if n_batches % args.print_every == 0:
            print(f"batch {n_batches}: rows {batch.rows[0]}..{batch.rows[-1]}, features {batch.features.shape} "
                  f"{batch.features.dtype}, table {batch.table.shape}")
    print(f"{n_rows} rows in {n_batches} batches, {n_nan_rows} NaN rows")
    print(f"Transferred {source.bytes_transferred / 1e6:.2f} MB (payload of the range reads)")
    peak = peak_memory_mb()
    print(f"Peak memory {peak:.0f} MB" if peak is not None else "Peak memory not available on this platform")


if __name__ == "__main__":
    main()
