"""Extract word features of every layer (CONTRACTS.md section 4).

Examples:
    PYTHONPATH=src python scripts/01_extract_features.py --model bert --treebank ewt
    PYTHONPATH=src python scripts/01_extract_features.py --model gpt2 --treebank ewt --split test --pooling last
    PYTHONPATH=src python scripts/01_extract_features.py --model bert --treebank ewt --init random

Set HLT_FEATURES_DIR to write the features to another disk.
"""

import argparse
import logging
import time

from lib.alignment import POOLINGS
from lib.features import INITS, extract_features
from lib.registry import get_treebank_spec


def _size_gb(folder) -> float:
    return sum(p.stat().st_size for p in folder.iterdir() if p.is_file()) / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Registry key, e.g. bert, distilbert, gpt2.")
    parser.add_argument("--treebank", required=True, help="Treebank key, e.g. ewt, pud_en.")
    parser.add_argument("--split", nargs="+", default=None, help="Splits (default: all splits of the treebank).")
    parser.add_argument("--init", choices=INITS, default="pretrained")
    parser.add_argument("--pooling", choices=POOLINGS, default="first")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None, help="Keep whole sentences up to this many rows.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute folders that are already complete.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    splits = args.split or list(get_treebank_spec(args.treebank)["splits"])

    total = time.perf_counter()
    for split in splits:
        start = time.perf_counter()
        folder = extract_features(
            args.model, args.treebank, split, init=args.init, pooling=args.pooling,
            batch_size=args.batch_size, max_rows=args.max_rows, overwrite=args.overwrite,
        )
        print(f"{split}: {folder} ({_size_gb(folder):.2f} GB, {time.perf_counter() - start:.0f} s)")
    print(f"Total time: {time.perf_counter() - total:.0f} s")


if __name__ == "__main__":
    main()
