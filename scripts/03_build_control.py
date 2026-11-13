"""Build the control task mappings (CONTRACTS.md section 5) from the train word tables.

Usage:
    PYTHONPATH=src python scripts/03_build_control.py                 # ewt and isdt
    PYTHONPATH=src python scripts/03_build_control.py --treebank ewt
"""

import argparse
import json

from lib.control_tasks import build_control_mapping
from lib.paths import control_path
from lib.word_table import load_word_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--treebank", nargs="+", default=["ewt", "isdt"], help="Train treebanks (default: ewt isdt).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for treebank in args.treebank:
        table = load_word_table(treebank, "train")
        control = build_control_mapping(table, seed=args.seed)
        control["treebank"] = treebank
        path = control_path(treebank)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(control, file, ensure_ascii=False)
        print(f"{treebank}: {len(control['mapping'])} word types, {len(table)} tokens -> {path}")


if __name__ == "__main__":
    main()
