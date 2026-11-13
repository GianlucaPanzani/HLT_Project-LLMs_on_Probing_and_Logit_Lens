#!/usr/bin/env python3
"""Build word-table parquet files for every treebank split.

Usage
-----
    PYTHONPATH=src python scripts/02_build_word_tables.py
    PYTHONPATH=src python scripts/02_build_word_tables.py --treebank ewt isdt
"""

from __future__ import annotations

import argparse

from lib.data import load_ud
from lib.paths import raw_dir, word_table_path
from lib.registry import get_treebank_spec
from lib.word_table import build_word_table

DEFAULT_TREEBANKS: tuple[str, ...] = ("ewt", "isdt", "pud_en", "pud_it")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build word-table parquet files from downloaded treebanks."
    )
    parser.add_argument(
        "--treebank",
        nargs="*",
        default=None,
        help="Treebank keys to process (default: all).",
    )
    args = parser.parse_args()

    keys = args.treebank if args.treebank else DEFAULT_TREEBANKS

    for key in keys:
        spec = get_treebank_spec(key)
        data_dir = raw_dir(key)
        splits = tuple(spec["splits"])
        lang = spec["lang"]

        print(f"[{key}]")
        dataset = load_ud(data_dir, splits=splits)

        for split in splits:
            sentences = dataset[split]
            table = build_word_table(sentences, lang)

            out_path = word_table_path(key, split)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(out_path, index=False)

            n_rows = len(table)
            n_sents = table["sent_id"].nunique()
            n_null_next = int(table["next_upos"].isna().sum())
            print(
                f"  {out_path}: {n_rows} rows, {n_sents} sentences, "
                f"{n_null_next} null next_upos"
            )
            # Sanity: null next_upos count must equal sentence count.
            assert n_null_next == n_sents, (
                f"next_upos null count ({n_null_next}) != "
                f"sentence count ({n_sents}) for {key}/{split}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
