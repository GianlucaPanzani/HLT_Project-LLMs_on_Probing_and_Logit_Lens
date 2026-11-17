"""Report which Italian/multilingual feature folders (issue E6) exist, with size, time and NaN rows.

Reads only meta.json and file sizes; it never loads a model, so it is safe on any machine.
Paste the printed table in the issue comment of the run.

Examples:
    PYTHONPATH=src python scripts/04_report_features.py
    PYTHONPATH=src python scripts/04_report_features.py --models bert_it geppetto

Set HLT_FEATURES_DIR if the features are on another disk.
"""

import argparse

from lib.feature_report import format_markdown, report_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None, help="Only these registry keys (default: all E6 models).")
    args = parser.parse_args()
    print(format_markdown(report_runs(tuple(args.models) if args.models else None)))


if __name__ == "__main__":
    main()
