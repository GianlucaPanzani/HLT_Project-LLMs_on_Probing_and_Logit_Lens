"""Summarize all results/probing/*.parquet into results/summary/cog.parquet.

Example:
    PYTHONPATH=src python scripts/11_summarize_probing.py
"""

import argparse

import pandas as pd

from lib.metrics import summarize_probing
from lib.paths import ROOT


def main(argv: list[str] | None = None) -> None:
    """Read probing results, compute the contracted summary and print its path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    # Path from which load the probing data
    paths = sorted((ROOT / "results" / "probing").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No probing parquet files in {ROOT / 'results' / 'probing'}.")
    
    # Unique Dataframe with all the results
    results = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)

    # Get summary by the metrics file
    summary = summarize_probing(results)
    if summary.empty:
        raise ValueError("No complete task/control pairs available for the summary.")
    
    # Output directory for the summary
    output = ROOT / "results" / "summary" / "cog.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save data
    summary.to_parquet(output, index=False)
    print(f"Summary results wrote to {output} ({len(summary)} rows from {len(paths)} result files).")
    return


if __name__ == "__main__":
    main()
