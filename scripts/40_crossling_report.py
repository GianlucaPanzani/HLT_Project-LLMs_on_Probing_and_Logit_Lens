"""Cross-linguistic Delta CoG report, figures and summary table.

Example:
    PYTHONPATH=src python scripts/40_crossling_report.py
"""

from __future__ import annotations

import argparse
import shutil

import pandas as pd

from lib.paths import ROOT
from viz.crossling_plots import plot_delta_cog, plot_selectivity_pairs
from ws3_crossling.delta_cog import PAIRS, delta_cog, selectivity_curves


def _markdown_table(delta: pd.DataFrame) -> str:
    cols = [
        "pair",
        "kind",
        "eval_set",
        "cog_norm_en",
        "cog_norm_it",
        "delta_cog",
    ]
    view = delta[cols].copy()
    for col in ("cog_norm_en", "cog_norm_it", "delta_cog"):
        view[col] = view[col].map(lambda x: f"{x:.4f}")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(view.iloc[i][c]) for c in cols) + " |" for i in range(len(view))]
    return "\n".join([header, sep, *rows])


def _extra_mbert_mlp(summary: pd.DataFrame) -> pd.DataFrame | None:
    try:
        return delta_cog(summary, pairs=PAIRS[:1], probe="mlp")
    except ValueError:
        return None


def main(argv: list[str] | None = None) -> None:
    """Run Delta CoG computation, write outputs and print paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", default="linear", help="Probe type filter (default: linear).")
    args = parser.parse_args(argv)
    probe = args.probe

    summary_path = ROOT / "results" / "summary" / "cog.parquet"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing {summary_path}; run scripts/11_summarize_probing.py first.")

    probing_dir = ROOT / "results" / "probing"
    paths = sorted(probing_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No probing parquet files in {probing_dir}.")

    summary = pd.read_parquet(summary_path)
    results = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    results = results.loc[results["probe"] == probe]

    delta = delta_cog(summary, probe=probe)
    curves = selectivity_curves(results)

    summary_out = ROOT / "results" / "summary" / "delta_cog.parquet"
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    delta.to_parquet(summary_out, index=False)

    fig_root = ROOT / "results" / "figures" / "ws3"
    doc_fig_root = ROOT / "docs" / "results" / "figures" / "ws3"
    sel_path = fig_root / "selectivity_pairs.png"
    bar_path = fig_root / "delta_cog.png"
    plot_selectivity_pairs(curves, delta, sel_path, probe=probe)
    plot_delta_cog(delta, bar_path)

    doc_fig_root.mkdir(parents=True, exist_ok=True)
    for src in (sel_path, bar_path):
        shutil.copy2(src, doc_fig_root / src.name)

    print(_markdown_table(delta))
    print()
    mlp = _extra_mbert_mlp(summary)
    if mlp is not None:
        print("mbert MLP (robustness):")
        print(_markdown_table(mlp))
        print()

    print(f"Wrote {summary_out}")
    print(f"Wrote {sel_path}")
    print(f"Wrote {bar_path}")
    print(f"Copied figures to {doc_fig_root}")


if __name__ == "__main__":
    main()