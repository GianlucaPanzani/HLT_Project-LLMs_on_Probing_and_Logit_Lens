#!/usr/bin/env python3
"""Generate overlay lens and probing curves for GPT-2 and GePpeTto (Issue L8).

Usage:
    PYTHONPATH=src python scripts/34_plot_lens.py
    PYTHONPATH=src python scripts/34_plot_lens.py --models gpt2,geppetto --output-dir results/figures/ws4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lib.paths import ROOT
from viz.lens_plots import build_commitment_table, plot_crossling_comparison, plot_model_overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate overlay figures of lens curves and probing selectivity (Issue L8)."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="gpt2,geppetto",
        help="Comma-separated model names (default: gpt2,geppetto).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "results" / "figures" / "ws4"),
        help="Directory where figures will be saved.",
    )
    args = parser.parse_args()

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating overlay figures for models: {models} -> {out_dir}")
    for model in models:
        save_path = out_dir / f"{model}_overlay.png"
        plot_model_overlay(model, save_path=save_path)
        print(f"  Saved figure: {save_path.relative_to(ROOT)}")

    if "gpt2" in models and "geppetto" in models:
        cross_path = out_dir / "crossling_gpt2_vs_geppetto.png"
        plot_crossling_comparison(save_path=cross_path)
        print(f"  Saved figure: {cross_path.relative_to(ROOT)}")

    print("\n" + "=" * 75)
    print("Commitment Layers Table (first layer reaching >= 90% of maximum):")
    print("=" * 75)
    df_commit = build_commitment_table(models=models)
    print(df_commit.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
