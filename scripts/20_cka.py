#!/usr/bin/env python3
"""Compute the CKA matrix between two models and generate the heatmap.

Usage:
    PYTHONPATH=src python scripts/20_cka.py --model-a bert --model-b distilbert

Output:
    results/cka/{model_a}__vs__{model_b}__{treebank}.parquet
    results/figures/ws2/{model_a}__vs__{model_b}_cka_heatmap.png
    results/figures/ws2/{model_a}__vs__{model_b}_cka_heatmap.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lib.paths import cka_result_path
from ws2_compression.cka import cka_matrix


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute CKA matrix between two models.")
    parser.add_argument("--model-a", required=True, help="First model registry key (e.g. bert)")
    parser.add_argument("--model-b", required=True, help="Second model registry key (e.g. distilbert)")
    parser.add_argument("--treebank", default="ewt", help="Treebank key (default: ewt)")
    parser.add_argument("--split", default="test", help="Split (default: test)")
    parser.add_argument("--pooling", default="first", help="Pooling strategy (default: first)")
    parser.add_argument("--n-tokens", type=int, default=20000, help="Number of tokens to sample (default: 20000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args(argv)

    # Compute CKA matrix
    print(f"Computing CKA matrix: {args.model_a} vs {args.model_b} on {args.treebank} ({args.split})...")
    df = cka_matrix(
        args.model_a, args.model_b, args.treebank, args.split,
        pooling=args.pooling, n_tokens=args.n_tokens, seed=args.seed,
    )

    # Save parquet
    out_path = cka_result_path(args.model_a, args.model_b, args.treebank)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")

    # Print summary: best BERT layer for each DistilBERT layer
    layers_b = sorted(df["layer_b"].unique())
    print(f"\nBest {args.model_a} layer for each {args.model_b} layer:")
    print(f"{'Model B layer':>14} -> {'Best Model A layer':>18}  (CKA)")
    print("-" * 50)
    for lb in layers_b:
        col = df.loc[df["layer_b"] == lb]
        best = col.loc[col["cka"].idxmax()]
        print(f"{lb:>14} -> {int(best['layer_a']):>18}  ({best['cka']:.4f})")

    # Generate heatmap
    try:
        from viz.cka_plots import plot_cka_heatmap

        fig_dir = Path(__file__).resolve().parents[1] / "results" / "figures" / "ws2"
        prefix = fig_dir / f"{args.model_a}__vs__{args.model_b}"
        plot_cka_heatmap(df, save_prefix=prefix)
        print(f"\nWrote {prefix}_cka_heatmap.png")
        print(f"Wrote {prefix}_cka_heatmap.html")
    except ImportError as e:
        print(f"\nSkipping heatmap (missing dependency): {e}", file=sys.stderr)

    plt_cleanup()


def plt_cleanup() -> None:
    """Close all matplotlib figures to free memory."""
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
