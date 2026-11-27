#!/usr/bin/env python3
"""Generate the compression workstream report (C4).

Loads probing results for BERT and DistilBERT and CKA results, builds
the layer mapping table, generates comparison plots, and writes the
markdown report to ``docs/results/ws2_compression.md``.

Usage:
    PYTHONPATH=src python scripts/21_compression_report.py

Requires P4 probing results and C3 CKA results to exist on disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from lib.paths import cka_result_path, probing_result_path
from ws2_compression.mapping import build_layer_mapping_table, compute_cog_comparison

ROOT = Path(__file__).resolve().parents[1]


def _load_probing(model: str, treebank: str = "ewt", probe: str = "linear",
                  init: str = "pretrained", pooling: str = "first") -> pd.DataFrame:
    """Load probing results parquet for a model."""
    path = probing_result_path(model, treebank, probe, init, pooling)
    if not path.is_file():
        raise FileNotFoundError(
            f"Probing results not found: {path}\n"
            f"Run P4 (probe runner) for {model} first."
        )
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate compression workstream report.")
    parser.add_argument("--treebank", default="ewt", help="Treebank (default: ewt)")
    parser.add_argument("--probe", default="linear", help="Probe type (default: linear)")
    parser.add_argument("--init", default="pretrained", help="Init (default: pretrained)")
    parser.add_argument("--pooling", default="first", help="Pooling (default: first)")
    args = parser.parse_args(argv)

    # Load data
    print("Loading probing results...")
    probing_bert = _load_probing("bert", args.treebank, args.probe, args.init, args.pooling)
    probing_distilbert = _load_probing("distilbert", args.treebank, args.probe, args.init, args.pooling)

    cka_path = cka_result_path("bert", "distilbert", args.treebank)
    if not cka_path.is_file():
        raise FileNotFoundError(f"CKA results not found: {cka_path}\nRun scripts/20_cka.py first.")
    cka_df = pd.read_parquet(cka_path)

    # Build mapping table
    print("Building layer mapping table...")
    mapping = build_layer_mapping_table(
        probing_bert, probing_distilbert, cka_df,
        treebank=args.treebank, probe=args.probe, init=args.init, pooling=args.pooling,
    )
    print(mapping.to_string(index=False))

    # Compute CoG comparison
    print("\nComputing CoG comparison...")
    cog = compute_cog_comparison(
        probing_bert, probing_distilbert,
        treebank=args.treebank, probe=args.probe, init=args.init, pooling=args.pooling,
    )
    print(f"  BERT CoG_norm:       {cog['bert_cog_norm']:.4f}")
    print(f"  DistilBERT CoG_norm: {cog['distilbert_cog_norm']:.4f}")

    # Generate plots
    fig_dir = ROOT / "results" / "figures" / "ws2"
    try:
        from viz.cka_plots import plot_selectivity_comparison

        sel_path = fig_dir / "bert_vs_distilbert_selectivity.png"
        plot_selectivity_comparison(
            cog["bert_selectivity"], cog["bert_tau"],
            cog["distilbert_selectivity"], cog["distilbert_tau"],
            bert_cog_norm=cog["bert_cog_norm"],
            distil_cog_norm=cog["distilbert_cog_norm"],
            save_path=sel_path,
        )
        print(f"\nWrote {sel_path}")
    except ImportError as e:
        print(f"\nSkipping plots (missing dependency): {e}", file=sys.stderr)

    # Write markdown report
    report_path = ROOT / "docs" / "results" / "ws2_compression.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    md = _build_report(mapping, cog, args)
    report_path.write_text(md, encoding="utf-8")
    print(f"Wrote {report_path}")

    # Cleanup
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except ImportError:
        pass


def _build_report(mapping: pd.DataFrame, cog: dict, args) -> str:
    """Build the markdown report content."""
    delta_cog = cog["distilbert_cog_norm"] - cog["bert_cog_norm"]
    lines = [
        "# Workstream 2: Model Compression & Linguistic Hierarchy",
        "",
        "Comparing `bert-base-cased` ($L=12$, teacher) and `distilbert-base-cased` ($L=6$, student) "
        f"on `{args.treebank}` (`test` split, `{args.pooling}`-subword pooling, `{args.probe}` probe, "
        f"`{args.init}` weights, seed 42, frozen backbones).",
        "",
        "C5 audit (27 November 2026): "
        "[baseline inventory and provenance](../../src/ws2_compression/BASELINES.md); "
        "[literature evidence and interpretation checks](../../src/ws2_compression/EVIDENCE.md).",
        "",
        "## BERT (L=12) vs DistilBERT (L=6)",
        "",
        "### CKA Heatmap",
        "",
        "![CKA Heatmap](../../results/figures/ws2/bert__vs__distilbert_cka_heatmap.png)",
        "",
        "### 1:2 Layer Mapping Table",
        "",
        "For each DistilBERT layer *l*, the rigid mapping assumes BERT layer *2l*.",
        "The CKA-best column shows the BERT layer with highest geometric similarity.",
        "",
    ]

    # Mapping table (explicitly format integer layer columns vs float metric columns)
    cols = [
        ("DistilBERT", "distilbert_layer", "int"),
        ("Distil Acc", "distilbert_acc", "float"),
        ("Distil Sel", "distilbert_sel", "float"),
        ("Rigid BERT", "rigid_bert_layer", "int"),
        ("Rigid Acc", "rigid_bert_acc", "float"),
        ("Rigid Sel", "rigid_bert_sel", "float"),
        ("CKA-Best BERT", "cka_best_bert_layer", "int"),
        ("CKA-Best Acc", "cka_best_bert_acc", "float"),
        ("CKA-Best Sel", "cka_best_bert_sel", "float"),
        ("CKA", "cka_best_cka", "float"),
    ]
    header = "| " + " | ".join(c[0] for c in cols) + " |"
    sep = "| " + " | ".join("---:" for _ in cols) + " |"
    lines.extend([header, sep])

    for _, row in mapping.iterrows():
        vals = []
        for _, key, kind in cols:
            v = row[key]
            if kind == "int":
                vals.append(str(int(v)))
            else:
                vals.append(f"{float(v):.4f}")
        lines.append("| " + " | ".join(vals) + " |")

    lines.extend([
        "",
        "### Selectivity Comparison",
        "",
        "![Selectivity](../../results/figures/ws2/bert_vs_distilbert_selectivity.png)",
        "",
        f"- **BERT CoG_norm**: {cog['bert_cog_norm']:.4f}",
        f"- **DistilBERT CoG_norm**: {cog['distilbert_cog_norm']:.4f}",
        f"- **Delta CoG_norm (DistilBERT - BERT)**: {delta_cog:+.4f}",
        "",
        "### Interpretation",
        "",
        "1. **Geometric correspondence (CKA)**: DistilBERT's distillation objective includes "
        "a hidden-state cosine embedding loss $\\mathcal{L}_{\\text{cos}}$ and initializes student "
        "layer *l* from teacher layer *2l* (Sanh et al., 2019, §§3.2–3.3), so near-diagonal "
        "alignment is expected rather than purely emergent. Empirically, CKA-best matches the "
        "1:2 diagonal at layers 0, 2, 4, and 6 (CKA 0.9470, 0.8294, 0.7731, 0.6208) and shifts "
        "by one layer at layers 1 (BERT 1, 0.8774), 3 (BERT 5, 0.8068), and 5 (BERT 11, 0.7285). "
        "High CKA reflects geometric similarity under this sample and pooling, not causal or "
        "functional equivalence (Kornblith et al., 2019, §6.1).",
        "",
        "2. **Functional correspondence (Probing)**: DistilBERT closely tracks BERT's UPOS "
        "accuracy (peak 0.9477 at layer 3 vs BERT 0.9519 at layer 6; rigid 1:2 differences within "
        "0.0049), while its selectivity is consistently lower across contextual layers "
        "(peak 0.3371 at layer 5 vs BERT 0.3818 at layer 11; rigid 1:2 gaps of -0.0210 to -0.0400).",
        "",
        "3. **Is the hierarchy compressed or shifted?** DistilBERT's $\\text{CoG}_{\\text{norm}}$ "
        f"({cog['distilbert_cog_norm']:.4f}) is slightly higher than BERT's "
        f"({cog['bert_cog_norm']:.4f}, $\\Delta = {delta_cog:+.4f}$), indicating that "
        "morphosyntactic selectivity concentrates at a similar—slightly deeper—relative depth "
        "within the 6-layer student.",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    main()
