#!/usr/bin/env python3
"""Run all lenses on evaluation data and write aggregate and per-token results (Issue L7).

Evaluates:
- Standard Logit Lens (nostalgebraist, 2020)
- Tuned Lens (Belrose et al., 2023)
- Jacobian Lens (Gurnee et al., 2026)

On the required evaluation runs:
    (gpt2, pud_en)
    (gpt2, wikitext)
    (geppetto, pud_it)

Writes:
    results/lens/{model}__{dataset}.parquet (aggregate metrics per (lens, layer))
    results/lens/tokens/{model}__{dataset}.parquet (per-token records for first 100 sequences)

Per CONTRACTS.md Sections 9.1, 9.3, and 9.4.

Usage
-----
    PYTHONPATH=src python scripts/32_run_lenses.py --model gpt2 --dataset pud_en
    PYTHONPATH=src python scripts/32_run_lenses.py --model gpt2 --dataset wikitext
    PYTHONPATH=src python scripts/32_run_lenses.py --model geppetto --dataset pud_it
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lib.data import load_model
from lib.paths import lens_result_path
from lib.registry import get_model_spec
from lib.word_table import load_word_table
from ws4_lens.evaluate import (
    evaluate_lenses,
    prepare_pud_batches,
    prepare_wikitext_batches,
)
from ws4_lens.jacobian_lens import load_jacobian_checkpoint
from ws4_lens.tuned_lens import (
    load_official_tuned_lens,
    load_tuned_lens_checkpoint,
)
from ws4_lens.upos_bridge import build_upos_lexicon


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all lenses and write aggregate and per-token results (Issue L7)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt2", "geppetto"],
        help="Model registry key (gpt2 or geppetto).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["pud_en", "pud_it", "wikitext"],
        help="Dataset key (pud_en, pud_it, or wikitext).",
    )
    parser.add_argument(
        "--lenses",
        type=str,
        default="logit,tuned,jacobian",
        help="Comma-separated lenses to evaluate (default: logit,tuned,jacobian).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size (default: 16).",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=200,
        help="Maximum windows for WikiText-103 (default: 200).",
    )
    parser.add_argument(
        "--max-token-sequences",
        type=int,
        default=100,
        help="Maximum sequences to record in per-token results (default: 100).",
    )

    args = parser.parse_args()
    lens_list = tuple(lens_name.strip() for lens_name in args.lenses.split(",") if lens_name.strip())

    spec = get_model_spec(args.model)
    print(f"Loading model '{args.model}' ({spec['hf_id']})...")
    model, tokenizer, meta = load_model(
        checkpoint=spec["hf_id"],
        kind=spec["kind"],
        tokenizer_kwargs={"add_prefix_space": True},
    )
    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    print(f"Model loaded on device: {device}")

    # Load lens components
    tuned_translators = None
    if "tuned" in lens_list:
        print(f"Loading Tuned Lens translators for '{args.model}'...")
        if args.model == "gpt2":
            tuned_translators = load_official_tuned_lens(model)
        else:
            tuned_translators = load_tuned_lens_checkpoint("geppetto").to(device)

    jacobian_matrices = None
    if "jacobian" in lens_list:
        print(f"Loading Jacobian matrices for '{args.model}'...")
        jacobian_matrices = load_jacobian_checkpoint(args.model)

    upos_lexicon = None
    if args.dataset in ("pud_en", "pud_it"):
        train_treebank = "ewt" if args.dataset == "pud_en" else "isdt"
        print(f"Building UPOS bridge lexicon from '{train_treebank}' train word table...")
        train_table = load_word_table(train_treebank, "train")
        upos_lexicon = build_upos_lexicon(train_table)
        print(f"UPOS lexicon built with {len(upos_lexicon):,} entries.")

    # Prepare data batches
    print(f"Preparing batches for '{args.dataset}'...")
    if args.dataset in ("pud_en", "pud_it"):
        batches = prepare_pud_batches(args.dataset, tokenizer, batch_size=args.batch_size)
    else:
        batches = prepare_wikitext_batches(
            tokenizer,
            window_size=256,
            max_windows=args.max_windows,
            batch_size=args.batch_size,
        )
    print(f"Prepared {len(batches)} batches.")

    print(f"Evaluating lenses {lens_list} on ({args.model}, {args.dataset})...")
    df_aggregate, df_tokens = evaluate_lenses(
        model,
        tokenizer,
        model_name=args.model,
        dataset_name=args.dataset,
        data_batches=batches,
        tuned_translators=tuned_translators,
        jacobian_matrices=jacobian_matrices,
        upos_lexicon=upos_lexicon,
        lenses=lens_list,
        max_token_sequences=args.max_token_sequences,
        device=device,
        verbose=True,
    )

    # Save outputs
    agg_path = lens_result_path(args.model, args.dataset)
    tok_path = lens_result_path(args.model, args.dataset, per_token=True)
    Path(agg_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tok_path).parent.mkdir(parents=True, exist_ok=True)

    df_aggregate.to_parquet(agg_path, index=False)
    df_tokens.to_parquet(tok_path, index=False)

    print(f"✅ Aggregate results written to: {agg_path} ({len(df_aggregate)} rows)")
    print(f"✅ Per-token results written to: {tok_path} ({len(df_tokens)} rows)")


if __name__ == "__main__":
    main()
