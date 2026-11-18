#!/usr/bin/env python3
"""Train Tuned Lens translators on ISDT train for GePpeTto (Issue L4).

This script trains layer-wise affine translators h -> h + A_l h + b_l for
intermediate layers of causal decoder models (GePpeTto) by minimizing the
Kullback-Leibler divergence to the final layer logits on the training treebank.
The resulting weights and metadata are saved to `checkpoints/tuned_lens/`.

Usage
-----
    PYTHONPATH=src python scripts/31_train_tuned_lens.py
    PYTHONPATH=src python scripts/31_train_tuned_lens.py --config configs/ws4.yaml
    PYTHONPATH=src python scripts/31_train_tuned_lens.py --max-steps 50  # fast dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from lib.data import load_model
from lib.registry import get_model_spec
from lib.word_table import load_word_table, table_to_sentences
from ws4_lens.tuned_lens import (
    evaluate_dev_kl,
    save_tuned_lens_checkpoint,
    train_tuned_lens,
)


def load_config(config_path: Path | str) -> dict[str, Any]:
    """Load configuration from YAML file or return sensible defaults."""
    p = Path(config_path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Tuned Lens affine translators for decoder models (Issue L4)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ws4.yaml",
        help="Path to YAML configuration file (default: configs/ws4.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key from registry (default: geppetto).",
    )
    parser.add_argument(
        "--treebank",
        type=str,
        default=None,
        help="Treebank key from registry (default: isdt).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default from config: 8).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default from config: 3).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for AdamW (default: 1e-3).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay for AdamW (default: 1e-3).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional maximum number of training steps (e.g. for profiling/dry-run).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target device override (default: auto-detected MPS/CUDA/CPU).",
    )
    args = parser.parse_args()

    # 1. Load config and merge CLI overrides
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})

    model_key = args.model or cfg.get("model", "geppetto")
    treebank_key = args.treebank or cfg.get("treebank", "isdt")
    train_split = cfg.get("train_split", "train")
    dev_split = cfg.get("dev_split", "dev")

    batch_size = args.batch_size or train_cfg.get("batch_size", 8)
    max_length = train_cfg.get("max_length", 128)
    lr = args.lr or train_cfg.get("lr", 1e-3)
    weight_decay = args.weight_decay or train_cfg.get("weight_decay", 1e-3)
    epochs = args.epochs or train_cfg.get("epochs", 3)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    log_interval = train_cfg.get("log_interval", 50)
    empty_cache_interval = train_cfg.get("empty_cache_interval", 50)
    out_dir = cfg.get("output", {}).get("checkpoint_dir", "checkpoints/tuned_lens")

    print("=== Workstream 4: Tuned Lens Training (Issue L4) ===")
    print(f"Model:       {model_key}")
    print(f"Treebank:    {treebank_key} (train: {train_split}, dev: {dev_split})")
    print(f"Batch Size:  {batch_size}")
    print(f"Epochs:      {epochs}")
    print(f"LR:          {lr}")
    print(f"Weight Decay:{weight_decay}")
    print(f"Seed:        {seed}")
    if args.max_steps is not None:
        print(f"Max Steps:   {args.max_steps} (dry-run mode)")

    # 2. Load model and tokenizer
    spec = get_model_spec(model_key)
    print(f"\nLoading model '{spec['hf_id']}'...")
    model, tokenizer, metadata = load_model(
        checkpoint=spec["hf_id"],
        kind=spec["kind"],
        revision=spec.get("revision", "main"),
        dtype=spec.get("dtype"),
        device=args.device,
        tokenizer_kwargs={"add_prefix_space": True},
    )
    device = metadata["device"]
    print(f"Loaded on device: {device}")

    # 3. Load word tables
    print(f"\nLoading word tables for '{treebank_key}'...")
    train_table = load_word_table(treebank_key, train_split)
    dev_table = load_word_table(treebank_key, dev_split)

    train_sentences = table_to_sentences(train_table)
    dev_sentences = table_to_sentences(dev_table)

    print(f"Loaded {len(train_sentences):,} sentences from {train_split} split.")
    print(f"Loaded {len(dev_sentences):,} sentences from {dev_split} split.")

    # 4. Train Tuned Lens Translators
    num_layers = spec["num_layers"]
    hidden_size = spec["hidden_size"]

    translators, history = train_tuned_lens(
        model=model,
        train_sentences=train_sentences,
        dev_sentences=dev_sentences,
        tokenizer=tokenizer,
        num_layers=num_layers,
        hidden_size=hidden_size,
        batch_size=batch_size,
        max_length=max_length,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        max_steps=args.max_steps,
        log_interval=log_interval,
        empty_cache_interval=empty_cache_interval,
        seed=seed,
    )

    # 5. Save Checkpoint and CONTRACTS.md Section 9.5 metadata
    print("\nSaving checkpoint and metadata...")
    pt_path, json_path = save_tuned_lens_checkpoint(
        translators=translators,
        model_name=model_key,
        output_dir=out_dir,
        train_data=f"{treebank_key}_{train_split}",
        epochs=epochs,
        seed=seed,
    )
    print(f"Written checkpoint: {pt_path}")
    print(f"Written metadata:   {json_path}")

    # 6. Evaluation Table for Acceptance Criteria
    print("\nEvaluating final Tuned Lens on validation set...")
    final_dev_kl = evaluate_dev_kl(
        model=model,
        sentences=dev_sentences,
        tokenizer=tokenizer,
        translators=translators,
        batch_size=16,
    )
    baseline_dev_kl = history["baseline_dev_kl"]

    print("\n" + "=" * 80)
    print(f"### {treebank_key.upper()} Dev Per-Layer KL Divergence Table (Issue [L4] Acceptance Criteria)")
    print("=" * 80)
    print(f"| {'Layer':<5} | {'Logit Lens KL':<13} | {'Tuned Lens KL':<13} | {'Reduction (%)':<13} | {'Status':<10} |")
    print(f"|{'-'*7}|{'-'*15}|{'-'*15}|{'-'*15}|{'-'*12}|")

    all_lower = True
    for layer_idx in range(num_layers):
        kl_logit = baseline_dev_kl[layer_idx]
        kl_tuned = final_dev_kl[layer_idx]
        reduction = ((kl_logit - kl_tuned) / kl_logit) * 100
        status = "✅ Lower" if kl_tuned < kl_logit else "❌ Higher"
        if kl_tuned >= kl_logit:
            all_lower = False
        print(f"| {layer_idx:<5} | {kl_logit:<13.4f} | {kl_tuned:<13.4f} | {reduction:>11.2f}% | {status:<10} |")

    print("-" * 80)
    print(f"Acceptance Criteria Passed (All intermediate layers show lower KL): {all_lower}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
