#!/usr/bin/env python3
"""Estimate Jacobian Lens matrices for autoregressive decoder models (Issue L5).

This script estimates first-order residual stream sensitivity matrices J_l across
all layers of causal decoder models (GPT-2 and GePpeTto) using autograd on the
calibration train splits (WikiText-103 train for GPT-2, ISDT train for GePpeTto).
The resulting tensor [L + 1, d, d] is saved to `checkpoints/jacobian_lens/{model}.pt`
per CONTRACTS.md Section 9.5.

Usage
-----
    PYTHONPATH=src python scripts/33_estimate_jacobians.py --model gpt2 --n-positions 128
    PYTHONPATH=src python scripts/33_estimate_jacobians.py --model geppetto --n-positions 128 --evaluate-kl
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from lib.alignment import encode_words
from lib.data import load_model
from lib.paths import ROOT
from lib.registry import get_model_spec
from lib.word_table import load_word_table, table_to_sentences
from ws4_lens.jacobian_lens import (
    estimate_jacobians,
    load_jacobian_checkpoint,
    save_jacobian_checkpoint,
)
from ws4_lens.logit_lens import _get_lens_components, run_model
from ws4_lens.metrics import kl_to_final


def build_isdt_batches(
    tokenizer,
    *,
    batch_size: int = 8,
    max_length: int = 128,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load ISDT train word table and encode into (input_ids, attention_mask) batches."""
    train_table = load_word_table("isdt", "train")
    sentences = table_to_sentences(train_table)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(sentences), batch_size):
        batch_words = [words for _, words in sentences[i : i + batch_size]]
        encoded, _ = encode_words(tokenizer, batch_words, add_bos=True, max_length=max_length)
        batches.append((encoded["input_ids"], encoded["attention_mask"]))
    return batches


def build_wikitext_batches(
    tokenizer,
    *,
    window_size: int = 256,
    max_windows: int = 200,
    batch_size: int = 4,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Stream WikiText-103 train into non-overlapping 256-token windows."""
    from datasets import load_dataset

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
    )
    target_tokens = window_size * max_windows
    token_buf: list[int] = []
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_buf.extend(ids)
        if len(token_buf) >= target_tokens:
            break

    n_win = min(max_windows, len(token_buf) // window_size)
    windows = [
        torch.tensor(token_buf[i * window_size : (i + 1) * window_size], dtype=torch.long)
        for i in range(n_win)
    ]

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, len(windows), batch_size):
        chunk = torch.stack(windows[i : i + batch_size], dim=0)
        mask = torch.ones_like(chunk)
        batches.append((chunk, mask))
    return batches


@torch.no_grad()
def evaluate_jacobian_vs_logit_kl(
    model: nn.Module,
    jacobians: torch.Tensor,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    n_sequences: int = 50,
    title: str = "Calibration Split",
) -> tuple[list[float], list[float]]:
    """Compare per-layer mean KL(P_L || P_l) of Logit Lens vs Jacobian Lens on `n_sequences`.

    Evaluates 1 sequence at a time and projects layer-by-layer to keep peak MPS/GPU
    memory tiny even on 256-token WikiText windows with V=50,257.
    """
    model.eval()
    ln_f, lm_head = _get_lens_components(model)
    num_layers = jacobians.shape[0] - 1
    logit_kl_sums = torch.zeros(num_layers + 1, dtype=torch.float64)
    jac_kl_sums = torch.zeros(num_layers + 1, dtype=torch.float64)
    total_positions = 0
    seen_seqs = 0

    for input_ids, attention_mask in batches:
        if seen_seqs >= n_sequences:
            break
        batch_size = input_ids.shape[0]
        for s_idx in range(batch_size):
            if seen_seqs >= n_sequences:
                break
            ids = input_ids[s_idx : s_idx + 1]
            mask = attention_mask[s_idx : s_idx + 1]

            hidden_states = run_model(model, ids, mask)
            # Next-token prediction positions 0 .. T-2 (skip last position per CONTRACTS.md 9.1)
            valid_mask = (mask[:, :-1] == 1).to(hidden_states[-1].device)
            n_valid = int(valid_mask.sum().item())
            if n_valid == 0:
                continue

            final_h = hidden_states[-1][:, :-1, :][valid_mask]
            final_logits = lm_head(final_h)

            for layer_idx in range(num_layers + 1):
                h_pos = hidden_states[layer_idx][:, :-1, :][valid_mask]
                if layer_idx < num_layers:
                    l_pos = lm_head(ln_f(h_pos))
                    j_l = jacobians[layer_idx].to(device=h_pos.device, dtype=torch.float32)
                    j_pos = lm_head(ln_f(torch.matmul(h_pos.float(), j_l.T).to(dtype=h_pos.dtype)))
                else:
                    l_pos = final_logits
                    j_pos = final_logits

                logit_kl_sums[layer_idx] += (
                    kl_to_final(l_pos, final_logits).detach().cpu().double().sum().item()
                )
                jac_kl_sums[layer_idx] += (
                    kl_to_final(j_pos, final_logits).detach().cpu().double().sum().item()
                )

            del hidden_states, final_h, final_logits
            total_positions += n_valid
            seen_seqs += 1

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_logit = (logit_kl_sums / max(total_positions, 1)).tolist()
    mean_jac = (jac_kl_sums / max(total_positions, 1)).tolist()

    print(f"\n### {title} — Per-Layer KL to Final ({seen_seqs} sequences, {total_positions:,} positions)\n")
    print(f"| {'Layer':<5} | {'Logit Lens KL':<13} | {'Jacobian Lens KL':<16} | {'Reduction (%)':<13} |")
    print(f"|{'-'*7}|{'-'*15}|{'-'*18}|{'-'*15}|")
    for layer_idx in range(num_layers + 1):
        kl_l = mean_logit[layer_idx]
        kl_j = mean_jac[layer_idx]
        red = ((kl_l - kl_j) / kl_l * 100.0) if kl_l > 1e-12 else 0.0
        print(f"| {layer_idx:<5} | {kl_l:<13.4f} | {kl_j:<16.4f} | {red:>11.2f}%  |")

    return mean_logit, mean_jac


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate Jacobian Lens matrices for autoregressive decoder models (Issue L5)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["gpt2", "geppetto"],
        help="Model registry key (gpt2 or geppetto).",
    )
    parser.add_argument(
        "--treebank",
        type=str,
        default=None,
        help="Treebank key (default: isdt for geppetto, wikitext for gpt2).",
    )
    parser.add_argument(
        "--n-positions",
        type=int,
        default=128,
        help="Number of positions to sample and average (default: 128).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Basis chunk size for batched autograd (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/jacobian_lens",
        help="Directory where checkpoint is saved (default: checkpoints/jacobian_lens).",
    )
    parser.add_argument(
        "--evaluate-kl",
        action="store_true",
        help="Evaluate per-layer KL to final on 50 calibration sequences vs Logit Lens.",
    )
    parser.add_argument(
        "--n-eval-sequences",
        type=int,
        default=50,
        help="Number of sequences for KL evaluation (default: 50).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-estimation even if checkpoint exists.",
    )

    args = parser.parse_args()

    # Resolve dataset/treebank
    if args.treebank is None:
        args.treebank = "isdt" if args.model == "geppetto" else "wikitext"

    spec = get_model_spec(args.model)
    print(f"Loading model '{args.model}' ({spec['hf_id']})...")
    model, tokenizer, meta = load_model(
        checkpoint=spec["hf_id"],
        kind=spec["kind"],
        tokenizer_kwargs={"add_prefix_space": True},
    )
    print(f"Model loaded on device: {meta['device']}")

    # Build calibration batches
    print(f"Building calibration batches for '{args.model}' using '{args.treebank}' train split...")
    if args.model == "geppetto":
        batches = build_isdt_batches(tokenizer, batch_size=8, max_length=128)
    else:
        batches = build_wikitext_batches(tokenizer, window_size=256, max_windows=200, batch_size=4)
    print(f"Built {len(batches)} calibration batches.")

    ckpt_path = Path(args.output_dir)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    target_file = ckpt_path / f"{args.model}.pt"

    if target_file.exists() and not args.force:
        print(f"Existing checkpoint found at: {target_file}")
        print("Loading checkpoint (use --force to re-estimate)...")
        jacobians = load_jacobian_checkpoint(args.model, checkpoint_dir=args.output_dir)
    else:
        print(
            f"Estimating Jacobians over {args.n_positions} positions "
            f"(chunk_size={args.chunk_size}, seed={args.seed})..."
        )
        t0 = time.perf_counter()
        jacobians = estimate_jacobians(
            model,
            batches,
            n_positions=args.n_positions,
            seed=args.seed,
            chunk_size=args.chunk_size,
            verbose=True,
        )
        total_time = time.perf_counter() - t0
        saved = save_jacobian_checkpoint(jacobians, args.model, output_dir=args.output_dir)
        print(f"Checkpoint saved to: {saved}")
        print(f"Estimation finished in {total_time:.2f}s ({total_time / args.n_positions:.3f}s/position).")

    if args.evaluate_kl:
        print(f"\nEvaluating per-layer KL to final on {args.n_eval_sequences} calibration sequences...")
        title = f"{args.model.upper()} ({args.treebank.upper()} Train)"
        evaluate_jacobian_vs_logit_kl(
            model,
            jacobians,
            batches,
            n_sequences=args.n_eval_sequences,
            title=title,
        )


if __name__ == "__main__":
    main()
