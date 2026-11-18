"""Evaluation engine for decoder model lenses (Issue L7).

This module runs the full evaluation spectrum across:
1. Standard Logit Lens (nostalgebraist, 2020)
2. Tuned Lens (Belrose et al., 2023)
3. Jacobian Lens (Gurnee et al., 2026)

Across evaluation datasets (PUD EN, PUD IT, WikiText-103 test split), computing:
- Next-token Top-1 accuracy
- Mean Shannon entropy (nats)
- Mean Kullback-Leibler divergence to final layer P_L (nats)
- Next-token UPOS bridge Top-1 accuracy (on PUD treebanks)
- Per-token trajectory records for the first 100 sequences (for Streamlit demo)

Follows CONTRACTS.md Sections 9.1, 9.3, and 9.4.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from lib.alignment import encode_words
from lib.metrics import relative_depth
from lib.word_table import load_word_table, table_to_sentences
from ws4_lens.logit_lens import _get_lens_components
from ws4_lens.metrics import entropy, kl_to_final
from ws4_lens.tuned_lens import TunedLensTranslators
from ws4_lens.upos_bridge import token_to_upos


def prepare_pud_batches(
    treebank: str,
    tokenizer: Any,
    *,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Prepare PUD evaluation batches with word-to-subword alignment for UPOS evaluation.

    Parameters
    ----------
    treebank : str
        Treebank registry key (e.g. 'pud_en', 'pud_it').
    tokenizer : Any
        PreTrainedTokenizer with `add_prefix_space=True`.
    batch_size : int, default=16
        Batch size.

    Returns
    -------
    list[dict[str, Any]]
        List of batch dicts containing 'input_ids', 'attention_mask', 'seq_ids',
        and 'word_alignments'.
    """
    table = load_word_table(treebank, "test")
    sentences = table_to_sentences(table)
    sent_groups = dict(tuple(table.groupby("sent_id", sort=False)))

    batches: list[dict[str, Any]] = []
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i : i + batch_size]
        seq_ids = [s_id for s_id, _ in chunk]
        batch_words = [words for _, words in chunk]

        encoded, word_ids_batch = encode_words(
            tokenizer,
            batch_words,
            add_bos=True,
            max_length=512,
        )

        # Build word alignments: for each word with non-null next_upos, find its last subword pos
        word_alignments: list[list[tuple[int, str]]] = []
        for b_idx, s_id in enumerate(seq_ids):
            df_sent = sent_groups[s_id]
            word_ids = word_ids_batch[b_idx]
            n_words = len(df_sent)

            last_subword_pos: dict[int, int] = {}
            for pos, w_id in enumerate(word_ids):
                if w_id is not None and w_id < n_words:
                    last_subword_pos[w_id] = pos

            next_upos_list = df_sent["next_upos"].tolist()
            sent_alignment: list[tuple[int, str]] = []
            for w_idx in range(n_words):
                gold_next = next_upos_list[w_idx]
                if pd.isna(gold_next) or gold_next is None:
                    continue
                if w_idx in last_subword_pos:
                    sent_alignment.append((last_subword_pos[w_idx], str(gold_next)))
            word_alignments.append(sent_alignment)

        batches.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "seq_ids": seq_ids,
            "word_alignments": word_alignments,
        })

    return batches


def prepare_wikitext_batches(
    tokenizer: Any,
    *,
    window_size: int = 256,
    max_windows: int = 200,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Stream WikiText-103 test split into non-overlapping 256-token windows.

    Parameters
    ----------
    tokenizer : Any
        PreTrainedTokenizer.
    window_size : int, default=256
        Window size in tokens.
    max_windows : int, default=200
        Maximum windows to evaluate (up to ~50k positions per CONTRACTS.md 9.1).
    batch_size : int, default=8
        Batch size.

    Returns
    -------
    list[dict[str, Any]]
        List of batch dicts containing 'input_ids', 'attention_mask', and 'seq_ids'.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="test",
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

    batches: list[dict[str, Any]] = []
    for i in range(0, len(windows), batch_size):
        chunk = torch.stack(windows[i : i + batch_size], dim=0)
        mask = torch.ones_like(chunk)
        seq_ids = [f"win_{i + idx}" for idx in range(len(chunk))]
        batches.append({
            "input_ids": chunk,
            "attention_mask": mask,
            "seq_ids": seq_ids,
        })
    return batches


@torch.no_grad()
def evaluate_lenses(
    model: nn.Module,
    tokenizer: Any,
    *,
    model_name: str,
    dataset_name: str,
    data_batches: list[dict[str, Any]],
    tuned_translators: Any = None,
    jacobian_matrices: torch.Tensor | None = None,
    upos_lexicon: dict[str, str] | None = None,
    lenses: tuple[str, ...] = ("logit", "tuned", "jacobian"),
    max_token_sequences: int = 100,
    device: torch.device | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run requested lenses over a dataset, returning aggregate and per-token DataFrames.

    Evaluates layer-by-layer and reduces logits immediately to ensure low memory usage.

    Parameters
    ----------
    model : nn.Module
        Causal decoder model.
    tokenizer : Any
        Matching tokenizer.
    model_name : str
        Model registry key (e.g. 'gpt2', 'geppetto').
    dataset_name : str
        Dataset key ('pud_en', 'pud_it', 'wikitext').
    data_batches : list[dict[str, Any]]
        Batches containing 'input_ids', 'attention_mask', 'seq_ids', and optional alignments.
    tuned_translators : object or None, optional
        Pretrained Tuned Lens translators (custom TunedLensTranslators or official TunedLens).
    jacobian_matrices : torch.Tensor or None, optional
        Jacobian tensor of shape [L + 1, d, d].
    upos_lexicon : dict[str, str] or None, optional
        Word form to UPOS lexicon for UPOS bridge accuracy (PUD only).
    lenses : tuple[str, ...], default=("logit", "tuned", "jacobian")
        Tuple of lens keys to evaluate.
    max_token_sequences : int, default=100
        Maximum number of sequences to record in per-token outputs.
    device : torch.device or None, optional
        Target computation device.
    verbose : bool, default=True
        Whether to log progress.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_aggregate, df_tokens) exactly matching CONTRACTS.md Sections 9.3 and 9.4.

    Raises
    ------
    ValueError
        If required translator/Jacobian components are missing for a requested lens.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    ln_f, lm_head = _get_lens_components(model)
    num_layers = getattr(model.config, "n_layer", None) or getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        raise AttributeError("Could not resolve num_layers from model config.")

    is_official_tuned = (
        tuned_translators is not None
        and not isinstance(tuned_translators, TunedLensTranslators)
        and hasattr(tuned_translators, "layer_translators")
    )

    # Token and UPOS caches for fast string decoding
    decode_cache: dict[int, str] = {}
    upos_cache: dict[int, str] = {}

    def get_decoded(tid: int) -> str:
        s = decode_cache.get(tid)
        if s is None:
            s = tokenizer.decode([tid])
            decode_cache[tid] = s
        return s

    def get_upos_tag(tid: int) -> str:
        tag = upos_cache.get(tid)
        if tag is None:
            tok_str = get_decoded(tid)
            tag = token_to_upos(tok_str, upos_lexicon)
            upos_cache[tid] = tag
        return tag

    # Aggregators per (lens, layer)
    agg_correct: dict[tuple[str, int], int] = {}
    agg_total: dict[tuple[str, int], int] = {}
    agg_entropy_sum: dict[tuple[str, int], float] = {}
    agg_kl_sum: dict[tuple[str, int], float] = {}
    agg_upos_correct: dict[tuple[str, int], int] = {}
    agg_upos_total: dict[tuple[str, int], int] = {}

    for l_key in lenses:
        for lyr in range(num_layers + 1):
            pair = (l_key, lyr)
            agg_correct[pair] = 0
            agg_total[pair] = 0
            agg_entropy_sum[pair] = 0.0
            agg_kl_sum[pair] = 0.0
            agg_upos_correct[pair] = 0
            agg_upos_total[pair] = 0

    token_records: list[dict[str, Any]] = []
    seqs_recorded = 0

    t_start = time.perf_counter()
    total_batches = len(data_batches)

    for b_idx, batch in enumerate(data_batches, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        seq_ids = batch["seq_ids"]
        word_alignments = batch.get("word_alignments", [None] * len(seq_ids))

        b_size, seq_len = input_ids.shape
        if seq_len < 2:
            continue

        # 1. Forward pass once: extract all hidden states [B, T, d]
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = list(outputs.hidden_states)

        # Valid prediction positions 0 .. T-2
        mask_pred = (attention_mask[:, :-1] == 1) & (attention_mask[:, 1:] == 1)
        targets = input_ids[:, 1:]  # [B, T-1]

        # Precompute final logits [B, T-1, V] for Layer L and for KL reference
        final_logits = lm_head(hidden_states[-1][:, :-1, :])

        # 2. Evaluate each lens and layer
        for l_key in lenses:
            for lyr in range(num_layers + 1):
                h_l = hidden_states[lyr][:, :-1, :]

                if lyr == num_layers:
                    logits_l = final_logits
                else:
                    if l_key == "logit":
                        logits_l = lm_head(ln_f(h_l))
                    elif l_key == "tuned":
                        if is_official_tuned:
                            logits_l = tuned_translators(h_l, lyr)
                        elif tuned_translators is not None:
                            h_tilde = tuned_translators(h_l, lyr)
                            logits_l = lm_head(ln_f(h_tilde))
                        else:
                            raise ValueError("Tuned lens requested but no translators provided.")
                    elif l_key == "jacobian":
                        if jacobian_matrices is None:
                            raise ValueError("Jacobian lens requested but no matrices provided.")
                        j_l = jacobian_matrices[lyr].to(device=device, dtype=torch.float32)
                        h_proj = torch.matmul(h_l.float(), j_l.T).to(dtype=h_l.dtype)
                        logits_l = lm_head(ln_f(h_proj))
                    else:
                        raise ValueError(f"Unknown lens: {l_key}")

                # Compute metrics on valid prediction positions
                valid_logits = logits_l[mask_pred]       # [N_valid, V]
                valid_targets = targets[mask_pred]       # [N_valid]
                valid_final = final_logits[mask_pred]    # [N_valid, V]

                n_v = valid_targets.numel()
                if n_v > 0:
                    top1_ids = valid_logits.argmax(dim=-1)
                    n_correct = int((top1_ids == valid_targets).sum().item())

                    ent_sum = float(entropy(valid_logits).detach().cpu().double().sum().item())
                    if lyr == num_layers:
                        kl_sum = 0.0
                    else:
                        kl_sum = float(
                            kl_to_final(valid_logits, valid_final).detach().cpu().double().sum().item()
                        )

                    pair = (l_key, lyr)
                    agg_correct[pair] += n_correct
                    agg_total[pair] += n_v
                    agg_entropy_sum[pair] += ent_sum
                    agg_kl_sum[pair] += kl_sum

                # UPOS Bridge Evaluation on PUD words
                if upos_lexicon is not None:
                    for b_i in range(b_size):
                        alignment = word_alignments[b_i]
                        if not alignment:
                            continue
                        for w_pos, gold_tag in alignment:
                            if w_pos < seq_len - 1 and mask_pred[b_i, w_pos]:
                                pred_id = int(logits_l[b_i, w_pos, :].argmax().item())
                                pred_tok = tokenizer.decode([pred_id])
                                pred_tag = token_to_upos(pred_tok, upos_lexicon)
                                pair = (l_key, lyr)
                                agg_upos_total[pair] += 1
                                if pred_tag == gold_tag:
                                    agg_upos_correct[pair] += 1

                # Record per-token results for the first max_token_sequences
                if seqs_recorded < max_token_sequences:
                    # Memory-efficient top-1: avoid materialising full [B, T, V] softmax.
                    # top1_prob = exp(max_logit - logsumexp(logits)) is equivalent to
                    # softmax(logits).max() but never allocates the full probability tensor.
                    top1_logits, top1_indices = logits_l.max(dim=-1)  # [B, T-1]
                    top1_probs = (top1_logits - torch.logsumexp(logits_l, dim=-1)).exp()
                    ent_per_pos = entropy(logits_l)
                    if lyr == num_layers:
                        kl_per_pos = torch.zeros_like(ent_per_pos)
                    else:
                        kl_per_pos = kl_to_final(logits_l, final_logits)

                    for b_i in range(b_size):
                        global_seq_idx = (b_idx - 1) * b_size + b_i
                        if global_seq_idx >= max_token_sequences:
                            continue
                        s_id = str(seq_ids[b_i])

                        for p in range(seq_len - 1):
                            if not mask_pred[b_i, p]:
                                continue
                            tok_str = tokenizer.decode([int(input_ids[b_i, p].item())])
                            tgt_str = tokenizer.decode([int(targets[b_i, p].item())])
                            t1_id = int(top1_indices[b_i, p].item())
                            t1_tok = tokenizer.decode([t1_id])
                            t1_pr = float(top1_probs[b_i, p].item())
                            e_val = float(ent_per_pos[b_i, p].item())
                            k_val = float(kl_per_pos[b_i, p].item())

                            token_records.append({
                                "seq_id": s_id,
                                "pos": p,
                                "token": tok_str,
                                "target_token": tgt_str,
                                "lens": l_key,
                                "layer": lyr,
                                "top1_token": t1_tok,
                                "top1_prob": t1_pr,
                                "entropy": e_val,
                                "kl": k_val,
                            })

                if lyr < num_layers:
                    del logits_l

        del outputs, hidden_states, final_logits, input_ids, attention_mask, targets, mask_pred
        seqs_recorded = min(max_token_sequences, b_idx * b_size)

        if verbose and (b_idx % 10 == 0 or b_idx == total_batches):
            if device.type == "mps" and hasattr(torch, "mps"):
                torch.mps.empty_cache()
            elif device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elapsed = time.perf_counter() - t_start
            print(f"  Processed batch {b_idx:>3}/{total_batches} in {elapsed:.1f}s")

    # 3. Construct Aggregate DataFrame per CONTRACTS.md Section 9.3
    agg_rows: list[dict[str, Any]] = []
    for l_key in lenses:
        for lyr in range(num_layers + 1):
            pair = (l_key, lyr)
            n_pos = agg_total[pair]
            acc = (agg_correct[pair] / n_pos) if n_pos > 0 else 0.0
            e_mean = (agg_entropy_sum[pair] / n_pos) if n_pos > 0 else 0.0
            k_mean = (agg_kl_sum[pair] / n_pos) if n_pos > 0 else 0.0

            if upos_lexicon is not None and agg_upos_total[pair] > 0:
                u_acc = agg_upos_correct[pair] / agg_upos_total[pair]
            else:
                u_acc = None

            agg_rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "lens": l_key,
                "layer": lyr,
                "L": num_layers,
                "tau": float(relative_depth(lyr, num_layers)),
                "top1_acc": float(acc),
                "entropy_mean": float(e_mean),
                "kl_mean": float(k_mean),
                "upos_top1_acc": u_acc,
                "n_positions": int(n_pos),
            })

    df_aggregate = pd.DataFrame(agg_rows)
    df_tokens = pd.DataFrame(token_records)

    return df_aggregate, df_tokens
