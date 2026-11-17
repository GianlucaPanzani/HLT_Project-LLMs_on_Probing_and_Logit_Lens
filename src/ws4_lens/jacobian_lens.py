"""Jacobian Lens for decoder models (Issue L5).

This module implements the training-free Jacobian Lens (Gurnee et al., 2026;
CONTRACTS.md Section 9.2) across all layers of causal language models such as
GPT-2 and GePpeTto.

The Jacobian matrix for layer l is the mean same-position Jacobian:
    J_l = E_t [ d r_L(t) / d h_l(t) ] in R^{d x d}
where r_L(t) is the final residual stream output of the last transformer block
before the final LayerNorm ln_f (captured via a forward hook on the final block).

The projected logits are computed as:
    logits_l = W_U(ln_f(h_l @ J_l.T))  for l < L
    logits_L = W_U(h_L)                for l = L  (J_L = I_d)
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lib.paths import ROOT
from ws4_lens.logit_lens import _get_lens_components


def _get_jacobian_modules(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Locate the embedding-dropout entry module and the final transformer block.

    Parameters
    ----------
    model : nn.Module
        Causal decoder model (e.g. GPT2LMHeadModel).

    Returns
    -------
    tuple[nn.Module, nn.Module]
        Pair of (entry_module, last_block), where `entry_module` produces
        `hidden_states[0]` and `last_block` produces `r_L` before `ln_f`.

    Raises
    ------
    AttributeError
        If transformer blocks or entry modules cannot be found on the model.
    ValueError
        If the transformer block list is empty.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
        entry = getattr(model.transformer, "drop", None)
    elif hasattr(model, "h"):
        blocks = model.h
        entry = getattr(model, "drop", None)
    else:
        raise AttributeError(
            f"Could not locate transformer blocks (transformer.h) on {type(model).__name__}."
        )

    if len(blocks) == 0:
        raise ValueError("Model transformer block list is empty.")
    if entry is None:
        raise AttributeError(
            f"Could not locate embedding dropout module (transformer.drop) on {type(model).__name__}."
        )

    return entry, blocks[-1]


def _check_float32_model(model: nn.Module) -> None:
    """Verify that model parameters are in float32 (never float16/bfloat16).

    Parameters
    ----------
    model : nn.Module
        The neural network model to inspect.

    Raises
    ------
    ValueError
        If any model parameter is float16 or bfloat16.
    """
    for param in model.parameters():
        if param.dtype in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"Jacobian estimation must be computed in float32, never {param.dtype}. "
                "Convert the model to float32 before calling estimate_jacobians."
            )
        break


def estimate_jacobians(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    n_positions: int = 512,
    seed: int = 42,
    chunk_size: int = 64,
    verbose: bool = True,
) -> torch.Tensor:
    """Estimate mean per-layer same-position Jacobians J_l = E[d r_L(t) / d h_l(t)].

    Computes first-order residual stream sensitivity matrices across all layers
    simultaneously in float32 using batched vector-Jacobian products.
    Causal prefix truncation is applied so that position t is evaluated on prefix
    tokens 0..t, ensuring bit-for-bit mathematical identity while reducing
    attention compute from O(T^2) to O(t^2).

    Parameters
    ----------
    model : nn.Module
        Frozen causal decoder model in float32.
    batches : Iterable[tuple[torch.Tensor, torch.Tensor]]
        Iterable yielding `(input_ids, attention_mask)` batches of shape `[B, T]`.
    n_positions : int, default=512
        Number of valid non-BOS, non-padding positions to sample and average.
    seed : int, default=42
        Random seed for `numpy.random.default_rng(seed)`.
    chunk_size : int, default=64
        Number of identity basis rows per batched `torch.autograd.grad` call.
    verbose : bool, default=True
        Whether to log progress and time per position.

    Returns
    -------
    torch.Tensor
        Float32 CPU tensor of shape `[L + 1, d, d]` with exact identity at `J[L]`.

    Raises
    ------
    ValueError
        If `n_positions <= 0`, `chunk_size <= 0`, or no valid positions are found.
    AttributeError
        If model architecture or dimensions cannot be resolved.
    """
    if n_positions <= 0:
        raise ValueError(f"n_positions must be positive, got {n_positions}.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    _check_float32_model(model)
    model.eval()

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    entry_module, last_block = _get_jacobian_modules(model)
    bos_token_id = getattr(getattr(model, "config", None), "bos_token_id", None)

    # 1. Materialize CPU token batches and collect all valid (batch_idx, b, t) positions
    stored_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    candidates: list[tuple[int, int, int]] = []

    for batch_idx, (input_ids, attention_mask) in enumerate(batches):
        ids_cpu = input_ids.detach().to("cpu")
        mask_cpu = attention_mask.detach().to("cpu")
        stored_batches.append((ids_cpu, mask_cpu))

        valid = mask_cpu == 1
        # Exclude position 0 (BOS or zero-context position)
        valid[:, 0] = False
        # Exclude explicit BOS tokens anywhere in the sequence
        if bos_token_id is not None:
            valid &= ids_cpu != bos_token_id

        coords = valid.nonzero(as_tuple=False)
        for b_val, t_val in coords.tolist():
            candidates.append((batch_idx, b_val, t_val))

    total_valid = len(candidates)
    if total_valid == 0:
        raise ValueError("No valid non-BOS, non-padding positions found in batches.")

    k = min(n_positions, total_valid)
    rng = np.random.default_rng(seed)
    chosen_indices = rng.choice(total_valid, size=k, replace=False)
    sampled_positions = [candidates[int(idx)] for idx in chosen_indices]

    # Resolve L and d
    num_layers = getattr(model.config, "n_layer", None) or getattr(model.config, "num_hidden_layers", None)
    hidden_size = getattr(model.config, "n_embd", None) or getattr(model.config, "hidden_size", None)
    if num_layers is None or hidden_size is None:
        raise AttributeError("Could not determine num_layers or hidden_size from model.config.")

    eye_d = torch.eye(hidden_size, device=device, dtype=torch.float32)
    j_sum = torch.zeros((num_layers + 1, hidden_size, hidden_size), device="cpu", dtype=torch.float32)

    captured_r_l: list[torch.Tensor] = []

    def _hook_entry(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> torch.Tensor:
        return out.detach().requires_grad_(True)

    def _hook_last_block(_module: nn.Module, _inp: tuple, out: torch.Tensor | tuple) -> None:
        r_tensor = out[0] if isinstance(out, tuple) else out
        captured_r_l.append(r_tensor)

    t_start = time.perf_counter()
    h_entry = entry_module.register_forward_hook(_hook_entry)
    h_last = last_block.register_forward_hook(_hook_last_block)

    try:
        for pos_num, (batch_idx, b_idx, t_idx) in enumerate(sampled_positions, start=1):
            ids_cpu, mask_cpu = stored_batches[batch_idx]
            # Exact causal prefix truncation to 0..t_idx (tokens > t_idx cannot affect r_L[t_idx])
            seq_ids = ids_cpu[b_idx : b_idx + 1, : t_idx + 1].to(device)
            seq_mask = mask_cpu[b_idx : b_idx + 1, : t_idx + 1].to(device)
            seq_len = t_idx + 1

            captured_r_l.clear()
            with torch.enable_grad():
                outputs = model(
                    input_ids=seq_ids,
                    attention_mask=seq_mask,
                    output_hidden_states=True,
                )
                hidden_states = list(outputs.hidden_states[:-1])  # h_0 .. h_{L-1}
                if len(hidden_states) != num_layers:
                    raise RuntimeError(
                        f"Expected {num_layers} pre-final hidden states, got {len(hidden_states)}."
                    )
                if not captured_r_l:
                    raise RuntimeError("Forward hook on last transformer block did not capture r_L.")
                r_l_vec = captured_r_l[0][0, t_idx, :].float()  # [d]

                # Adaptive chunk size on MPS for long prefixes to prevent memory pressure
                eff_chunk = chunk_size
                if device.type == "mps" and seq_len > 128:
                    eff_chunk = min(chunk_size, 16)
                elif device.type == "mps" and seq_len > 64:
                    eff_chunk = min(chunk_size, 32)

                for start in range(0, hidden_size, eff_chunk):
                    end = min(start + eff_chunk, hidden_size)
                    basis_chunk = eye_d[start:end]  # [C, d]
                    retain = end < hidden_size
                    grads = torch.autograd.grad(
                        r_l_vec,
                        hidden_states,
                        grad_outputs=basis_chunk,
                        is_grads_batched=True,
                        retain_graph=retain,
                    )
                    for layer_idx, g in enumerate(grads):
                        # g has shape [C, 1, seq_len, d]; slice position t_idx
                        j_sum[layer_idx, start:end, :] += (
                            g[:, 0, t_idx, :].detach().to("cpu", dtype=torch.float32)
                        )

            del outputs, hidden_states, r_l_vec, seq_ids, seq_mask
            captured_r_l.clear()

            if pos_num % 16 == 0:
                if device.type == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                elif device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if verbose and (pos_num == 1 or pos_num % max(1, k // 8) == 0 or pos_num == k):
                if device.type == "mps" and hasattr(torch, "mps"):
                    torch.mps.synchronize()
                elapsed = time.perf_counter() - t_start
                print(
                    f"  Position {pos_num:>4}/{k} | Elapsed: {elapsed:>7.2f}s | "
                    f"Avg: {elapsed / pos_num:.3f}s/pos | Prefix len: {seq_len}"
                )
    finally:
        h_entry.remove()
        h_last.remove()
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    j_mean = j_sum / float(k)
    j_mean[num_layers] = torch.eye(hidden_size, dtype=torch.float32)

    total_time = time.perf_counter() - t_start
    if verbose:
        print(
            f"Done! Estimated Jacobians [{num_layers + 1}, {hidden_size}, {hidden_size}] "
            f"over {k} positions in {total_time:.2f}s ({total_time / k:.3f}s/pos)."
        )
    return j_mean


def jacobian_lens(
    model: nn.Module,
    jacobians: torch.Tensor,
    hidden_states: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Compute Jacobian Lens logits across all layers 0 .. L.

    For intermediate layers l < L:
        logits_l = lm_head(ln_f(h_l @ J_l.T))
    For final layer l = L:
        logits_L = lm_head(h_L)  (exact match with model(...).logits)

    Parameters
    ----------
    model : nn.Module
        Causal decoder model with accessible ln_f and lm_head.
    jacobians : torch.Tensor
        Tensor of shape `[L + 1, d, d]` from `estimate_jacobians`.
    hidden_states : list[torch.Tensor]
        List of `L + 1` hidden state tensors of shape `[B, T, d]`.

    Returns
    -------
    list[torch.Tensor]
        List of `L + 1` logit tensors of shape `[B, T, V]`.

    Raises
    ------
    ValueError
        If `hidden_states` is empty or if dimension mismatches occur with `jacobians`.
    """
    if not hidden_states:
        raise ValueError("hidden_states list must not be empty.")
    if jacobians.ndim != 3:
        raise ValueError(
            f"jacobians must be a 3D tensor [L + 1, d, d], got shape {tuple(jacobians.shape)}."
        )
    if jacobians.shape[0] != len(hidden_states):
        raise ValueError(
            f"Number of Jacobian matrices ({jacobians.shape[0]}) does not match "
            f"len(hidden_states) ({len(hidden_states)})."
        )

    hidden_size = hidden_states[0].shape[-1]
    if jacobians.shape[1] != hidden_size or jacobians.shape[2] != hidden_size:
        raise ValueError(
            f"Jacobian matrix dimensions {tuple(jacobians.shape[1:])} do not match "
            f"hidden_size {hidden_size}."
        )

    ln_f, lm_head = _get_lens_components(model)
    num_layers = len(hidden_states) - 1
    logits: list[torch.Tensor] = []

    with torch.no_grad():
        for layer_idx, h in enumerate(hidden_states):
            if layer_idx < num_layers:
                j_l = jacobians[layer_idx].to(device=h.device, dtype=torch.float32)
                h_proj = torch.matmul(h.float(), j_l.T).to(dtype=h.dtype)
                logits_l = lm_head(ln_f(h_proj))
            else:
                logits_l = lm_head(h)
            logits.append(logits_l)

    return logits


def save_jacobian_checkpoint(
    jacobians: torch.Tensor,
    model_name: str,
    output_dir: str | Path = "checkpoints/jacobian_lens",
) -> Path:
    """Save Jacobian tensor `[L + 1, d, d]` per CONTRACTS.md Section 9.5.

    Parameters
    ----------
    jacobians : torch.Tensor
        Estimated Jacobian tensor of shape `[L + 1, d, d]`.
    model_name : str
        Model registry key (e.g. "gpt2" or "geppetto").
    output_dir : str | Path, default="checkpoints/jacobian_lens"
        Directory where checkpoint `.pt` file is written.

    Returns
    -------
    Path
        Path to the saved `.pt` file.
    """
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)
    pt_path = out_path / f"{model_name}.pt"
    torch.save(jacobians.detach().to("cpu", dtype=torch.float32), pt_path)
    return pt_path


def load_jacobian_checkpoint(
    model_name: str,
    checkpoint_dir: str | Path = "checkpoints/jacobian_lens",
) -> torch.Tensor:
    """Load Jacobian tensor `[L + 1, d, d]` from checkpoint directory.

    Parameters
    ----------
    model_name : str
        Model registry key (e.g. "gpt2" or "geppetto").
    checkpoint_dir : str | Path, default="checkpoints/jacobian_lens"
        Directory containing the checkpoint file.

    Returns
    -------
    torch.Tensor
        Float32 CPU tensor of shape `[L + 1, d, d]`.

    Raises
    ------
    FileNotFoundError
        If the checkpoint `.pt` file does not exist.
    """
    p_dir = Path(checkpoint_dir)
    if not p_dir.is_absolute() and not (p_dir / f"{model_name}.pt").exists():
        if (ROOT / p_dir / f"{model_name}.pt").exists():
            p_dir = ROOT / p_dir
    pt_path = p_dir / f"{model_name}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"Jacobian checkpoint not found at: {pt_path}")
    return torch.load(pt_path, map_location="cpu", weights_only=True)
