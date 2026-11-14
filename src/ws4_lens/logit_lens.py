"""Standard Logit Lens for decoder models (Issue L1).

This module implements the Standard Logit Lens (nostalgebraist, 2020) across all
hidden layers of causal language models such as GPT-2 and GePpeTto, projecting
intermediate representations to the vocabulary space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _get_lens_components(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """
    Extract the final layer normalization (ln_f) and unembedding head (lm_head)
    from a causal language model.

    Parameters
    ----------
    model : nn.Module
        Causal decoder model (e.g. GPT2LMHeadModel).

    Returns
    -------
    tuple[nn.Module, nn.Module]
        Pair of (ln_f, lm_head).

    Raises
    ------
    AttributeError
        If either final layer normalization or output head cannot be found.
    """
    # Resolve final layer normalization
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        ln_f = model.transformer.ln_f
    elif hasattr(model, "ln_f"):
        ln_f = model.ln_f
    elif hasattr(model, "model") and hasattr(model.model, "norm"):
        ln_f = model.model.norm
    else:
        raise AttributeError(
            f"Could not locate final layer normalization (ln_f) on model {type(model).__name__}."
        )

    # Resolve unembedding head
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
    elif hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
        lm_head = model.get_output_embeddings()
    else:
        raise AttributeError(
            f"Could not locate output unembedding head (lm_head) on model {type(model).__name__}."
        )

    return ln_f, lm_head


def run_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """
    Execute a causal language model and extract all intermediate hidden states.

    Parameters
    ----------
    model : nn.Module
        Pretrained or initialized causal decoder model.
    input_ids : torch.Tensor
        Token IDs tensor of shape [batch_size, seq_len].
    attention_mask : torch.Tensor | None
        Attention mask tensor of shape [batch_size, seq_len]. If None, all
        tokens are attended to. Defaults to None.

    Returns
    -------
    list[torch.Tensor]
        List of L + 1 tensors of shape [batch_size, seq_len, hidden_size],
        corresponding to layer 0 (static embeddings) through layer L.

    Raises
    ------
    ValueError
        If the model output does not contain hidden states.
    """
    model.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = getattr(model, "device", input_ids.device)

    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise ValueError(
            "Model outputs do not contain 'hidden_states'. "
            "Ensure the model was invoked with output_hidden_states=True."
        )

    return list(hidden_states)


def logit_lens(
    model: nn.Module,
    hidden_states: list[torch.Tensor],
) -> list[torch.Tensor]:
    """
    Compute Standard Logit Lens logits across all hidden layers.

    For layers 0 <= l < L, the representation is first normalized using the
    model's final layer normalization (ln_f) and then projected via lm_head:
        logits_l = W_U(ln_f(h_l))

    At the final layer l = L, the hidden state h_L from Hugging Face has
    already been normalized by ln_f, so the unembedding head is applied directly:
        logits_L = W_U(h_L)

    Parameters
    ----------
    model : nn.Module
        Causal decoder model with accessible ln_f and lm_head.
    hidden_states : list[torch.Tensor]
        List of L + 1 tensors of shape [batch_size, seq_len, hidden_size].

    Returns
    -------
    list[torch.Tensor]
        List of L + 1 tensors of shape [batch_size, seq_len, vocab_size].

    Raises
    ------
    ValueError
        If hidden_states is empty.
    """
    if not hidden_states:
        raise ValueError("hidden_states list cannot be empty.")

    ln_f, lm_head = _get_lens_components(model)
    num_layers = len(hidden_states) - 1
    logits_list: list[torch.Tensor] = []

    for layer_idx, h_l in enumerate(hidden_states):
        if layer_idx < num_layers:
            # Intermediate layers (0 <= layer_idx < L): apply final LayerNorm, then unembed
            logits_l = lm_head(ln_f(h_l))
        else:
            # Final layer (layer_idx == L): h_L is already post-ln_f in Hugging Face
            logits_l = lm_head(h_l)
        logits_list.append(logits_l)

    return logits_list
