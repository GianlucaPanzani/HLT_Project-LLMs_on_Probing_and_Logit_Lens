"""Tuned Lens for decoder models (Issue L3).

This module implements the Tuned Lens (Belrose et al., 2023) across all hidden
layers of causal language models. It supports both custom per-layer affine
translators (TunedLensTranslators) initialized to zero and the official
pretrained Tuned Lens checkpoint for GPT-2.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ws4_lens.logit_lens import _get_lens_components


class TunedLensTranslators(nn.Module):
    """Learned per-layer affine translators for the Tuned Lens.

    Each translator computes:
        h_tilde = h + A_l h + b_l

    The linear layers are initialized with zero weights and zero biases so
    that the translator begins as an exact identity mapping, strictly matching
    the Standard Logit Lens at initialization.

    Parameters
    ----------
    num_layers : int
        Number of transformer layers to translate (layers 0 .. num_layers - 1).
    hidden_size : int
        Hidden dimension size (d_model).
    """

    def __init__(self, num_layers: int, hidden_size: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.translators = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])

        # Zero-initialization: A_l = 0, b_l = 0
        for linear in self.translators:
            nn.init.zeros_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(self, layer_idx: int, h: torch.Tensor) -> torch.Tensor:
        """Apply translator for a given layer index.

        Parameters
        ----------
        layer_idx : int
            Layer index. If layer_idx >= num_layers (e.g. final layer L),
            the input hidden state is returned unchanged (identity).
        h : torch.Tensor
            Hidden state tensor of shape [..., hidden_size].

        Returns
        -------
        torch.Tensor
            Translated hidden state tensor of shape [..., hidden_size].
        """
        if layer_idx < self.num_layers:
            return h + self.translators[layer_idx](h)
        return h

    def __getitem__(self, idx: int) -> nn.Linear:
        """Access translator linear module for a given layer index."""
        return self.translators[idx]

    def __len__(self) -> int:
        """Return the number of layer translators."""
        return self.num_layers


def load_official_tuned_lens(model: nn.Module) -> Any:
    """Load the official pretrained Tuned Lens for GPT-2 (Belrose et al.).

    Pulls the official checkpoint from the Hugging Face Hub via the
    `tuned-lens` library (`lens_resource_id="gpt2"`) and aligns it with the
    target model's device.

    Parameters
    ----------
    model : nn.Module
        Causal decoder model (e.g. GPT2LMHeadModel).

    Returns
    -------
    object
        Official TunedLens instance placed on the model's device.

    Raises
    ------
    ImportError
        If `tuned_lens` package is not installed.
    """
    try:
        from tuned_lens.nn.lenses import TunedLens
    except ImportError as err:
        raise ImportError(
            "The `tuned-lens` package is required to load official checkpoints. "
            "Please install it with `pip install tuned-lens`."
        ) from err

    lens = TunedLens.from_model_and_pretrained(model, lens_resource_id="gpt2")

    # Align device
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")

    return lens.to(device)


def tuned_lens(
    model: nn.Module,
    translators: Any,
    hidden_states: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Compute Tuned Lens logits across all layers 0 .. L.

    Accepts either:
    1. A custom `TunedLensTranslators` module (Section 10 of CONTRACTS.md).
    2. An official `TunedLens` instance from the `tuned-lens` library.

    For intermediate layers (l < L), the representations are translated
    and projected onto the vocabulary space.
    For the final layer (l = L), the model unembedding head is applied
    directly to `hidden_states[L]`, preserving exact agreement with the model
    logits (matching the convention in `logit_lens`).

    Parameters
    ----------
    model : nn.Module
        Causal decoder model (e.g. GPT2LMHeadModel).
    translators : Any
        TunedLensTranslators instance or official TunedLens instance.
    hidden_states : list[torch.Tensor]
        List of L + 1 hidden state tensors from `run_model`, each of shape
        [B, T, d].

    Returns
    -------
    list[torch.Tensor]
        List of L + 1 logit tensors, each of shape [B, T, V].

    Raises
    ------
    ValueError
        If hidden_states is empty.
    """
    if not hidden_states:
        raise ValueError("hidden_states list must not be empty.")

    ln_f, lm_head = _get_lens_components(model)
    num_layers = len(hidden_states) - 1  # L
    is_official_lens = hasattr(translators, "layer_translators")

    logits: list[torch.Tensor] = []
    for layer_idx, h in enumerate(hidden_states):
        if layer_idx < num_layers:
            if is_official_lens:
                # Official TunedLens applies translator, LayerNorm, and unembedding
                logits_l = translators(h, layer_idx)
            elif isinstance(translators, TunedLensTranslators):
                # Custom TunedLensTranslators: h_tilde = h + A_l h + b_l
                h_tilde = translators(layer_idx, h)
                logits_l = lm_head(ln_f(h_tilde))
            elif callable(translators):
                # Callable translator callback: translates hidden state h -> h_tilde
                h_tilde = translators(layer_idx, h)
                logits_l = lm_head(ln_f(h_tilde))
            else:
                raise TypeError(
                    f"Unsupported translators type: {type(translators).__name__}. "
                    "Expected TunedLensTranslators or official TunedLens."
                )
        else:
            # Layer L convention: in Hugging Face, hidden_states[L] already has ln_f applied.
            # Applying lm_head directly matches the model output bit-for-bit.
            logits_l = lm_head(h)

        logits.append(logits_l)

    return logits
