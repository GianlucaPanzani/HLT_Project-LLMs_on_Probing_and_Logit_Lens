"""Tuned Lens for decoder models (Issues L3 and L4).

This module implements the Tuned Lens (Belrose et al., 2023) across all hidden
layers of causal language models. It supports custom per-layer affine
translators (TunedLensTranslators) initialized to zero, official pretrained
Tuned Lens checkpoints (GPT-2), and self-supervised training on UD treebanks (L4).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.alignment import encode_words
from ws4_lens.logit_lens import _get_lens_components, run_model


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

    def forward(
        self,
        arg1: torch.Tensor | int,
        arg2: torch.Tensor | int,
    ) -> torch.Tensor:
        """Apply translator for a given layer index.

        Supports both `forward(h, layer_idx)` (CONTRACTS.md Section 10)
        and `forward(layer_idx, h)`.

        Parameters
        ----------
        arg1 : torch.Tensor | int
            Hidden state tensor `h` [..., hidden_size] or `layer_idx`.
        arg2 : torch.Tensor | int
            `layer_idx` or hidden state tensor `h` [..., hidden_size].

        Returns
        -------
        torch.Tensor
            Translated hidden state tensor of shape [..., hidden_size].
        """
        if isinstance(arg1, torch.Tensor):
            h = arg1
            layer_idx = int(arg2)
        else:
            layer_idx = int(arg1)
            h = arg2

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
                h_tilde = translators(h, layer_idx)
                logits_l = lm_head(ln_f(h_tilde))
            elif callable(translators):
                # Callable translator callback: translates hidden state h -> h_tilde
                h_tilde = translators(h, layer_idx)
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


@torch.no_grad()
def evaluate_dev_kl(
    model: nn.Module,
    sentences: list[tuple[str, list[str]]],
    tokenizer: Any,
    translators: TunedLensTranslators | None = None,
    *,
    batch_size: int = 16,
    max_length: int = 128,
    max_sentences: int | None = None,
) -> list[float]:
    """Compute mean KL divergence to final layer P_L per layer on a dataset.

    Parameters
    ----------
    model : nn.Module
        Causal decoder model.
    sentences : list[tuple[str, list[str]]]
        List of (sent_id, words) tuples from `table_to_sentences`.
    tokenizer : Any
        Tokenizer configured with `add_prefix_space=True`.
    translators : TunedLensTranslators or None, optional
        If provided, evaluates the Tuned Lens. If None, evaluates baseline Logit Lens.
    batch_size : int, default=16
        Evaluation batch size.
    max_length : int, default=128
        Maximum token sequence length.
    max_sentences : int or None, optional
        Maximum number of sentences to evaluate (for fast validation).

    Returns
    -------
    list[float]
        Mean KL divergence D_KL(P_L || P_l) for each layer 0 .. L - 1.
    """
    model.eval()
    if translators is not None:
        translators.eval()

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    ln_f, lm_head = _get_lens_components(model)
    num_layers = getattr(model.config, "n_layer", None) or getattr(model.config, "num_hidden_layers", 12)

    eval_sents = sentences if max_sentences is None else sentences[:max_sentences]
    kl_sums = [0.0] * num_layers
    total_tokens = 0

    for i in range(0, len(eval_sents), batch_size):
        batch = eval_sents[i : i + batch_size]
        batch_words = [words for _, words in batch]

        encoded, _ = encode_words(tokenizer, batch_words, add_bos=True, max_length=max_length)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        valid_mask = attention_mask == 1
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            continue

        hidden_states = run_model(model, input_ids, attention_mask)
        final_h = hidden_states[-1][valid_mask]
        final_logits = lm_head(final_h).float()
        logp_L = F.log_softmax(final_logits, dim=-1)
        p_L = logp_L.exp()

        for layer_idx in range(num_layers):
            h_l = hidden_states[layer_idx][valid_mask]
            h_tilde = translators(h_l, layer_idx) if translators is not None else h_l
            logits_l = lm_head(ln_f(h_tilde)).float()
            logp_l = F.log_softmax(logits_l, dim=-1)
            kl = (p_L * (logp_L - logp_l)).sum(dim=-1).sum().item()
            kl_sums[layer_idx] += kl

        total_tokens += n_valid

    return [kl / max(total_tokens, 1) for kl in kl_sums]


def train_tuned_lens(
    model: nn.Module,
    train_sentences: list[tuple[str, list[str]]],
    dev_sentences: list[tuple[str, list[str]]],
    tokenizer: Any,
    *,
    num_layers: int = 12,
    hidden_size: int = 768,
    batch_size: int = 8,
    max_length: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    epochs: int = 3,
    max_steps: int | None = None,
    log_interval: int = 50,
    empty_cache_interval: int = 50,
    seed: int = 42,
) -> tuple[TunedLensTranslators, dict[str, Any]]:
    """Train TunedLensTranslators on word-table sentences with memory optimization.

    Follows Issue [L4] and CONTRACTS.md Section 9.5:
    - Layer-by-layer loss computation and backward pass to minimize memory footprint.
    - Model weights remain frozen; only translators are optimized with AdamW.
    - Evaluates dev KL per layer per epoch.
    - Periodic cache clearing on MPS devices to avoid memory accumulation.

    Parameters
    ----------
    model : nn.Module
        Causal decoder model (frozen).
    train_sentences : list[tuple[str, list[str]]]
        Training sentences from `table_to_sentences`.
    dev_sentences : list[tuple[str, list[str]]]
        Validation sentences from `table_to_sentences`.
    tokenizer : Any
        Tokenizer configured with `add_prefix_space=True`.
    num_layers : int, default=12
        Number of transformer layers.
    hidden_size : int, default=768
        Hidden dimension size.
    batch_size : int, default=8
        Training batch size.
    max_length : int, default=128
        Maximum token sequence length.
    lr : float, default=1e-3
        Learning rate for AdamW.
    weight_decay : float, default=1e-3
        L2 weight decay for AdamW.
    epochs : int, default=3
        Number of training epochs.
    max_steps : int or None, optional
        Maximum training steps (for dry-runs or debugging).
    log_interval : int, default=50
        Logging step frequency.
    empty_cache_interval : int, default=50
        Step frequency for flushing MPS/CUDA memory caches.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    tuple[TunedLensTranslators, dict[str, Any]]
        Trained translators module and training history dictionary.
    """
    torch.manual_seed(seed)
    random.seed(seed)

    device = getattr(model, "device", None)
    if device is None:
        device = next(model.parameters()).device

    is_mps = device.type == "mps"
    is_cuda = device.type == "cuda"

    model.eval()
    ln_f, lm_head = _get_lens_components(model)

    translators = TunedLensTranslators(num_layers=num_layers, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.AdamW(translators.parameters(), lr=lr, weight_decay=weight_decay)

    print("Computing baseline Logit Lens KL on validation split...")
    baseline_dev_kl = evaluate_dev_kl(model, dev_sentences, tokenizer, translators=None, batch_size=16)
    print(f"Baseline Dev Mean KL across layers 0..{num_layers-1}: {sum(baseline_dev_kl)/len(baseline_dev_kl):.4f}")

    history: dict[str, Any] = {
        "baseline_dev_kl": baseline_dev_kl,
        "train_loss": [],
        "dev_kl_per_epoch": [],
    }

    n_train = len(train_sentences)
    steps_per_epoch = math.ceil(n_train / batch_size)
    global_step = 0
    best_dev_kl = float("inf")

    print(f"Starting Tuned Lens Training ({epochs} epochs, {steps_per_epoch} steps/epoch, batch_size={batch_size})...")

    for epoch in range(1, epochs + 1):
        shuffled_indices = list(range(n_train))
        random.shuffle(shuffled_indices)

        translators.train()
        epoch_loss = 0.0
        batches_in_epoch = 0

        for b_idx in range(steps_per_epoch):
            if max_steps is not None and global_step >= max_steps:
                break

            batch_idx_slice = shuffled_indices[b_idx * batch_size : (b_idx + 1) * batch_size]
            batch_words = [train_sentences[idx][1] for idx in batch_idx_slice]

            encoded, _ = encode_words(tokenizer, batch_words, add_bos=True, max_length=max_length)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            valid_mask = attention_mask == 1
            if valid_mask.sum() == 0:
                continue

            with torch.no_grad():
                hidden_states = run_model(model, input_ids, attention_mask)
                final_h = hidden_states[-1][valid_mask]
                final_logits = lm_head(final_h).float()
                logp_L = F.log_softmax(final_logits, dim=-1)
                p_L = logp_L.exp()

            optimizer.zero_grad()
            batch_loss = 0.0

            # Compute loss layer by layer, summing gradients (saves memory!)
            for layer_idx in range(num_layers):
                h_l = hidden_states[layer_idx][valid_mask]
                h_tilde = translators(h_l, layer_idx)
                logits_l = lm_head(ln_f(h_tilde)).float()
                logp_l = F.log_softmax(logits_l, dim=-1)

                kl_l = (p_L * (logp_L - logp_l)).sum(dim=-1).mean()
                kl_l.backward()
                batch_loss += kl_l.item()

            optimizer.step()

            epoch_loss += batch_loss
            batches_in_epoch += 1
            global_step += 1

            if empty_cache_interval > 0 and global_step % empty_cache_interval == 0:
                if is_mps and hasattr(torch, "mps"):
                    torch.mps.empty_cache()
                elif is_cuda and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if global_step % log_interval == 0 or global_step == 1:
                avg_kl = batch_loss / num_layers
                print(
                    f"Epoch {epoch:2d}/{epochs} | Step {global_step:4d} | "
                    f"Batch Total KL: {batch_loss:>7.4f} | Avg KL/layer: {avg_kl:>6.4f}"
                )

        avg_epoch_loss = epoch_loss / max(batches_in_epoch, 1)
        history["train_loss"].append(avg_epoch_loss)

        # Validate on dev split
        dev_kl = evaluate_dev_kl(model, dev_sentences, tokenizer, translators=translators, batch_size=16)
        mean_dev_kl = sum(dev_kl) / len(dev_kl)
        history["dev_kl_per_epoch"].append(dev_kl)

        print(
            f"--> Epoch {epoch:2d} Completed: Avg Train Loss = {avg_epoch_loss:.4f} | "
            f"Mean Dev KL = {mean_dev_kl:.4f}"
        )

        if mean_dev_kl < best_dev_kl:
            best_dev_kl = mean_dev_kl

        if max_steps is not None and global_step >= max_steps:
            print(f"Reached max_steps ({max_steps}). Stopping early.")
            break

    translators.eval()
    print("Training finished successfully!")
    return translators, history


def save_tuned_lens_checkpoint(
    translators: TunedLensTranslators,
    model_name: str,
    output_dir: str | Path = "checkpoints/tuned_lens",
    *,
    train_data: str = "isdt_train",
    epochs: int = 3,
    seed: int = 42,
    **extra_meta: Any,
) -> tuple[Path, Path]:
    """Save trained translators state_dict and contract metadata JSON.

    Adheres strictly to CONTRACTS.md Section 9.5:
    - `checkpoints/tuned_lens/{model}.pt`: state_dict of TunedLensTranslators.
    - `checkpoints/tuned_lens/{model}.json`: {"model": key, "train_data": ..., "epochs": ..., "seed": 42}.

    Parameters
    ----------
    translators : TunedLensTranslators
        Trained translators module.
    model_name : str
        Model registry key (e.g. "geppetto").
    output_dir : str or Path, default="checkpoints/tuned_lens"
        Directory where checkpoint and metadata are saved.
    train_data : str, default="isdt_train"
        Training dataset identifier.
    epochs : int, default=3
        Number of epochs trained.
    seed : int, default=42
        Random seed used for training.
    extra_meta : Any
        Additional metadata key-values.

    Returns
    -------
    tuple[Path, Path]
        Paths to the saved (.pt, .json) files.
    """
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        from lib.paths import ROOT
        out_path = ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    pt_path = out_path / f"{model_name}.pt"
    json_path = out_path / f"{model_name}.json"

    # Save state dict
    torch.save(translators.state_dict(), pt_path)

    # Save metadata
    meta = {
        "model": model_name,
        "train_data": train_data,
        "epochs": epochs,
        "seed": seed,
        "num_layers": translators.num_layers,
        "hidden_size": translators.hidden_size,
        **extra_meta,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return pt_path, json_path


def load_tuned_lens_checkpoint(
    model_name: str,
    num_layers: int = 12,
    hidden_size: int = 768,
    checkpoint_dir: str | Path = "checkpoints/tuned_lens",
    device: torch.device | str | None = None,
) -> tuple[TunedLensTranslators, dict[str, Any]]:
    """Load trained TunedLensTranslators and metadata from checkpoint directory.

    Parameters
    ----------
    model_name : str
        Model registry key (e.g. "geppetto").
    num_layers : int, default=12
        Number of transformer layers.
    hidden_size : int, default=768
        Hidden dimension size.
    checkpoint_dir : str or Path, default="checkpoints/tuned_lens"
        Directory containing the checkpoint files.
    device : torch.device or str or None, optional
        Target device for translators.

    Returns
    -------
    tuple[TunedLensTranslators, dict[str, Any]]
        Loaded translators module and metadata dict.

    Raises
    ------
    FileNotFoundError
        If checkpoint or metadata file does not exist.
    """
    p_dir = Path(checkpoint_dir)
    if not p_dir.is_absolute() and not (p_dir / f"{model_name}.pt").exists():
        from lib.paths import ROOT
        if (ROOT / p_dir / f"{model_name}.pt").exists():
            p_dir = ROOT / p_dir

    pt_path = p_dir / f"{model_name}.pt"
    json_path = p_dir / f"{model_name}.json"

    if not pt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {pt_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata not found at: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    n_layers = meta.get("num_layers", num_layers)
    h_size = meta.get("hidden_size", hidden_size)

    translators = TunedLensTranslators(num_layers=n_layers, hidden_size=h_size)
    map_loc = device or "cpu"
    translators.load_state_dict(torch.load(pt_path, map_location=map_loc))

    if device is not None:
        translators.to(device)

    translators.eval()
    return translators, meta
