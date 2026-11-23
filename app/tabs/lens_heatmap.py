"""Tab 1: live lens heatmap (issue A1)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

from lib.alignment import encode_words_detailed
from lib.data import load_model
from lib.registry import get_model_spec
from ws4_lens.jacobian_lens import jacobian_lens, load_jacobian_checkpoint
from ws4_lens.logit_lens import logit_lens, run_model
from ws4_lens.metrics import entropy, kl_to_final
from ws4_lens.tuned_lens import load_official_tuned_lens, load_tuned_lens_checkpoint, tuned_lens

MODELS: tuple[str, ...] = ("gpt2", "geppetto")
LENSES: tuple[str, ...] = ("logit", "tuned", "jacobian")
METRICS: tuple[str, ...] = ("entropy", "kl")
MAX_TOKENS = 40

DEFAULT_SENTENCES: dict[str, str] = {
    "gpt2": "The cat sat on the mat.",
    "geppetto": "Il gatto si è seduto sul tappeto.",
}

_EMPTY_TOKEN = "∅"


def _display_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    text = text.replace("\n", "↵").strip()
    if not text:
        return _EMPTY_TOKEN
    return text


def _x_labels(tokenizer, input_ids: torch.Tensor, valid_len: int) -> list[str]:
    labels: list[str] = []
    counts: dict[str, int] = {}
    for pos in range(valid_len):
        tid = int(input_ids[0, pos].item())
        if pos == 0:
            base = "<bos>"
        else:
            base = _display_token(tokenizer, tid)
        counts[base] = counts.get(base, 0) + 1
        if counts[base] > 1:
            labels.append(f"{pos}:{base}")
        else:
            labels.append(base)
    return labels


def _top1_cell_text(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    text = text.lstrip()
    if not text:
        return _EMPTY_TOKEN
    return text.replace("\n", "↵")


@st.cache_resource(show_spinner="Loading model…")
def load_lens_model(key: str):
    """Load and cache a decoder model and tokenizer."""
    spec = get_model_spec(key)
    return load_model(
        spec["hf_id"],
        kind=spec["kind"],
        revision=spec.get("revision", "main"),
        tokenizer_kwargs={"add_prefix_space": True},
    )


@st.cache_resource(show_spinner="Loading lens parameters…")
def load_lens_params(key: str, lens: str):
    """
  Load lens-specific parameters.

    Returns
    -------
    tuple[object | None, str | None]
        (params, error_message); params is None when the lens cannot be loaded.
    """
    if lens == "logit":
        return None, None

    model, _tokenizer, _meta = load_lens_model(key)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    try:
        if lens == "tuned":
            if key == "gpt2":
                params = load_official_tuned_lens(model)
            else:
                params, _meta = load_tuned_lens_checkpoint(key, device=device)
            return params, None
        if lens == "jacobian":
            jacobians = load_jacobian_checkpoint(key)
            return jacobians, None
    except FileNotFoundError as err:
        return None, f"Checkpoint missing for {lens} lens on {key}: {err}"
    except ImportError as err:
        return None, f"Cannot load {lens} lens: {err}"
    except OSError as err:
        return None, f"Cannot load {lens} lens (download or I/O error): {err}"

    return None, f"Unknown lens: {lens}"


def compute_lens_grid(
    model: torch.nn.Module,
    tokenizer,
    words: list[str],
    lens: str,
    params: object | None,
    *,
    max_tokens: int = MAX_TOKENS,
) -> tuple[list[str], list[list[str]], np.ndarray, np.ndarray, bool]:
    """
    Run the selected lens on one sentence and build heatmap data.

    Returns
    -------
    x_labels, top1_text, entropy_grid, kl_grid, truncated
        Shapes of metric grids are [L + 1, T] for valid sequence length T.
    """
    if not words:
        raise ValueError("words must not be empty.")

    batch, _word_ids, truncated_flags = encode_words_detailed(
        tokenizer,
        [words],
        add_bos=True,
        max_length=max_tokens + 1,
    )
    truncated = bool(truncated_flags[0])

    input_ids = batch["input_ids"].to(next(model.parameters()).device)
    attention_mask = batch["attention_mask"].to(input_ids.device)
    valid_len = int(attention_mask[0].sum().item())

    hidden_states = run_model(model, input_ids, attention_mask)

    if lens == "logit":
        logits_list = logit_lens(model, hidden_states)
    elif lens == "tuned":
        if params is None:
            raise ValueError("tuned lens requires parameters.")
        logits_list = tuned_lens(model, params, hidden_states)
    elif lens == "jacobian":
        if params is None:
            raise ValueError("jacobian lens requires parameters.")
        logits_list = jacobian_lens(model, params, hidden_states)
    else:
        raise ValueError(f"Unknown lens: {lens}")

    final_logits = logits_list[-1]
    num_layers = len(logits_list)
    seq_len = valid_len

    ent = np.zeros((num_layers, seq_len), dtype=np.float32)
    kl = np.zeros((num_layers, seq_len), dtype=np.float32)
    top1_text: list[list[str]] = []

    for layer_idx, logits_l in enumerate(logits_list):
        ent_l = entropy(logits_l[0, :seq_len, :]).detach().cpu().numpy()
        kl_l = kl_to_final(logits_l[0, :seq_len, :], final_logits[0, :seq_len, :]).detach().cpu().numpy()
        ent[layer_idx, :] = ent_l
        kl[layer_idx, :] = kl_l
        preds = logits_l[0, :seq_len, :].argmax(dim=-1).tolist()
        top1_text.append([_top1_cell_text(tokenizer, pid) for pid in preds])

    x_labels = _x_labels(tokenizer, input_ids.cpu(), valid_len)
    return x_labels, top1_text, ent, kl, truncated


def _build_heatmap(
    x_labels: list[str],
    top1_text: list[list[str]],
    z_grid: np.ndarray,
    metric_name: str,
    kl_grid: np.ndarray,
    ent_grid: np.ndarray,
) -> go.Figure:
    num_layers = z_grid.shape[0]
    y_labels = [str(layer) for layer in range(num_layers)]

    hover = []
    for layer_idx in range(num_layers):
        row = []
        for tok_idx, x_lab in enumerate(x_labels):
            cell_top1 = top1_text[layer_idx][tok_idx]
            row.append(
                f"Layer {layer_idx}<br>Token: {x_lab}<br>"
                f"Top-1: {cell_top1}<br>"
                f"Entropy: {ent_grid[layer_idx, tok_idx]:.3f} nats<br>"
                f"KL: {kl_grid[layer_idx, tok_idx]:.3f} nats"
            )
        hover.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_grid,
            x=x_labels,
            y=y_labels,
            text=top1_text,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertext=hover,
            hoverinfo="text",
            colorbar={"title": f"{metric_name} (nats)"},
        )
    )
    fig.update_layout(
        title="Lens heatmap (cell text = top-1 prediction at this layer)",
        xaxis_title="Input token (position t predicts token at t+1)",
        yaxis_title="Layer",
        yaxis={"autorange": "reversed"},
        height=max(400, 28 * num_layers),
    )
    return fig


def render() -> None:
    """Tab 1 UI: model, text, lens, metric, and plotly heatmap."""
    col_model, col_lens, col_metric = st.columns(3)
    with col_model:
        model_key = st.selectbox("Model", MODELS, format_func=lambda k: k)
    with col_lens:
        lens = st.radio("Lens", LENSES, horizontal=True)
    with col_metric:
        metric = st.radio("Colour", METRICS, format_func=lambda m: m.upper(), horizontal=True)

    default_text = DEFAULT_SENTENCES.get(model_key, "")
    text = st.text_area("Sentence (whitespace-separated words)", value=default_text, height=80)

    if not text.strip():
        st.info("Enter a sentence to run the lens.")
        return

    words = text.strip().split()
    if len(words) > MAX_TOKENS:
        st.warning(f"Input has {len(words)} words; only the first {MAX_TOKENS} are encoded.")

    params, lens_error = load_lens_params(model_key, lens)
    if lens_error:
        st.warning(lens_error)
        return

    model, tokenizer, _meta = load_lens_model(model_key)

    with st.spinner("Running lens…"):
        try:
            x_labels, top1_text, ent_grid, kl_grid, truncated = compute_lens_grid(
                model, tokenizer, words[:MAX_TOKENS], lens, params
            )
        except ValueError as err:
            st.error(str(err))
            return

    if truncated:
        st.warning(f"Sequence truncated to {MAX_TOKENS} subword tokens (plus BOS).")

    z_grid = ent_grid if metric == "entropy" else kl_grid
    fig = _build_heatmap(x_labels, top1_text, z_grid, metric, kl_grid, ent_grid)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "At position t the lens predicts the token at t+1. "
        "KL is D_KL(P_L || P_l) in natural log units (nats)."
    )