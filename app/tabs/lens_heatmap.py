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

# Light sequential scales: dark cell labels (#1a1a1a) stay readable on every stop.
_COLORSCALE_ENTROPY = [
    [0.0, "#ffffff"],
    [0.15, "#f0f6fc"],
    [0.35, "#d4e6f5"],
    [0.55, "#a8cce8"],
    [0.75, "#7eb8dd"],
    [1.0, "#5aa9d6"],
]
_COLORSCALE_KL = [
    [0.0, "#ffffff"],
    [0.2, "#f7fcf5"],
    [0.45, "#e3f2da"],
    [0.65, "#fff3cd"],
    [0.85, "#fde0b3"],
    [1.0, "#f4b183"],
]
_CELL_TEXT_COLOR = "#1a1a1a"


def _display_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    text = text.replace("\n", "↵").strip()
    if not text:
        return _EMPTY_TOKEN
    return text


def _column_labels(
    tokenizer,
    input_ids: torch.Tensor,
    valid_len: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Build axis labels for each position t.

    Returns input token, gold next token, and a compact multi-line x-tick label.
    """
    input_toks: list[str] = []
    next_toks: list[str] = []
    x_ticks: list[str] = []
    for pos in range(valid_len):
        tid = int(input_ids[0, pos].item())
        input_tok = "<bos>" if pos == 0 else _display_token(tokenizer, tid)
        input_toks.append(input_tok)
        if pos + 1 < valid_len:
            next_tid = int(input_ids[0, pos + 1].item())
            next_tok = _display_token(tokenizer, next_tid)
        else:
            next_tok = "—"
        next_toks.append(next_tok)
        x_ticks.append(f"<b>{input_tok}</b> → {next_tok}")
    return input_toks, next_toks, x_ticks


def _layer_label(layer_idx: int, final_layer: int) -> str:
    if layer_idx == 0:
        return "0 · input"
    if layer_idx == final_layer:
        return f"L={final_layer} · final"
    return str(layer_idx)


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
    """Load lens-specific parameters.

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
) -> tuple[list[str], list[str], list[str], list[list[str]], np.ndarray, np.ndarray, bool]:
    """
    Run the selected lens on one sentence and build heatmap data.

    Returns
    -------
    x_ticks, input_toks, next_toks, top1_text, entropy_grid, kl_grid, truncated
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

    input_ids_cpu = input_ids.cpu()
    input_toks, next_toks, x_ticks = _column_labels(tokenizer, input_ids_cpu, valid_len)
    return x_ticks, input_toks, next_toks, top1_text, ent, kl, truncated


def _cell_display(top1: str, gold: str) -> str:
    if gold == "—":
        return top1
    if top1 == gold:
        return f"✓ {top1}"
    return top1


def _build_heatmap(
    x_ticks: list[str],
    next_toks: list[str],
    top1_text: list[list[str]],
    z_grid: np.ndarray,
    metric_name: str,
    kl_grid: np.ndarray,
    ent_grid: np.ndarray,
    *,
    lens: str,
) -> go.Figure:
    """Build heatmap with one row per layer (0 … L); Y is numeric to avoid lexicographic sort."""
    num_layers = z_grid.shape[0]
    final_layer = num_layers - 1
    y_indices = list(range(num_layers))
    y_ticktext = [_layer_label(layer, final_layer) for layer in y_indices]

    display_text: list[list[str]] = [
        [_cell_display(top1_text[layer_idx][tok_idx], next_toks[tok_idx]) for tok_idx in range(len(x_ticks))]
        for layer_idx in range(num_layers)
    ]

    hover = []
    for layer_idx in range(num_layers):
        row = []
        for tok_idx, x_lab in enumerate(x_ticks):
            cell_top1 = top1_text[layer_idx][tok_idx]
            gold = next_toks[tok_idx]
            match = cell_top1 == gold and gold != "—"
            row.append(
                f"Layer {_layer_label(layer_idx, final_layer)} · t={tok_idx}<br>"
                f"Top-1: <b>{cell_top1}</b> · Gold: <b>{gold}</b>"
                f"{' ✓' if match else ''}<br>"
                f"Entropy: {ent_grid[layer_idx, tok_idx]:.2f} nats<br>"
                f"KL to final: {kl_grid[layer_idx, tok_idx]:.2f} nats"
            )
        hover.append(row)

    if metric_name == "entropy":
        colorscale = _COLORSCALE_ENTROPY
        colorbar_title = "Entropy (↑ = more uncertain)"
    else:
        colorscale = _COLORSCALE_KL
        colorbar_title = "KL to final (↑ = farther from model)"

    fig = go.Figure(
        data=go.Heatmap(
            z=z_grid,
            x=x_ticks,
            y=y_indices,
            text=display_text,
            texttemplate="%{text}",
            textfont={"size": 10, "color": _CELL_TEXT_COLOR},
            hovertext=hover,
            hoverinfo="text",
            colorscale=colorscale,
            zmin=0.0,
            zauto=True,
            colorbar={
                "title": {"text": colorbar_title, "side": "right"},
                "thickness": 18,
            },
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title={"text": f"{lens} lens · ✓ = top-1 = gold", "font": {"size": 16}},
        xaxis={
            "title": "Input → next token (gold)",
            "tickangle": -25,
            "side": "bottom",
        },
        yaxis={
            "title": "Layer",
            "type": "linear",
            "tickmode": "array",
            "tickvals": y_indices,
            "ticktext": y_ticktext,
            "dtick": 1,
            "range": [-0.5, final_layer + 0.5],
            "autorange": False,
            "fixedrange": False,
        },
        xaxis_autorange=True,
        margin={"l": 90, "r": 40, "b": 120, "t": 80},
        height=max(460, 36 * num_layers),
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
        metric = st.radio("Color", METRICS, format_func=lambda m: m.upper(), horizontal=True)

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
            x_ticks, _input_toks, next_toks, top1_text, ent_grid, kl_grid, truncated = compute_lens_grid(
                model, tokenizer, words[:MAX_TOKENS], lens, params
            )
        except ValueError as err:
            st.error(str(err))
            return

    spec = get_model_spec(model_key)
    expected_rows = int(spec["num_layers"]) + 1
    if ent_grid.shape[0] != expected_rows:
        st.warning(
            f"Expected {expected_rows} layers (0…L) but the lens returned {ent_grid.shape[0]} rows. "
            "Check model and checkpoint compatibility."
        )

    if truncated:
        st.warning(f"Sequence truncated to {MAX_TOKENS} subword tokens (plus BOS).")

    z_grid = ent_grid if metric == "entropy" else kl_grid
    fig = _build_heatmap(
        x_ticks, next_toks, top1_text, z_grid, metric, kl_grid, ent_grid, lens=lens
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )

    final_preds = top1_text[-1]
    n_scored = sum(1 for g in next_toks if g != "—")
    n_correct = sum(
        1 for pred, gold in zip(final_preds, next_toks, strict=True) if gold != "—" and pred == gold
    )
    if n_scored:
        st.caption(f"Final layer: **{n_correct}/{n_scored}** correct top-1 predictions.")