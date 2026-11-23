"""Tests for app Tab 1 lens heatmap (issue A1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("streamlit")
pytest.importorskip("plotly")

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from tabs import cka, crossling, probing  # noqa: E402, I001
from tabs.lens_heatmap import _build_heatmap, compute_lens_grid  # noqa: E402, I001


@pytest.fixture
def tiny_gpt2(gpt2_tokenizer):
    from transformers import GPT2Config, GPT2LMHeadModel

    vocab_size = len(gpt2_tokenizer)
    config = GPT2Config(
        n_layer=2,
        n_embd=16,
        n_head=2,
        n_positions=64,
        vocab_size=vocab_size,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    model.config.output_hidden_states = True
    model.config.use_cache = False
    return model, gpt2_tokenizer


def test_compute_lens_grid_shapes(tiny_gpt2):
    model, tokenizer = tiny_gpt2
    words = ["The", "cat", "sat", "on", "the", "mat", "."]
    x_ticks, _input_toks, next_toks, top1_text, ent, kl, truncated = compute_lens_grid(
        model, tokenizer, words, "logit", None, max_tokens=20
    )
    assert len(x_ticks) == len(next_toks) == ent.shape[1] == kl.shape[1]
    assert ent.shape[0] == kl.shape[0] == model.config.n_layer + 1
    assert len(top1_text) == ent.shape[0]
    assert len(top1_text[0]) == ent.shape[1]
    assert (ent >= 0).all()
    assert np.allclose(kl[-1], 0.0, atol=1e-4)
    assert truncated is False


def test_compute_lens_grid_truncation(tiny_gpt2):
    model, tokenizer = tiny_gpt2
    words = ["word"] * 50
    max_tokens = 8
    x_ticks, _input_toks, _next_toks, _top1, ent, kl, truncated = compute_lens_grid(
        model, tokenizer, words, "logit", None, max_tokens=max_tokens
    )
    assert truncated is True
    assert ent.shape[1] <= max_tokens + 1
    assert kl.shape == ent.shape
    assert len(x_ticks) == ent.shape[1]


def test_build_heatmap_maps_each_layer_row():
    """Y must be 0..L in order (Plotly lexicographic category sort breaks plain '10' vs '2')."""
    num_layers = 13
    num_tokens = 4
    z = np.linspace(0, 1, num_layers * num_tokens, dtype=np.float32).reshape(num_layers, num_tokens)
    top1 = [["tok"] * num_tokens for _ in range(num_layers)]
    x_ticks = [f"t={i}" for i in range(num_tokens)]
    next_toks = ["g"] * num_tokens
    fig = _build_heatmap(
        x_ticks, next_toks, top1, z, "entropy", z, z, lens="logit",
    )
    trace = fig.data[0]
    assert trace.z.shape == (num_layers, num_tokens)
    assert list(trace.y) == list(range(num_layers))
    yaxis = fig.layout.yaxis
    assert list(yaxis.tickvals) == list(range(num_layers))
    assert len(yaxis.ticktext) == num_layers
    assert yaxis.ticktext[0] == "0 · input"
    assert yaxis.ticktext[-1] == "L=12 · final"


def test_placeholder_tabs_expose_render():
    assert callable(probing.render)
    assert callable(cka.render)
    assert callable(crossling.render)