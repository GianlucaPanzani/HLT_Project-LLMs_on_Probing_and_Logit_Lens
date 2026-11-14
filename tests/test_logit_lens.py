"""Unit and integration tests for ws4_lens.logit_lens (Issue L1)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from ws4_lens.logit_lens import logit_lens, run_model  # noqa: E402, I001


# --------------------------------------------------------------------------- #
# Synthetic / Fast Fixtures & Mocks (No Network / CI Friendly)
# --------------------------------------------------------------------------- #

class DummyTransformer(nn.Module):
    """Minimal transformer container with final layer norm."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.ln_f = nn.LayerNorm(hidden_size)


class DummyDecoderModel(nn.Module):
    """Minimal causal language model with transformer.ln_f and lm_head."""

    def __init__(self, hidden_size: int = 16, vocab_size: int = 32, num_layers: int = 4):
        super().__init__()
        self.transformer = DummyTransformer(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ):
        batch_size, seq_len = input_ids.shape
        # Create L + 1 synthetic hidden state tensors
        hidden_states = tuple(
            torch.randn(batch_size, seq_len, self.hidden_size)
            for _ in range(self.num_layers + 1)
        )
        logits = self.lm_head(hidden_states[-1])
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


# --------------------------------------------------------------------------- #
# Fast Unit Tests (Synthetic Data)
# --------------------------------------------------------------------------- #

def test_run_model_synthetic():
    """Verify run_model extracts L + 1 hidden states with expected shapes."""
    model = DummyDecoderModel(hidden_size=16, vocab_size=32, num_layers=4)
    input_ids = torch.randint(0, 32, (2, 6))

    hidden_states = run_model(model, input_ids)

    assert isinstance(hidden_states, list)
    assert len(hidden_states) == 5  # L + 1 for num_layers = 4
    for h in hidden_states:
        assert h.shape == (2, 6, 16)


def test_run_model_missing_hidden_states_raises():
    """Verify ValueError is raised if model outputs lack hidden_states."""
    class BrokenModel(nn.Module):
        def forward(self, *args, **kwargs):
            return SimpleNamespace(logits=torch.randn(1, 2, 4))

    model = BrokenModel()
    with pytest.raises(ValueError, match="do not contain 'hidden_states'"):
        run_model(model, torch.zeros((1, 2), dtype=torch.long))


def test_logit_lens_synthetic_projection_and_sanity_check():
    """Verify logit_lens applies ln_f for l < L, identity for l = L, and matches logits."""
    hidden_size = 16
    vocab_size = 32
    num_layers = 3
    batch_size = 2
    seq_len = 5

    model = DummyDecoderModel(hidden_size, vocab_size, num_layers)
    hidden_states = [
        torch.randn(batch_size, seq_len, hidden_size)
        for _ in range(num_layers + 1)
    ]

    lens_logits = logit_lens(model, hidden_states)

    assert isinstance(lens_logits, list)
    assert len(lens_logits) == num_layers + 1

    # Check intermediate layers: logits_l == lm_head(ln_f(h_l))
    for layer_idx in range(num_layers):
        expected_l = model.lm_head(model.transformer.ln_f(hidden_states[layer_idx]))
        assert torch.allclose(lens_logits[layer_idx], expected_l, atol=1e-6)
        assert lens_logits[layer_idx].shape == (batch_size, seq_len, vocab_size)

    # Check layer L sanity check: logits_L == lm_head(h_L)
    expected_L = model.lm_head(hidden_states[num_layers])
    assert torch.allclose(lens_logits[num_layers], expected_L, atol=1e-6)


def test_logit_lens_empty_hidden_states_raises():
    """Verify ValueError is raised if hidden_states is empty."""
    model = DummyDecoderModel()
    with pytest.raises(ValueError, match="cannot be empty"):
        logit_lens(model, [])


def test_logit_lens_missing_components_raises():
    """Verify AttributeError is raised when ln_f or lm_head cannot be resolved."""
    class EmptyModel(nn.Module):
        pass

    model = EmptyModel()
    with pytest.raises(AttributeError, match="Could not locate final layer normalization"):
        logit_lens(model, [torch.randn(1, 2, 4)])


# --------------------------------------------------------------------------- #
# Slow Integration Tests (Real Models: GPT-2 and GePpeTto)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_logit_lens_gpt2_integration():
    """Verify logit_lens on official pretrained GPT-2 satisfies CONTRACTS.md sanity check."""
    pytest.importorskip("transformers")
    from lib.data import load_model
    from lib.registry import get_model_spec

    spec = get_model_spec("gpt2")
    model, tokenizer, metadata = load_model(
        checkpoint=spec["hf_id"],
        kind=spec["kind"],
        revision=spec.get("revision", "main"),
        dtype=spec.get("dtype"),
        tokenizer_kwargs={"add_prefix_space": True},
    )

    prompt = "The capital of France is"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    hidden_states = run_model(model, input_ids, attention_mask)
    assert len(hidden_states) == metadata["num_layers"] + 1

    lens_logits = logit_lens(model, hidden_states)
    assert len(lens_logits) == metadata["num_layers"] + 1

    # Sanity check: logits_L == model.forward().logits
    with torch.no_grad():
        official_logits = model(
            input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
        ).logits

    logits_L = lens_logits[-1]
    assert torch.allclose(logits_L, official_logits, atol=1e-5)


@pytest.mark.slow
def test_logit_lens_geppetto_integration():
    """Verify logit_lens on official pretrained GePpeTto satisfies CONTRACTS.md sanity check."""
    pytest.importorskip("transformers")
    from lib.data import load_model
    from lib.registry import get_model_spec

    spec = get_model_spec("geppetto")
    model, tokenizer, metadata = load_model(
        checkpoint=spec["hf_id"],
        kind=spec["kind"],
        revision=spec.get("revision", "main"),
        dtype=spec.get("dtype"),
    )

    prompt = "La capitale d'Italia è"
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    hidden_states = run_model(model, input_ids, attention_mask)
    assert len(hidden_states) == metadata["num_layers"] + 1

    lens_logits = logit_lens(model, hidden_states)
    assert len(lens_logits) == metadata["num_layers"] + 1

    with torch.no_grad():
        official_logits = model(
            input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
        ).logits

    logits_L = lens_logits[-1]
    assert torch.allclose(logits_L, official_logits, atol=1e-5)
