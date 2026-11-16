"""Unit and integration tests for ws4_lens.tuned_lens (Issues L3 and L4)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ws4_lens.logit_lens import logit_lens  # noqa: E402
from ws4_lens.tuned_lens import (  # noqa: E402
    TunedLensTranslators,
    load_official_tuned_lens,
    load_tuned_lens_checkpoint,
    save_tuned_lens_checkpoint,
    tuned_lens,
)

# --------------------------------------------------------------------------- #
# Synthetic Model Mocks for Fast Unit Tests
# --------------------------------------------------------------------------- #

class MockTransformer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.ln_f = nn.LayerNorm(hidden_size)


class MockDecoderModel(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 4) -> None:
        super().__init__()
        self.transformer = MockTransformer(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.config = SimpleNamespace(n_layer=num_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = True,
    ) -> SimpleNamespace:
        batch_size, seq_len = input_ids.shape
        hidden_states = tuple(
            torch.randn(batch_size, seq_len, self.hidden_size)
            for _ in range(self.num_layers + 1)
        )
        logits = self.lm_head(hidden_states[-1])
        return SimpleNamespace(logits=logits, hidden_states=hidden_states)


class MockOfficialLens(nn.Module):
    """Mock simulating the official TunedLens interface from `tuned-lens`."""

    def __init__(self, num_layers: int, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.layer_translators = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, h: torch.Tensor, idx: int) -> torch.Tensor:
        if idx >= len(self.layer_translators):
            raise IndexError(f"index {idx} is out of range")
        h_trans = h + self.layer_translators[idx](h)
        return self.lm_head(self.ln_f(h_trans))

    def __len__(self) -> int:
        return len(self.layer_translators)


# --------------------------------------------------------------------------- #
# Fast Unit Tests
# --------------------------------------------------------------------------- #

def test_translators_initialization():
    """Verify layer translators initialization shape, zero weights, and zero biases."""
    num_layers = 12
    hidden_size = 64
    translators = TunedLensTranslators(num_layers=num_layers, hidden_size=hidden_size)

    assert len(translators) == num_layers
    assert translators.num_layers == num_layers
    assert translators.hidden_size == hidden_size

    for i in range(num_layers):
        lin = translators[i]
        assert isinstance(lin, nn.Linear)
        assert lin.weight.shape == (hidden_size, hidden_size)
        assert lin.bias.shape == (hidden_size,)
        assert torch.all(lin.weight == 0.0)
        assert torch.all(lin.bias == 0.0)


def test_translators_forward_zero_init():
    """Zero-initialized translators must act as the identity mapping."""
    translators = TunedLensTranslators(num_layers=4, hidden_size=32)
    h = torch.randn(2, 5, 32)

    # For layer l < num_layers: h + 0 = h (supports both (layer_idx, h) and (h, layer_idx))
    for layer_idx in range(4):
        h_out_1 = translators(layer_idx, h)
        h_out_2 = translators(h, layer_idx)
        assert torch.equal(h_out_1, h)
        assert torch.equal(h_out_2, h)

    # For layer l >= num_layers: returned unchanged
    h_out_final = translators(4, h)
    assert torch.equal(h_out_final, h)


def test_translators_forward_with_weights():
    """Verify affine residual calculation h + A_l h + b_l when weights are non-zero."""
    translators = TunedLensTranslators(num_layers=2, hidden_size=4)
    # Set explicit weights for layer 0: W = 2 * I, bias = [1, 1, 1, 1]
    with torch.no_grad():
        translators[0].weight.copy_(torch.eye(4) * 2.0)
        translators[0].bias.copy_(torch.ones(4))

    h = torch.ones(1, 1, 4)
    # h_tilde = h + (h @ W.T + b) = [1, 1, 1, 1] + ([2, 2, 2, 2] + [1, 1, 1, 1]) = [4, 4, 4, 4]
    h_out = translators(0, h)
    expected = torch.full((1, 1, 4), 4.0)
    assert torch.allclose(h_out, expected)


def test_tuned_lens_zero_init_equals_logit_lens():
    """With zero-initialized translators, tuned_lens must strictly equal logit_lens."""
    hidden_size = 16
    vocab_size = 32
    num_layers = 4
    model = MockDecoderModel(hidden_size, vocab_size, num_layers)
    translators = TunedLensTranslators(num_layers, hidden_size)

    # L + 1 hidden states
    hidden_states = [torch.randn(2, 6, hidden_size) for _ in range(num_layers + 1)]

    logits_standard = logit_lens(model, hidden_states)
    logits_tuned = tuned_lens(model, translators, hidden_states)

    assert len(logits_tuned) == num_layers + 1
    for layer_idx in range(num_layers + 1):
        assert torch.allclose(logits_tuned[layer_idx], logits_standard[layer_idx], atol=1e-6)


def test_tuned_lens_layer_L_equals_model_unembedding():
    """At layer L, tuned_lens applies lm_head(h_L) directly regardless of translator weights."""
    hidden_size = 16
    vocab_size = 32
    num_layers = 3
    model = MockDecoderModel(hidden_size, vocab_size, num_layers)
    translators = TunedLensTranslators(num_layers, hidden_size)

    # Fill translators with arbitrary non-zero weights
    for lin in translators.translators:
        nn.init.normal_(lin.weight)
        nn.init.normal_(lin.bias)

    hidden_states = [torch.randn(1, 4, hidden_size) for _ in range(num_layers + 1)]
    logits = tuned_lens(model, translators, hidden_states)

    # Layer L must match model.lm_head(hidden_states[-1])
    expected_L = model.lm_head(hidden_states[-1])
    assert torch.allclose(logits[-1], expected_L)


def test_tuned_lens_with_mock_official_lens():
    """Verify polymorphic support for official TunedLens instances."""
    hidden_size = 16
    vocab_size = 32
    num_layers = 4
    model = MockDecoderModel(hidden_size, vocab_size, num_layers)
    official_lens = MockOfficialLens(num_layers, hidden_size, vocab_size)

    hidden_states = [torch.randn(2, 5, hidden_size) for _ in range(num_layers + 1)]
    logits = tuned_lens(model, official_lens, hidden_states)

    assert len(logits) == num_layers + 1
    # Check intermediate layer was computed via official_lens
    expected_0 = official_lens(hidden_states[0], 0)
    assert torch.allclose(logits[0], expected_0)
    # Check layer L was computed via model.lm_head
    expected_L = model.lm_head(hidden_states[-1])
    assert torch.allclose(logits[-1], expected_L)


def test_tuned_lens_empty_hidden_states_raises():
    """An empty hidden_states list must raise ValueError."""
    model = MockDecoderModel(16, 32)
    translators = TunedLensTranslators(4, 16)
    with pytest.raises(ValueError, match="hidden_states list must not be empty"):
        tuned_lens(model, translators, [])


def test_tuned_lens_invalid_translator_raises():
    """Passing an unsupported object type as translators must raise TypeError."""
    model = MockDecoderModel(16, 32)
    hidden_states = [torch.randn(1, 2, 16) for _ in range(3)]
    with pytest.raises(TypeError, match="Unsupported translators type"):
        tuned_lens(model, "invalid_string_translator", hidden_states)


def test_save_and_load_tuned_lens_checkpoint(tmp_path: Path):
    """Verify saving state_dict and CONTRACTS.md Section 9.5 metadata and reloading."""
    num_layers = 3
    hidden_size = 8
    translators = TunedLensTranslators(num_layers, hidden_size)

    # Modify one weight to non-zero
    with torch.no_grad():
        translators[0].weight.fill_(0.5)

    pt_path, json_path = save_tuned_lens_checkpoint(
        translators=translators,
        model_name="mock_model",
        output_dir=tmp_path,
        train_data="test_treebank",
        epochs=3,
        seed=42,
    )

    assert pt_path.exists()
    assert json_path.exists()

    loaded_translators, meta = load_tuned_lens_checkpoint(
        model_name="mock_model",
        checkpoint_dir=tmp_path,
    )

    assert meta["model"] == "mock_model"
    assert meta["train_data"] == "test_treebank"
    assert meta["epochs"] == 3
    assert meta["seed"] == 42
    assert torch.allclose(loaded_translators[0].weight, translators[0].weight)


def test_tuned_lens_tiny_model_training_step():
    """Verify acceptance criterion: zero-init equals logit lens and 1 step lowers loss.

    Requirement from Issue [L4]:
    - Zero-initialised translators give exactly the logit lens.
    - One optimiser step lowers the loss.
    """
    torch.manual_seed(42)
    hidden_size = 16
    vocab_size = 32
    num_layers = 3
    batch_size = 2
    seq_len = 4

    model = MockDecoderModel(hidden_size, vocab_size, num_layers)
    for p in model.parameters():
        p.requires_grad = False

    translators = TunedLensTranslators(num_layers, hidden_size)
    optimizer = torch.optim.AdamW(translators.parameters(), lr=1e-3, weight_decay=1e-3)

    # Create synthetic hidden states
    hidden_states = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(num_layers + 1)]
    final_h = hidden_states[-1]
    final_logits = model.lm_head(final_h)
    p_L = F.log_softmax(final_logits, dim=-1).exp()
    logp_L = F.log_softmax(final_logits, dim=-1)

    # 1. Verify zero-initialized loss equals logit lens loss
    logits_logit_lens = logit_lens(model, hidden_states)
    logits_tuned_0 = tuned_lens(model, translators, hidden_states)

    loss_baseline = 0.0
    loss_init = 0.0
    for layer_idx in range(num_layers):
        logp_logit_l = F.log_softmax(logits_logit_lens[layer_idx], dim=-1)
        logp_tuned_l = F.log_softmax(logits_tuned_0[layer_idx], dim=-1)
        loss_baseline += (p_L * (logp_L - logp_logit_l)).sum(dim=-1).mean().item()
        loss_init += (p_L * (logp_L - logp_tuned_l)).sum(dim=-1).mean().item()

    assert abs(loss_init - loss_baseline) < 1e-6, "Zero-init tuned lens must equal logit lens"

    # 2. Perform 1 optimizer step
    optimizer.zero_grad()
    for layer_idx in range(num_layers):
        h_tilde = translators(hidden_states[layer_idx], layer_idx)
        logits_l = model.lm_head(model.transformer.ln_f(h_tilde))
        logp_l = F.log_softmax(logits_l, dim=-1)
        kl_l = (p_L * (logp_L - logp_l)).sum(dim=-1).mean()
        kl_l.backward()

    optimizer.step()

    # 3. Compute new loss
    loss_after = 0.0
    with torch.no_grad():
        logits_tuned_1 = tuned_lens(model, translators, hidden_states)
        for layer_idx in range(num_layers):
            logp_tuned_l = F.log_softmax(logits_tuned_1[layer_idx], dim=-1)
            loss_after += (p_L * (logp_L - logp_tuned_l)).sum(dim=-1).mean().item()

    assert loss_after < loss_init, f"Loss after step ({loss_after:.6f}) must be lower than init ({loss_init:.6f})"


# --------------------------------------------------------------------------- #
# Slow Integration Tests (Real GPT-2 & Official Checkpoint)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_official_tuned_lens_gpt2_integration():
    """Integration test with pretrained GPT-2 and official Tuned Lens checkpoint.

    Verifies acceptance criteria:
    - Output has L + 1 entries.
    - Layer L equals the model logits.
    """
    from lib.data import load_model
    from lib.registry import get_model_spec
    from ws4_lens.logit_lens import run_model

    spec = get_model_spec("gpt2")
    model, tokenizer, _ = load_model(
        checkpoint=spec["hf_id"],
        kind=spec["kind"],
        tokenizer_kwargs={"add_prefix_space": True},
    )

    official_lens = load_official_tuned_lens(model)
    assert len(official_lens) == 12

    prompt = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model_logits = model(inputs["input_ids"]).logits
        hidden_states = run_model(model, inputs["input_ids"], inputs["attention_mask"])

    logits = tuned_lens(model, official_lens, hidden_states)

    assert len(logits) == 13
    assert torch.allclose(logits[12], model_logits, atol=1e-4)
