"""Unit tests for ws4_lens.jacobian_lens (Issue L5)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from ws4_lens.jacobian_lens import (  # noqa: E402
    estimate_jacobians,
    jacobian_lens,
    load_jacobian_checkpoint,
    save_jacobian_checkpoint,
)
from ws4_lens.logit_lens import logit_lens, run_model  # noqa: E402


@pytest.fixture(scope="module")
def tiny_gpt2():
    """Create a tiny random GPT-2 (L=3, d=16, V=50) for fast unit checks."""
    torch.manual_seed(42)
    cfg = GPT2Config(
        vocab_size=50,
        n_positions=32,
        n_embd=16,
        n_layer=3,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture(scope="module")
def tiny_batch():
    """Create a synthetic token batch with BOS token at position 0."""
    input_ids = torch.tensor([[0, 5, 12, 23, 7]])  # pos 0 is BOS=0
    attn_mask = torch.ones_like(input_ids)
    return [(input_ids, attn_mask)]


def test_jacobian_shape_and_identity_at_L(tiny_gpt2, tiny_batch):
    """Verify estimate_jacobians returns [L + 1, d, d] in float32 with J_L == I."""
    jacobians = estimate_jacobians(
        tiny_gpt2,
        tiny_batch,
        n_positions=1,
        seed=42,
        chunk_size=8,
        verbose=False,
    )
    assert jacobians.shape == (4, 16, 16)
    assert jacobians.dtype == torch.float32
    assert torch.allclose(jacobians[-1], torch.eye(16), atol=1e-7)


def test_jacobian_finite_difference(tiny_gpt2, tiny_batch):
    """Verify autograd Jacobian matches central finite differences with rel_err < 1e-3."""
    rng = np.random.default_rng(42)
    picked_t = [1, 2, 3, 4][int(rng.choice(4, size=1, replace=False)[0])]

    jacobians = estimate_jacobians(
        tiny_gpt2,
        tiny_batch,
        n_positions=1,
        seed=42,
        chunk_size=8,
        verbose=False,
    )

    input_ids, attn_mask = tiny_batch[0]
    eps = 1e-4
    i_idx, j_idx = 5, 7

    for layer_idx in range(3):
        def eval_r_l_entry(delta: float) -> float:
            captured = []

            def perturb_hook(_m, args):
                h = args[0].clone()
                h[0, picked_t, j_idx] += delta
                return (h,) + args[1:]

            def cap_hook(_m, _inp, out):
                captured.append((out[0] if isinstance(out, tuple) else out).detach())

            p_h = tiny_gpt2.transformer.h[layer_idx].register_forward_pre_hook(perturb_hook)
            c_h = tiny_gpt2.transformer.h[-1].register_forward_hook(cap_hook)
            with torch.no_grad():
                tiny_gpt2(input_ids=input_ids, attention_mask=attn_mask)
            p_h.remove()
            c_h.remove()
            return captured[0][0, picked_t, i_idx].item()

        fd_val = (eval_r_l_entry(+eps) - eval_r_l_entry(-eps)) / (2.0 * eps)
        auto_val = jacobians[layer_idx, i_idx, j_idx].item()
        rel_err = abs(fd_val - auto_val) / max(abs(fd_val), abs(auto_val), 1e-8)
        assert rel_err < 1e-3, f"Rel error too high at layer {layer_idx}: {rel_err}"


def test_jacobian_lens_layer_L_matches_model_logits(tiny_gpt2, tiny_batch):
    """Verify jacobian_lens output at layer L equals model.logits bit-for-bit."""
    input_ids, attn_mask = tiny_batch[0]
    hidden_states = run_model(tiny_gpt2, input_ids, attn_mask)

    jacobians = estimate_jacobians(
        tiny_gpt2,
        tiny_batch,
        n_positions=1,
        seed=42,
        chunk_size=8,
        verbose=False,
    )
    j_logits = jacobian_lens(tiny_gpt2, jacobians, hidden_states)

    with torch.no_grad():
        official_logits = tiny_gpt2(input_ids=input_ids, attention_mask=attn_mask).logits

    assert torch.allclose(j_logits[-1], official_logits, atol=1e-5)


def test_jacobian_lens_identity_matches_logit_lens(tiny_gpt2, tiny_batch):
    """Verify jacobian_lens with identity matrices matches Standard Logit Lens identically."""
    input_ids, attn_mask = tiny_batch[0]
    hidden_states = run_model(tiny_gpt2, input_ids, attn_mask)

    j_identity = torch.eye(16).unsqueeze(0).repeat(4, 1, 1)
    j_logits = jacobian_lens(tiny_gpt2, j_identity, hidden_states)
    l_logits = logit_lens(tiny_gpt2, hidden_states)

    for j_l, l_l in zip(j_logits, l_logits):
        assert torch.allclose(j_l, l_l, atol=1e-5)


def test_save_and_load_jacobian_checkpoint(tmp_path):
    """Verify serialization and deserialization roundtrip of Jacobian checkpoint."""
    j_tensor = torch.randn(13, 32, 32, dtype=torch.float32)
    j_tensor[-1] = torch.eye(32)

    saved_path = save_jacobian_checkpoint(j_tensor, "test_model", output_dir=tmp_path)
    assert saved_path.exists()
    assert saved_path.name == "test_model.pt"

    loaded = load_jacobian_checkpoint("test_model", checkpoint_dir=tmp_path)
    assert loaded.shape == (13, 32, 32)
    assert loaded.dtype == torch.float32
    assert torch.allclose(j_tensor, loaded)


def test_validation_errors(tiny_gpt2, tiny_batch, tmp_path):
    """Verify validation checks on parameters and dimensions."""
    # Invalid n_positions
    with pytest.raises(ValueError, match="n_positions must be positive"):
        estimate_jacobians(tiny_gpt2, tiny_batch, n_positions=0)

    # Invalid chunk_size
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        estimate_jacobians(tiny_gpt2, tiny_batch, chunk_size=0)

    # No valid positions (all masked)
    masked_batch = [(torch.tensor([[0, 1, 2]]), torch.zeros(1, 3))]
    with pytest.raises(ValueError, match="No valid non-BOS, non-padding positions"):
        estimate_jacobians(tiny_gpt2, masked_batch, n_positions=1)

    # Missing checkpoint
    with pytest.raises(FileNotFoundError, match="Jacobian checkpoint not found"):
        load_jacobian_checkpoint("non_existent_model", checkpoint_dir=tmp_path)

    # Invalid jacobians tensor dimensions for jacobian_lens
    hs = [torch.randn(1, 4, 16) for _ in range(4)]
    with pytest.raises(ValueError, match="jacobians must be a 3D tensor"):
        jacobian_lens(tiny_gpt2, torch.eye(16), hs)

    with pytest.raises(ValueError, match="Number of Jacobian matrices"):
        jacobian_lens(tiny_gpt2, torch.eye(16).unsqueeze(0).repeat(3, 1, 1), hs)

    with pytest.raises(ValueError, match="Jacobian matrix dimensions"):
        jacobian_lens(tiny_gpt2, torch.eye(32).unsqueeze(0).repeat(4, 1, 1), hs)
