"""Unit tests for ws4_lens.metrics (Issue L2)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from ws4_lens.metrics import entropy, kl_to_final, top1_accuracy  # noqa: E402, I001


# --------------------------------------------------------------------------- #
# Tests for entropy()
# --------------------------------------------------------------------------- #

def test_entropy_uniform_distribution():
    """Uniform distribution over V outcomes must have entropy exactly equal to ln(V)."""
    vocab_size = 8
    logits = torch.zeros(vocab_size)
    ent = entropy(logits).item()
    expected = math.log(vocab_size)
    assert abs(ent - expected) < 1e-6


def test_entropy_deterministic_distribution():
    """One-hot / peaked distribution must have entropy close to 0.0."""
    logits = torch.tensor([100.0, 0.0, 0.0, 0.0])
    ent = entropy(logits).item()
    assert abs(ent - 0.0) < 1e-6


def test_entropy_shape_and_dtype_preservation():
    """Test multidimensional shape reduction and float16 auto-casting to float32."""
    # Input shape: [2, 3, 10], dtype: float16
    logits_fp16 = torch.randn(2, 3, 10, dtype=torch.float16)
    ent = entropy(logits_fp16)

    assert ent.shape == (2, 3)
    assert ent.dtype == torch.float32
    assert (ent >= 0.0).all()


def test_entropy_non_negative():
    """Entropy must be non-negative for any random input."""
    logits = torch.randn(5, 50)
    ent = entropy(logits)
    assert (ent >= 0.0).all()


# --------------------------------------------------------------------------- #
# Tests for kl_to_final()
# --------------------------------------------------------------------------- #

def test_kl_to_final_self_divergence():
    """KL divergence of a distribution with itself must be identically 0.0."""
    logits = torch.randn(3, 4, 16)
    kl = kl_to_final(logits, logits)

    assert kl.shape == (3, 4)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_kl_to_final_non_negative():
    """KL divergence KL(P_L || P_l) must be non-negative everywhere."""
    logits_l = torch.randn(4, 32)
    logits_L = torch.randn(4, 32)
    kl = kl_to_final(logits_l, logits_L)

    assert (kl >= 0.0).all()


def test_kl_to_final_float16_casting():
    """Ensure float16 inputs are cast to float32 and produce float32 outputs."""
    logits_l = torch.randn(2, 5, 20, dtype=torch.float16)
    logits_L = torch.randn(2, 5, 20, dtype=torch.float16)
    kl = kl_to_final(logits_l, logits_L)

    assert kl.shape == (2, 5)
    assert kl.dtype == torch.float32


def test_kl_to_final_known_binary_value():
    """Verify analytical value for two known distributions."""
    # P_L = [0.5, 0.5] (logits [0, 0])
    # P_l = [0.8, 0.2] (logits [log(0.8), log(0.2)])
    p_l = torch.tensor([0.8, 0.2])
    logits_l = torch.log(p_l)
    logits_L = torch.tensor([0.0, 0.0])  # P_L = [0.5, 0.5]

    # KL(P_L || P_l) = 0.5 * ln(0.5 / 0.8) + 0.5 * ln(0.5 / 0.2)
    expected_kl = 0.5 * math.log(0.5 / 0.8) + 0.5 * math.log(0.5 / 0.2)
    kl = kl_to_final(logits_l, logits_L).item()
    assert abs(kl - expected_kl) < 1e-5


# --------------------------------------------------------------------------- #
# Tests for top1_accuracy()
# --------------------------------------------------------------------------- #

def test_top1_accuracy_all_correct():
    """Accuracy should be 1.0 when all predictions match targets."""
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    targets = torch.tensor([0, 1])
    assert top1_accuracy(logits, targets) == 1.0


def test_top1_accuracy_all_wrong():
    """Accuracy should be 0.0 when no predictions match targets."""
    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    targets = torch.tensor([1, 0])
    assert top1_accuracy(logits, targets) == 0.0


def test_top1_accuracy_with_ignored_positions():
    """Negative target indices (such as -100) must be excluded from evaluation."""
    logits = torch.tensor([
        [10.0, 0.0],  # pred 0, target 0 -> correct
        [0.0, 10.0],  # pred 1, target 1 -> correct
        [10.0, 0.0],  # pred 0, target 1 -> wrong
        [10.0, 0.0],  # pred 0, target -100 -> ignored!
        [0.0, 10.0],  # pred 1, target -1 -> ignored!
    ])
    targets = torch.tensor([0, 1, 1, -100, -1])

    # Valid positions count is 3 (first 3). Correct count is 2 -> accuracy = 2/3
    acc = top1_accuracy(logits, targets)
    assert abs(acc - (2.0 / 3.0)) < 1e-5


def test_top1_accuracy_all_ignored():
    """If all targets are negative (ignored), return 0.0 without division by zero."""
    logits = torch.randn(3, 5)
    targets = torch.tensor([-100, -100, -100])
    assert top1_accuracy(logits, targets) == 0.0


def test_top1_accuracy_multidimensional():
    """Test accuracy over batched sequences [B, T, V]."""
    logits = torch.zeros(2, 4, 3)
    # Batch 0: predictions [0, 1, 2, 0]
    logits[0, 0, 0] = 10.0
    logits[0, 1, 1] = 10.0
    logits[0, 2, 2] = 10.0
    logits[0, 3, 0] = 10.0

    # Batch 1: predictions [1, 1, 1, 1]
    logits[1, :, 1] = 10.0

    targets = torch.tensor([
        [0, 1, 2, 1],  # 3 correct, 1 wrong
        [1, 1, -100, -100],  # 2 correct, 2 ignored
    ])

    # Valid positions: 4 + 2 = 6. Correct: 3 + 2 = 5 -> accuracy = 5/6
    acc = top1_accuracy(logits, targets)
    assert abs(acc - (5.0 / 6.0)) < 1e-5
