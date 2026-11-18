"""Unit tests for ws4_lens.evaluate (Issue L7)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from ws4_lens.evaluate import evaluate_lenses  # noqa: E402
from ws4_lens.tuned_lens import TunedLensTranslators  # noqa: E402


@pytest.fixture(scope="module")
def tiny_gpt2_eval():
    """Create a tiny random GPT-2 model (L=3, d=16, V=50) for fast unit checks."""
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


class MockTokenizer:
    """Mock tokenizer returning deterministic token strings."""

    def decode(self, token_ids: list[int]) -> str:
        tid = token_ids[0]
        if tid == 0:
            return "<bos>"
        return f" tok_{tid}"


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def synthetic_eval_batch():
    """Create a synthetic batch with input_ids and word alignments."""
    input_ids = torch.tensor([[0, 5, 12, 23, 7]])  # len 5
    attention_mask = torch.ones_like(input_ids)
    seq_ids = ["sent_001"]
    # word alignment: word 0 last subword pos 1 -> next_upos 'VERB'
    word_alignments = [[(1, "VERB"), (2, "NOUN")]]
    return [{
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "seq_ids": seq_ids,
        "word_alignments": word_alignments,
    }]


def test_evaluate_lenses_output_format_and_sanity(
    tiny_gpt2_eval,
    mock_tokenizer,
    synthetic_eval_batch,
):
    """Verify evaluate_lenses produces exact contract columns and passes layer L sanity checks."""
    model = tiny_gpt2_eval
    num_layers = 3
    hidden_size = 16

    # Identity translators and Jacobians
    translators = TunedLensTranslators(num_layers, hidden_size)
    jacobians = torch.eye(hidden_size).unsqueeze(0).repeat(num_layers + 1, 1, 1)
    lexicon = {"tok_12": "VERB", "tok_23": "NOUN"}

    df_agg, df_tok = evaluate_lenses(
        model,
        mock_tokenizer,
        model_name="tiny_gpt2",
        dataset_name="synthetic",
        data_batches=synthetic_eval_batch,
        tuned_translators=translators,
        jacobian_matrices=jacobians,
        upos_lexicon=lexicon,
        lenses=("logit", "tuned", "jacobian"),
        max_token_sequences=10,
        verbose=False,
    )

    # 1. Verify Aggregate DataFrame columns per CONTRACTS.md Section 9.3
    expected_agg_cols = [
        "model", "dataset", "lens", "layer", "L", "tau",
        "top1_acc", "entropy_mean", "kl_mean", "upos_top1_acc", "n_positions"
    ]
    assert list(df_agg.columns) == expected_agg_cols
    assert len(df_agg) == 3 * (num_layers + 1)  # 3 lenses x 4 layers = 12 rows

    # 2. Verify Per-Token DataFrame columns per CONTRACTS.md Section 9.4
    expected_tok_cols = [
        "seq_id", "pos", "token", "target_token", "lens", "layer",
        "top1_token", "top1_prob", "entropy", "kl"
    ]
    assert list(df_tok.columns) == expected_tok_cols
    assert len(df_tok) > 0

    # 3. Acceptance criterion Sanity Check:
    # At layer L, all three lenses give the exact same top1_acc
    layer_L_rows = df_agg[df_agg["layer"] == num_layers]
    acc_values = layer_L_rows["top1_acc"].tolist()
    assert len(set(acc_values)) == 1, f"Top-1 accuracies at layer L differ: {acc_values}"

    # 4. Acceptance criterion Sanity Check:
    # At layer L, KL to final layer is 0.0 for all lenses
    kl_values = layer_L_rows["kl_mean"].tolist()
    for kl in kl_values:
        assert abs(kl) < 1e-6, f"KL at layer L is non-zero: {kl}"


def test_evaluate_lenses_wikitext_no_upos(
    tiny_gpt2_eval,
    mock_tokenizer,
    synthetic_eval_batch,
):
    """Verify upos_top1_acc is null when upos_lexicon is None (WikiText run)."""
    model = tiny_gpt2_eval
    num_layers = 3
    hidden_size = 16

    translators = TunedLensTranslators(num_layers, hidden_size)
    jacobians = torch.eye(hidden_size).unsqueeze(0).repeat(num_layers + 1, 1, 1)

    df_agg, _ = evaluate_lenses(
        model,
        mock_tokenizer,
        model_name="tiny_gpt2",
        dataset_name="wikitext",
        data_batches=synthetic_eval_batch,
        tuned_translators=translators,
        jacobian_matrices=jacobians,
        upos_lexicon=None,
        lenses=("logit", "tuned", "jacobian"),
        verbose=False,
    )

    # All upos_top1_acc values must be null/None
    assert df_agg["upos_top1_acc"].isna().all()
