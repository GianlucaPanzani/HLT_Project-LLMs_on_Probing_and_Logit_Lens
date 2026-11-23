"""Tests for ws2_compression.mapping – layer mapping comparison.

Uses synthetic probing and CKA DataFrames. No downloads or torch required.
"""

import numpy as np
import pandas as pd
import pytest

from ws2_compression.mapping import (
    build_layer_mapping_table,
    compute_cog_comparison,
)

# ------------------------------------------------------------------ #
# Fixtures: synthetic probing and CKA data
# ------------------------------------------------------------------ #

def _make_probing_df(model: str, num_layers: int, treebank: str = "ewt") -> pd.DataFrame:
    """Create a synthetic probing results DataFrame (CONTRACTS.md §6)."""
    rng = np.random.default_rng(42 if model == "bert" else 99)
    rows = []
    for layer in range(num_layers + 1):
        tau = layer / num_layers
        # Accuracy increases roughly with depth, then drops
        acc_upos = 0.3 + 0.5 * np.sin(np.pi * tau) + rng.normal(0, 0.02)
        acc_ctrl = 0.1 + 0.05 * rng.random()
        for task, acc in [("upos", acc_upos), ("control", acc_ctrl)]:
            rows.append({
                "model": model,
                "train_treebank": treebank,
                "eval_treebank": treebank,
                "lang": "en",
                "kind": "encoder",
                "probe": "linear",
                "task": task,
                "init": "pretrained",
                "pooling": "first",
                "layer": layer,
                "L": num_layers,
                "tau": tau,
                "acc": acc,
                "n_train": 10000,
                "n_eval": 2000,
                "hparams": '{"C": 1.0}',
                "seed": 42,
            })
    return pd.DataFrame(rows)


def _make_cka_df(L_a: int = 12, L_b: int = 6) -> pd.DataFrame:
    """Create a synthetic CKA matrix DataFrame (CONTRACTS.md §8).

    Simulates a structure where layer 2l of model A is most similar to
    layer l of model B (the 1:2 mapping).
    """
    records = []
    for la in range(L_a + 1):
        for lb in range(L_b + 1):
            # Higher CKA when la ≈ 2 * lb
            dist = abs(la - 2 * lb)
            cka = max(0.0, 1.0 - 0.08 * dist)
            records.append({
                "layer_a": la,
                "layer_b": lb,
                "cka": cka,
                "n_tokens": 5000,
                "seed": 42,
            })
    return pd.DataFrame(records)


@pytest.fixture()
def probing_bert():
    return _make_probing_df("bert", num_layers=12)


@pytest.fixture()
def probing_distilbert():
    return _make_probing_df("distilbert", num_layers=6)


@pytest.fixture()
def cka_df():
    return _make_cka_df()


# ------------------------------------------------------------------ #
# build_layer_mapping_table
# ------------------------------------------------------------------ #

class TestBuildLayerMappingTable:
    """Test the mapping table construction."""

    def test_shape(self, probing_bert, probing_distilbert, cka_df):
        table = build_layer_mapping_table(probing_bert, probing_distilbert, cka_df)
        # 7 DistilBERT layers (0..6)
        assert len(table) == 7

    def test_columns_present(self, probing_bert, probing_distilbert, cka_df):
        table = build_layer_mapping_table(probing_bert, probing_distilbert, cka_df)
        expected = {
            "distilbert_layer", "distilbert_acc", "distilbert_sel",
            "rigid_bert_layer", "rigid_bert_acc", "rigid_bert_sel",
            "rigid_acc_diff", "rigid_sel_diff",
            "cka_best_bert_layer", "cka_best_bert_acc", "cka_best_bert_sel",
            "cka_best_cka", "cka_best_acc_diff", "cka_best_sel_diff",
        }
        assert expected.issubset(set(table.columns))

    def test_rigid_mapping(self, probing_bert, probing_distilbert, cka_df):
        table = build_layer_mapping_table(probing_bert, probing_distilbert, cka_df)
        # Rigid mapping: DistilBERT layer l -> BERT layer 2l
        for _, row in table.iterrows():
            assert row["rigid_bert_layer"] == 2 * row["distilbert_layer"]

    def test_cka_best_matches_diagonal(self, probing_bert, probing_distilbert, cka_df):
        """With our synthetic CKA data, the best BERT layer should be 2l."""
        table = build_layer_mapping_table(probing_bert, probing_distilbert, cka_df)
        for _, row in table.iterrows():
            # Our synthetic CKA data peaks at la = 2*lb
            assert row["cka_best_bert_layer"] == 2 * row["distilbert_layer"]

    def test_acc_diff_sign(self, probing_bert, probing_distilbert, cka_df):
        """Verify acc difference = distilbert - bert."""
        table = build_layer_mapping_table(probing_bert, probing_distilbert, cka_df)
        for _, row in table.iterrows():
            expected_diff = row["distilbert_acc"] - row["rigid_bert_acc"]
            assert row["rigid_acc_diff"] == pytest.approx(expected_diff, abs=1e-10)


# ------------------------------------------------------------------ #
# compute_cog_comparison
# ------------------------------------------------------------------ #

class TestComputeCogComparison:
    """Test CoG computation for both models."""

    def test_keys_present(self, probing_bert, probing_distilbert):
        result = compute_cog_comparison(probing_bert, probing_distilbert)
        expected_keys = {
            "bert_cog_norm", "distilbert_cog_norm",
            "bert_selectivity", "distilbert_selectivity",
            "bert_tau", "distilbert_tau",
        }
        assert expected_keys == set(result.keys())

    def test_cog_norm_range(self, probing_bert, probing_distilbert):
        result = compute_cog_comparison(probing_bert, probing_distilbert)
        for key in ["bert_cog_norm", "distilbert_cog_norm"]:
            val = result[key]
            if np.isfinite(val):
                assert 0.0 <= val <= 1.0

    def test_selectivity_length(self, probing_bert, probing_distilbert):
        result = compute_cog_comparison(probing_bert, probing_distilbert)
        # BERT has 13 entries (layers 0..12)
        assert len(result["bert_selectivity"]) == 13
        assert len(result["bert_tau"]) == 13
        # DistilBERT has 7 entries (layers 0..6)
        assert len(result["distilbert_selectivity"]) == 7
        assert len(result["distilbert_tau"]) == 7

    def test_tau_endpoints(self, probing_bert, probing_distilbert):
        result = compute_cog_comparison(probing_bert, probing_distilbert)
        assert result["bert_tau"][0] == pytest.approx(0.0)
        assert result["bert_tau"][-1] == pytest.approx(1.0)
        assert result["distilbert_tau"][0] == pytest.approx(0.0)
        assert result["distilbert_tau"][-1] == pytest.approx(1.0)
