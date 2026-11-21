"""Tests for ws2_compression.cka – linear_cka, sample_rows, cka_matrix.

All tests use synthetic data and require no downloads or torch.
"""

import warnings

import numpy as np
import pytest

from ws2_compression.cka import linear_cka, sample_rows

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def X(rng):
    return rng.standard_normal((200, 50))


@pytest.fixture()
def Y(rng):
    return rng.standard_normal((200, 30))


# ------------------------------------------------------------------ #
# linear_cka: core properties
# ------------------------------------------------------------------ #

class TestLinearCkaSelfSimilarity:
    """CKA(X, X) must be 1."""

    def test_self_similarity(self, X):
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-10)

    def test_self_similarity_different_dims(self, rng):
        X = rng.standard_normal((300, 768))
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-10)


class TestLinearCkaInvariance:
    """Orthogonal and isotropic scale/shift invariance."""

    def test_orthogonal_invariance(self, rng, X):
        # Random orthogonal Q via QR decomposition
        M = rng.standard_normal((X.shape[1], X.shape[1]))
        Q, _ = np.linalg.qr(M)
        assert linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-10)

    def test_isotropic_scale_shift(self, X):
        assert linear_cka(X, 3.0 * X + 5.0) == pytest.approx(1.0, abs=1e-10)

    def test_negative_scale(self, X):
        assert linear_cka(X, -2.0 * X) == pytest.approx(1.0, abs=1e-10)


class TestLinearCkaSymmetry:
    """CKA(X, Y) == CKA(Y, X)."""

    def test_symmetry(self, X, Y):
        assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-14)


class TestLinearCkaIndependence:
    """Two independent random matrices should have CKA near 0."""

    def test_independent_low(self):
        rng = np.random.default_rng(99)
        A = rng.standard_normal((2000, 50))
        B = rng.standard_normal((2000, 50))
        assert linear_cka(A, B) < 0.1


class TestLinearCkaDifferentDims:
    """d1 != d2 should work."""

    def test_d1_768_d2_2048(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((500, 768))
        Y = rng.standard_normal((500, 2048))
        val = linear_cka(X, Y)
        assert 0.0 <= val <= 1.0


class TestLinearCkaRange:
    """Output should be in [0, 1]."""

    def test_range(self, X, Y):
        val = linear_cka(X, Y)
        assert 0.0 <= val <= 1.0 + 1e-10


# ------------------------------------------------------------------ #
# linear_cka: error handling
# ------------------------------------------------------------------ #

class TestLinearCkaErrors:
    """Validation errors."""

    def test_mismatched_rows(self):
        with pytest.raises(ValueError, match="Row counts must match"):
            linear_cka(np.ones((10, 5)), np.ones((11, 5)))

    def test_n_less_than_2(self):
        with pytest.raises(ValueError, match="N >= 2"):
            linear_cka(np.ones((1, 5)), np.ones((1, 5)))

    def test_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            linear_cka(np.ones(10), np.ones(10))

    def test_3d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            linear_cka(np.ones((2, 3, 4)), np.ones((2, 3, 4)))

    def test_nan_in_X(self):
        X = np.ones((10, 5))
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            linear_cka(X, np.ones((10, 5)))

    def test_nan_in_Y(self):
        Y = np.ones((10, 5))
        Y[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            linear_cka(np.ones((10, 5)), Y)

    def test_inf_in_X(self):
        X = np.ones((10, 5))
        X[0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            linear_cka(X, np.ones((10, 5)))

    def test_zero_denominator(self):
        # Constant matrices -> centred to zero -> denominator = 0
        X = np.ones((10, 5))
        Y = np.ones((10, 3))
        assert linear_cka(X, Y) == 0.0


# ------------------------------------------------------------------ #
# sample_rows
# ------------------------------------------------------------------ #

class TestSampleRows:
    """Tests for aligned token sampling."""

    def test_all_valid(self):
        mask = np.ones(100, dtype=bool)
        idx = sample_rows(mask, 50, seed=42)
        assert len(idx) == 50
        assert np.all(mask[idx])
        # Sorted
        assert np.all(np.diff(idx) > 0)

    def test_subset_valid(self):
        mask = np.zeros(100, dtype=bool)
        mask[10:30] = True  # 20 valid
        idx = sample_rows(mask, 15, seed=42)
        assert len(idx) == 15
        assert np.all(mask[idx])

    def test_fewer_valid_than_requested(self):
        mask = np.zeros(100, dtype=bool)
        mask[0:5] = True  # only 5 valid
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            idx = sample_rows(mask, 50, seed=42)
            assert len(w) == 1
            assert "only 5" in str(w[0].message)
        assert len(idx) == 5
        assert np.all(mask[idx])

    def test_reproducibility(self):
        mask = np.ones(200, dtype=bool)
        idx1 = sample_rows(mask, 50, seed=42)
        idx2 = sample_rows(mask, 50, seed=42)
        np.testing.assert_array_equal(idx1, idx2)

    def test_different_seed(self):
        mask = np.ones(200, dtype=bool)
        idx1 = sample_rows(mask, 50, seed=42)
        idx2 = sample_rows(mask, 50, seed=99)
        assert not np.array_equal(idx1, idx2)

    def test_no_invalid_indices(self):
        mask = np.array([True, False, True, False, True])
        idx = sample_rows(mask, 100, seed=42)
        assert len(idx) == 3
        for i in idx:
            assert mask[i]

    def test_1d_requirement(self):
        with pytest.raises(ValueError, match="1-D"):
            sample_rows(np.ones((5, 5), dtype=bool), 3)


# ------------------------------------------------------------------ #
# cka_matrix (synthetic features mock)
# ------------------------------------------------------------------ #

class TestCkaMatrix:
    """Test full CKA matrix computation with synthetic feature layers."""

    @pytest.fixture()
    def mock_feature_store(self, monkeypatch):
        """Mock load_meta and load_layer so no files on disk are required."""
        rng = np.random.default_rng(42)
        n_rows = 100
        d = 32

        # 13 layers for bert (0..12), 7 layers for distilbert (0..6)
        bert_layers = {layer_idx: rng.standard_normal((n_rows, d)) for layer_idx in range(13)}
        distil_layers = {layer_idx: rng.standard_normal((n_rows, d)) for layer_idx in range(7)}

        # Introduce a couple NaN rows at the same positions
        for layer_idx in range(13):
            bert_layers[layer_idx][5] = np.nan
        for layer_idx in range(7):
            distil_layers[layer_idx][5] = np.nan

        def fake_load_meta(model, init, pooling, treebank, split):
            num_layers = 12 if model == "bert" else 6
            return {"num_layers": num_layers, "hidden_size": d}

        def fake_load_layer(model, init, pooling, treebank, split, layer):
            if model == "bert":
                return bert_layers[layer]
            return distil_layers[layer]

        monkeypatch.setattr("ws2_compression.cka.load_meta", fake_load_meta)
        monkeypatch.setattr("ws2_compression.cka.load_layer", fake_load_layer)
        return {"bert": bert_layers, "distilbert": distil_layers}

    def test_bert_vs_distilbert_shape_and_columns(self, mock_feature_store):
        from ws2_compression.cka import cka_matrix

        df = cka_matrix("bert", "distilbert", n_tokens=50, seed=42)
        assert len(df) == 13 * 7  # 91 rows
        assert list(df.columns) == ["layer_a", "layer_b", "cka", "n_tokens", "seed"]
        assert df["n_tokens"].iloc[0] == 50
        assert df["seed"].iloc[0] == 42
        assert set(df["layer_a"].unique()) == set(range(13))
        assert set(df["layer_b"].unique()) == set(range(7))
        assert (df["cka"] >= 0.0).all() and (df["cka"] <= 1.0 + 1e-10).all()

    def test_bert_vs_bert_sanity_diagonal_and_symmetry(self, monkeypatch):
        """bert vs bert must have 1s on diagonal and be symmetric."""
        rng = np.random.default_rng(123)
        n_rows = 80
        d = 32
        layers = {layer_idx: rng.standard_normal((n_rows, d)) for layer_idx in range(5)}

        def fake_load_meta(model, init, pooling, treebank, split):
            return {"num_layers": 4, "hidden_size": d}

        def fake_load_layer(model, init, pooling, treebank, split, layer):
            return layers[layer]

        monkeypatch.setattr("ws2_compression.cka.load_meta", fake_load_meta)
        monkeypatch.setattr("ws2_compression.cka.load_layer", fake_load_layer)

        from ws2_compression.cka import cka_matrix

        df = cka_matrix("bert", "bert", n_tokens=50, seed=42)
        assert len(df) == 5 * 5  # 25 rows

        pivot = df.pivot(index="layer_a", columns="layer_b", values="cka")
        matrix = pivot.to_numpy()

        # Diagonal must be 1.0
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-10)

        # Matrix must be symmetric
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)

