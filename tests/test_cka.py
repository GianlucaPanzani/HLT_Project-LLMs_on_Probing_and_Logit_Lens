"""Tests for ws2_compression.cka – linear_cka and sample_rows (issues C1 #15, C2 #16).

All tests use synthetic data and require no downloads or torch.
"""

import warnings

import numpy as np
import pytest

from ws2_compression.cka import cka_matrix, linear_cka, sample_rows

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
# linear_cka: constant representations & scale invariance
# ------------------------------------------------------------------ #

class TestLinearCkaConstantRepresentations:
    """Zero-variance and constant input handling without breaking scale invariance."""

    @pytest.mark.parametrize("val", [0.0, 1.0, 0.1, -0.7, 3.14159, 1e-20, 1e20])
    def test_constant_matrices_return_zero(self, val):
        """Constant matrices have zero sample variance and must return 0.0."""
        X_const = np.full((10, 5), val)
        assert linear_cka(X_const, X_const) == 0.0

    def test_constant_vs_varying_returns_zero(self, rng):
        X_const = np.full((50, 10), 0.1)
        Y_var = rng.standard_normal((50, 10))
        assert linear_cka(X_const, Y_var) == 0.0
        assert linear_cka(Y_var, X_const) == 0.0

    def test_scale_invariance_small_valued_inputs(self, rng):
        """Scale invariance must hold for legitimate small-valued representations."""
        X_tiny = 1e-20 * rng.standard_normal((50, 10))
        assert linear_cka(X_tiny, X_tiny) == pytest.approx(1.0, abs=1e-10)
        assert linear_cka(X_tiny, 3.0 * X_tiny) == pytest.approx(1.0, abs=1e-10)

    def test_partially_constant_columns_preserved(self, rng):
        """If some columns are constant but others vary, CKA should measure varying columns."""
        X = rng.standard_normal((50, 5))
        X[:, 0] = 0.1  # Column 0 is constant
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-10)


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


# ------------------------------------------------------------------ #
# sample_rows: token sampling (issue C2, #16)
# ------------------------------------------------------------------ #

class TestSampleRows:
    """Tests for aligned token sampling."""

    def test_all_valid(self):
        mask = np.ones(100, dtype=bool)
        idx = sample_rows(mask, 50, seed=42)
        assert len(idx) == 50
        assert np.all(mask[idx])
        assert np.all(np.diff(idx) > 0)

    def test_subset_valid(self):
        mask = np.zeros(100, dtype=bool)
        mask[10:30] = True
        idx = sample_rows(mask, 15, seed=42)
        assert len(idx) == 15
        assert np.all(mask[idx])

    def test_fewer_valid_than_requested(self):
        mask = np.zeros(100, dtype=bool)
        mask[0:5] = True
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
# cka_matrix: full layer-pair CKA matrix (issue C3, #17)
# ------------------------------------------------------------------ #

class TestCkaMatrix:
    """Tests for full layer-pair CKA matrix generation."""

    @pytest.fixture()
    def mock_loaders(self):
        rng = np.random.default_rng(42)
        n_rows, hidden_dim = 100, 32

        # Model A: 12 layers (0..12), Model B: 6 layers (0..6)
        data_a = {la: rng.standard_normal((n_rows, hidden_dim)) for la in range(13)}
        # Model B layers correlate with Model A layers
        data_b = {lb: data_a[2 * lb] + 0.1 * rng.standard_normal((n_rows, hidden_dim)) for lb in range(7)}

        def mock_meta(model, init, pooling, treebank, split):
            if model == "bert":
                return {"num_layers": 12, "hidden_dim": hidden_dim}
            elif model == "distilbert":
                return {"num_layers": 6, "hidden_dim": hidden_dim}
            raise ValueError(f"Unknown model {model}")

        def mock_layer(model, init, pooling, treebank, split, layer):
            if model == "bert":
                return data_a[layer]
            elif model == "distilbert":
                return data_b[layer]
            raise ValueError(f"Unknown model {model}")

        return mock_meta, mock_layer

    def test_matrix_shape_and_columns(self, mock_loaders):
        mock_meta, mock_layer = mock_loaders
        df = cka_matrix(
            "bert", "distilbert",
            n_tokens=50, seed=42,
            _load_meta=mock_meta, _load_layer=mock_layer,
        )
        assert len(df) == 13 * 7  # 91 layer pairs
        assert list(df.columns) == ["layer_a", "layer_b", "cka", "n_tokens", "seed"]
        assert (df["cka"] >= 0.0).all() and (df["cka"] <= 1.0 + 1e-6).all()
        assert (df["n_tokens"] == 50).all()

    def test_self_similarity_diagonal(self, mock_loaders):
        mock_meta, mock_layer = mock_loaders
        df = cka_matrix(
            "bert", "bert",
            n_tokens=50, seed=42,
            _load_meta=mock_meta, _load_layer=mock_layer,
        )
        assert len(df) == 13 * 13
        diag = df[df["layer_a"] == df["layer_b"]]
        for _, row in diag.iterrows():
            assert pytest.approx(row["cka"], abs=1e-5) == 1.0
