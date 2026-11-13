"""Tests for lib.paths – no network, no torch."""

from pathlib import Path

from lib.paths import (
    ROOT,
    cka_result_path,
    control_path,
    feature_dir,
    layer_path,
    lens_result_path,
    probing_result_path,
    raw_dir,
    word_table_path,
)


class TestRoot:
    def test_root_exists(self):
        assert ROOT.is_dir()

    def test_root_contains_src(self):
        assert (ROOT / "src").is_dir()


class TestRawDir:
    def test_pattern(self):
        p = raw_dir("ewt")
        assert p == ROOT / "data" / "raw" / "ewt"


class TestWordTablePath:
    def test_pattern(self):
        p = word_table_path("ewt", "train")
        assert p == ROOT / "data" / "processed" / "ewt_train.parquet"


class TestControlPath:
    def test_pattern(self):
        p = control_path("ewt")
        assert p == ROOT / "data" / "processed" / "control_ewt.json"


class TestFeatureDir:
    def test_pattern(self):
        p = feature_dir("gpt2", "pretrained", "first", "ewt", "train")
        assert p == ROOT / "features" / "gpt2" / "pretrained" / "first" / "ewt_train"


class TestLayerPath:
    def test_pattern(self):
        p = layer_path("bert", "pretrained", "first", "ewt", "test", 5)
        assert p == ROOT / "features" / "bert" / "pretrained" / "first" / "ewt_test" / "layer_05.npy"

    def test_zero_padded(self):
        p = layer_path("bert", "pretrained", "first", "ewt", "test", 0)
        assert p.name == "layer_00.npy"

    def test_double_digit(self):
        p = layer_path("mgpt", "pretrained", "first", "ewt", "test", 12)
        assert p.name == "layer_12.npy"


class TestFeaturesDirEnvVar:
    def test_hlt_features_dir(self, monkeypatch):
        monkeypatch.setenv("HLT_FEATURES_DIR", "/tmp/my_features")
        p = feature_dir("bert", "pretrained", "first", "ewt", "train")
        assert p == Path("/tmp/my_features/bert/pretrained/first/ewt_train")

    def test_layer_path_with_env(self, monkeypatch):
        monkeypatch.setenv("HLT_FEATURES_DIR", "/tmp/my_features")
        p = layer_path("bert", "pretrained", "first", "ewt", "train", 3)
        assert p == Path("/tmp/my_features/bert/pretrained/first/ewt_train/layer_03.npy")

    def test_default_without_env(self, monkeypatch):
        monkeypatch.delenv("HLT_FEATURES_DIR", raising=False)
        p = feature_dir("bert", "pretrained", "first", "ewt", "train")
        assert p == ROOT / "features" / "bert" / "pretrained" / "first" / "ewt_train"


class TestProbingResultPath:
    def test_pattern(self):
        p = probing_result_path("bert", "ewt", "linear", "pretrained", "first")
        assert p == ROOT / "results" / "probing" / "bert__ewt__linear__pretrained__first.parquet"


class TestCkaResultPath:
    def test_pattern(self):
        p = cka_result_path("bert", "distilbert", "ewt")
        assert p == ROOT / "results" / "cka" / "bert__vs__distilbert__ewt.parquet"


class TestLensResultPath:
    def test_aggregate(self):
        p = lens_result_path("gpt2", "pud_en")
        assert p == ROOT / "results" / "lens" / "gpt2__pud_en.parquet"

    def test_per_token(self):
        p = lens_result_path("gpt2", "pud_en", per_token=True)
        assert p == ROOT / "results" / "lens" / "tokens" / "gpt2__pud_en.parquet"
