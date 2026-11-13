"""Tests for scripts/00_download_data.py – no network required."""

import sys
from pathlib import Path

# Add scripts/ so we can import the download module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lib.registry import get_treebank_spec


class TestBuildUrl:
    """Test the URL builder without downloading anything."""

    def setup_method(self):
        # Import here so that sys.path is already set up.
        from importlib import import_module

        self.mod = import_module("00_download_data")

    def test_ewt_train(self):
        spec = get_treebank_spec("ewt")
        url = self.mod.build_url(spec, "train")
        assert url == (
            "https://raw.githubusercontent.com/"
            "UniversalDependencies/UD_English-EWT/r2.19/"
            "en_ewt-ud-train.conllu"
        )

    def test_ewt_test(self):
        spec = get_treebank_spec("ewt")
        url = self.mod.build_url(spec, "test")
        assert url.endswith("en_ewt-ud-test.conllu")

    def test_isdt_dev(self):
        spec = get_treebank_spec("isdt")
        url = self.mod.build_url(spec, "dev")
        assert "UD_Italian-ISDT" in url
        assert url.endswith("it_isdt-ud-dev.conllu")

    def test_pud_en_test(self):
        spec = get_treebank_spec("pud_en")
        url = self.mod.build_url(spec, "test")
        assert "UD_English-PUD" in url
        assert "r2.19" in url
        assert url.endswith("en_pud-ud-test.conllu")

    def test_pud_it_test(self):
        spec = get_treebank_spec("pud_it")
        url = self.mod.build_url(spec, "test")
        assert "UD_Italian-PUD" in url
        assert url.endswith("it_pud-ud-test.conllu")


class TestBuildFilename:
    def setup_method(self):
        from importlib import import_module

        self.mod = import_module("00_download_data")

    def test_ewt_train(self):
        spec = get_treebank_spec("ewt")
        assert self.mod.build_filename(spec, "train") == "en_ewt-ud-train.conllu"

    def test_isdt_test(self):
        spec = get_treebank_spec("isdt")
        assert self.mod.build_filename(spec, "test") == "it_isdt-ud-test.conllu"
