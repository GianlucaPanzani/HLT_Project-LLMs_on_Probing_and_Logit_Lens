"""Unit tests for ws4_lens.upos_bridge (Issue L6)."""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from ws4_lens.upos_bridge import (  # noqa: E402
    _is_number,
    _is_punctuation,
    build_upos_lexicon,
    token_to_upos,
)


@pytest.fixture
def toy_train_table() -> pd.DataFrame:
    """Create a small toy word table for deterministic unit testing."""
    return pd.DataFrame({
        "form": [
            "the", "The", "cat", "sat", "on", "the", "mat",
            ",", "1999", "tie_word", "tie_word", "May", "may", "may"
        ],
        "upos": [
            "DET", "DET", "NOUN", "VERB", "ADP", "DET", "NOUN",
            "PUNCT", "NUM", "NOUN", "ADJ", "PROPN", "AUX", "AUX"
        ],
    })


# --------------------------------------------------------------------------- #
# Acceptance Criteria Tests (from Issue #24)
# --------------------------------------------------------------------------- #

def test_acceptance_criteria_the(toy_train_table):
    """Verify ' the' maps to DET with an EWT-like table."""
    lexicon = build_upos_lexicon(toy_train_table)
    assert token_to_upos(" the", lexicon) == "DET"


def test_acceptance_criteria_ing(toy_train_table):
    """Verify 'ing' (no leading space, subword continuation) maps to X."""
    lexicon = build_upos_lexicon(toy_train_table)
    assert token_to_upos("ing", lexicon) == "X"


def test_acceptance_criteria_punct(toy_train_table):
    """Verify ' ,' maps to PUNCT."""
    lexicon = build_upos_lexicon(toy_train_table)
    assert token_to_upos(" ,", lexicon) == "PUNCT"


def test_acceptance_criteria_num(toy_train_table):
    """Verify ' 1999' maps to NUM."""
    lexicon = build_upos_lexicon(toy_train_table)
    assert token_to_upos(" 1999", lexicon) == "NUM"


def test_acceptance_criteria_tie_breaking(toy_train_table):
    """Verify tie-breaking is deterministic by UPOS_TAGS order.

    'tie_word' has 1 NOUN and 1 ADJ.
    In UPOS_TAGS: ('ADJ', 'ADP', ..., 'NOUN', ...), ADJ has rank 0 and NOUN has rank 7.
    Therefore, ADJ must win.
    """
    lexicon = build_upos_lexicon(toy_train_table)
    assert lexicon["tie_word"] == "ADJ"
    assert token_to_upos(" tie_word", lexicon) == "ADJ"


# --------------------------------------------------------------------------- #
# Helper Unit Tests (_is_punctuation, _is_number)
# --------------------------------------------------------------------------- #

def test_is_punctuation():
    """Verify punctuation detection on ASCII and Unicode characters."""
    assert _is_punctuation(",")
    assert _is_punctuation(".")
    assert _is_punctuation("--")
    assert _is_punctuation("...")
    assert _is_punctuation(";")
    assert _is_punctuation("«")
    assert _is_punctuation("»")
    assert _is_punctuation("“")
    assert _is_punctuation("”")
    assert _is_punctuation("?")
    assert _is_punctuation("!")

    # Non-punctuation
    assert not _is_punctuation("")
    assert not _is_punctuation("word")
    assert not _is_punctuation("123")
    assert not _is_punctuation("word,")
    assert not _is_punctuation(",word")


def test_is_number():
    """Verify number detection on digits with optional '.' or ','."""
    assert _is_number("1999")
    assert _is_number("0")
    assert _is_number("3.14")
    assert _is_number("1,000")
    assert _is_number("12,345.67")
    assert _is_number(".5")
    assert _is_number("10.")

    # Non-numbers
    assert not _is_number("")
    assert not _is_number(".")  # no digits
    assert not _is_number(",")  # no digits
    assert not _is_number(".,.")  # no digits
    assert not _is_number("1990s")
    assert not _is_number("3rd")
    assert not _is_number("num")


# --------------------------------------------------------------------------- #
# Lowercase Fallback and Case Sensitivity Tests
# --------------------------------------------------------------------------- #

def test_case_sensitivity_and_fallback():
    """Verify exact case takes precedence, and lowercase acts as fallback."""
    df = pd.DataFrame({
        "form": ["May", "may", "may", "London"],
        "upos": ["PROPN", "AUX", "AUX", "PROPN"],
    })
    lexicon = build_upos_lexicon(df)

    # Exact lookups
    assert token_to_upos(" May", lexicon) == "PROPN"
    assert token_to_upos(" may", lexicon) == "AUX"

    # Lowercase lookup fallback for MAY (falls back to 'may' -> AUX)
    assert token_to_upos(" MAY", lexicon) == "AUX"

    # 'London' was only seen capitalized; 'london' added as lowercase fallback
    assert token_to_upos(" London", lexicon) == "PROPN"
    assert token_to_upos(" london", lexicon) == "PROPN"
    assert token_to_upos(" LONDON", lexicon) == "PROPN"


# --------------------------------------------------------------------------- #
# Edge Cases and Error Handling
# --------------------------------------------------------------------------- #

def test_unseen_tokens_and_fallbacks(toy_train_table):
    """Verify unseen tokens correctly route to PUNCT, NUM, or X."""
    lexicon = build_upos_lexicon(toy_train_table)

    # Unseen punctuation
    assert token_to_upos(" ;", lexicon) == "PUNCT"
    assert token_to_upos(" ---", lexicon) == "PUNCT"
    assert token_to_upos(" ¿", lexicon) == "PUNCT"

    # Unseen numbers
    assert token_to_upos(" 2026", lexicon) == "NUM"
    assert token_to_upos(" 42.5", lexicon) == "NUM"

    # Unseen words
    assert token_to_upos(" xylophone", lexicon) == "X"
    assert token_to_upos(" Quidditch", lexicon) == "X"

    # Whitespace-only or empty tokens
    assert token_to_upos("", lexicon) == "X"
    assert token_to_upos(" ", lexicon) == "X"
    assert token_to_upos("   ", lexicon) == "X"


def test_build_upos_lexicon_missing_columns():
    """Verify ValueError is raised if 'form' or 'upos' is missing."""
    with pytest.raises(ValueError, match="train_table must contain 'form' and 'upos'"):
        build_upos_lexicon(pd.DataFrame({"word": ["the"], "pos": ["DET"]}))

    with pytest.raises(ValueError, match="train_table must contain 'form' and 'upos'"):
        build_upos_lexicon(pd.DataFrame({"form": ["the"]}))


def test_build_upos_lexicon_drops_nan_rows():
    """Verify NaN rows are dropped gracefully without error."""
    df = pd.DataFrame({
        "form": ["the", None, "dog", float("nan")],
        "upos": ["DET", "NOUN", "NOUN", "VERB"],
    })
    lexicon = build_upos_lexicon(df)
    assert "the" in lexicon
    assert "dog" in lexicon
    assert len(lexicon) >= 2
