"""Tests for lib.word_table – no network, no torch.

Uses hand-written sentences to verify column schema, dtypes, next_upos logic,
round-trip via table_to_sentences, and validation of invalid UPOS tags.
"""

import pandas as pd
import pytest

from lib.word_table import build_word_table, table_to_sentences

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

SENTENCES = [
    {
        "sent_id": "s1",
        "words": ["The", "cat", "sat"],
        "upos": ["DET", "NOUN", "VERB"],
    },
    {
        "sent_id": "s2",
        "words": ["I", "run"],
        "upos": ["PRON", "VERB"],
    },
]


@pytest.fixture()
def table():
    return build_word_table(SENTENCES, lang="en")


# ------------------------------------------------------------------ #
# Schema and dtype tests
# ------------------------------------------------------------------ #

class TestSchema:
    def test_columns(self, table):
        expected = ["sent_id", "word_idx", "form", "upos", "next_upos", "lang"]
        assert list(table.columns) == expected

    def test_row_count(self, table):
        # 3 words + 2 words = 5 rows
        assert len(table) == 5

    def test_dtypes(self, table):
        assert table["sent_id"].dtype == "string"
        assert table["word_idx"].dtype == "int32"
        assert table["form"].dtype == "string"
        assert table["upos"].dtype == "string"
        assert table["next_upos"].dtype == "string"
        assert table["lang"].dtype == "string"

    def test_lang_value(self, table):
        assert (table["lang"] == "en").all()


# ------------------------------------------------------------------ #
# next_upos logic
# ------------------------------------------------------------------ #

class TestNextUpos:
    def test_first_word_of_s1(self, table):
        # "The" -> next is "NOUN"
        row = table.iloc[0]
        assert row["next_upos"] == "NOUN"

    def test_second_word_of_s1(self, table):
        # "cat" -> next is "VERB"
        row = table.iloc[1]
        assert row["next_upos"] == "VERB"

    def test_last_word_of_s1_is_null(self, table):
        # "sat" -> last in sentence, next_upos is null
        row = table.iloc[2]
        assert pd.isna(row["next_upos"])

    def test_first_word_of_s2(self, table):
        # "I" -> next is "VERB"
        row = table.iloc[3]
        assert row["next_upos"] == "VERB"

    def test_last_word_of_s2_is_null(self, table):
        # "run" -> last in sentence, next_upos is null
        row = table.iloc[4]
        assert pd.isna(row["next_upos"])

    def test_null_count_equals_sentence_count(self, table):
        n_null = int(table["next_upos"].isna().sum())
        n_sents = table["sent_id"].nunique()
        assert n_null == n_sents == 2


# ------------------------------------------------------------------ #
# word_idx
# ------------------------------------------------------------------ #

class TestWordIdx:
    def test_zero_based(self, table):
        s1 = table[table["sent_id"] == "s1"]
        assert list(s1["word_idx"]) == [0, 1, 2]

    def test_resets_per_sentence(self, table):
        s2 = table[table["sent_id"] == "s2"]
        assert list(s2["word_idx"]) == [0, 1]


# ------------------------------------------------------------------ #
# Round-trip
# ------------------------------------------------------------------ #

class TestTableToSentences:
    def test_round_trip(self, table):
        result = table_to_sentences(table)
        assert len(result) == 2
        assert result[0] == ("s1", ["The", "cat", "sat"])
        assert result[1] == ("s2", ["I", "run"])

    def test_forms_match(self, table):
        result = table_to_sentences(table)
        all_forms = [w for _, words in result for w in words]
        assert all_forms == list(table["form"])


# ------------------------------------------------------------------ #
# Validation
# ------------------------------------------------------------------ #

class TestValidation:
    def test_invalid_upos_raises(self):
        bad = [
            {
                "sent_id": "s_bad",
                "words": ["oops"],
                "upos": ["INVALID_TAG"],
            },
        ]
        with pytest.raises(ValueError, match="Invalid UPOS tag"):
            build_word_table(bad, lang="en")

    def test_empty_input(self):
        table = build_word_table([], lang="en")
        assert len(table) == 0
        assert list(table.columns) == [
            "sent_id", "word_idx", "form", "upos", "next_upos", "lang",
        ]


# ------------------------------------------------------------------ #
# Case preservation
# ------------------------------------------------------------------ #

class TestCasePreservation:
    def test_forms_not_lowered(self):
        sents = [
            {"sent_id": "s_case", "words": ["The", "Cat"], "upos": ["DET", "NOUN"]},
        ]
        table = build_word_table(sents, lang="en")
        assert list(table["form"]) == ["The", "Cat"]
