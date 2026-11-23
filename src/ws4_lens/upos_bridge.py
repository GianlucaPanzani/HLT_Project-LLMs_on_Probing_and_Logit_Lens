"""UPOS Bridge for decoder model lenses (Issue L6).

This module implements the mapping from top-1 predicted vocabulary tokens to
Universal Dependencies Part-of-Speech (UPOS) tags (PROPOSAL.md WS4 Point 5;
CONTRACTS.md Section 10). It enables testing whether a causal decoder model
commits to the correct grammatical category of the next word before converging
on the exact surface form across intermediate layers.
"""

from __future__ import annotations

import string
import unicodedata

import pandas as pd

from lib.data import UPOS_TAGS


def _is_punctuation(s: str) -> bool:
    """Check if a string consists solely of punctuation characters.

    Parameters
    ----------
    s : str
        Input string without leading/trailing whitespace.

    Returns
    -------
    bool
        True if non-empty and every character is Unicode punctuation or ASCII punctuation.
    """
    if not s:
        return False
    return all(unicodedata.category(c).startswith("P") or c in string.punctuation for c in s)


def _is_number(s: str) -> bool:
    """Check if a string consists solely of digits with optional '.' or ',' separators.

    Parameters
    ----------
    s : str
        Input string without leading/trailing whitespace.

    Returns
    -------
    bool
        True if string contains at least one digit and only digits, dots, or commas.
    """
    if not s:
        return False
    has_digit = False
    for c in s:
        if c.isdigit():
            has_digit = True
        elif c in (".", ","):
            continue
        else:
            return False
    return has_digit


def build_upos_lexicon(train_table: pd.DataFrame) -> dict[str, str]:
    """Build a word-form to most-frequent UPOS tag mapping from a training word table.

    For each distinct surface word form, the most frequent UPOS tag is selected.
    Ties are broken deterministically by the official order in `lib.data.UPOS_TAGS`.
    In addition, lowercase forms are computed and added as fallback entries for
    words that were not observed in lowercase in the training split.

    Parameters
    ----------
    train_table : pd.DataFrame
        Training word table containing at least `form` and `upos` columns.

    Returns
    -------
    dict[str, str]
        Dictionary mapping word forms (and fallback lowercase forms) to their
        most frequent UPOS tag.

    Raises
    ------
    ValueError
        If `form` or `upos` columns are missing from `train_table`.
    """
    if "form" not in train_table.columns or "upos" not in train_table.columns:
        raise ValueError("train_table must contain 'form' and 'upos' columns.")

    tag_rank = {tag: i for i, tag in enumerate(UPOS_TAGS)}
    valid = train_table.dropna(subset=["form", "upos"])

    # 1. Count occurrences of each (form, upos) and (lowercase_form, upos)
    exact_counts: dict[str, dict[str, int]] = {}
    lower_counts: dict[str, dict[str, int]] = {}

    for form, upos in zip(valid["form"], valid["upos"]):
        form_str = str(form)
        upos_str = str(upos)

        # Exact form counts
        c_map = exact_counts.setdefault(form_str, {})
        c_map[upos_str] = c_map.get(upos_str, 0) + 1

        # Lowercase form counts
        low_str = form_str.lower()
        l_map = lower_counts.setdefault(low_str, {})
        l_map[upos_str] = l_map.get(upos_str, 0) + 1

    def pick_best_tag(counts: dict[str, int]) -> str:
        # Sort key: highest count first (-counts[t]), then earliest UPOS_TAGS rank (tag_rank[t])
        return min(counts.keys(), key=lambda t: (-counts[t], tag_rank.get(t, 999)))

    lexicon: dict[str, str] = {}
    for form_str, counts in exact_counts.items():
        lexicon[form_str] = pick_best_tag(counts)

    # 2. Add lowercase fallback keys for any lower_str not already present
    for low_str, counts in lower_counts.items():
        if low_str not in lexicon:
            lexicon[low_str] = pick_best_tag(counts)

    return lexicon


def token_to_upos(token: str, lexicon: dict[str, str]) -> str:
    """Map a decoded token string to its most likely UPOS tag.

    Follows the decision rules specified in Issue #24 and CONTRACTS.md Section 10:
    1. If the token does not start with a leading space, it continues the current
       subword/word rather than starting a new word, so return `'X'`.
    2. Otherwise, strip the space. If empty, return `'X'`.
    3. Perform exact lookup in `lexicon`. If found, return the mapped UPOS.
    4. Perform lowercase lookup in `lexicon`. If found, return the mapped UPOS.
    5. If the token consists solely of punctuation characters, return `'PUNCT'`.
    6. If the token consists solely of digits (with optional '.' or ','), return `'NUM'`.
    7. Otherwise, return `'X'`.

    Parameters
    ----------
    token : str
        Decoded token string (e.g. from `tokenizer.decode([token_id])`).
    lexicon : dict[str, str]
        Word-form to UPOS lexicon from `build_upos_lexicon`.

    Returns
    -------
    str
        Predicted UPOS tag (one of `lib.data.UPOS_TAGS`), or `'X'` if subword / unknown.
    """
    # Rule 1: No leading space -> token continues the current word -> return X
    if not token.startswith(" "):
        return "X"

    # Rule 2: Strip space
    word = token.strip()
    if not word:
        return "X"

    # Rule 3: Exact lookup
    if word in lexicon:
        return lexicon[word]

    # Rule 4: Lowercase lookup
    word_lower = word.lower()
    if word_lower in lexicon:
        return lexicon[word_lower]

    # Rule 5: Only punctuation characters -> PUNCT
    if _is_punctuation(word):
        return "PUNCT"

    # Rule 6: Only digits (with . or ,) -> NUM
    if _is_number(word):
        return "NUM"

    # Rule 7: Else -> X
    return "X"
