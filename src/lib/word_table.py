"""Word-table construction, I/O and utilities.

A *word table* is a pandas DataFrame with one row per UD syntactic word,
aligned by row order with every feature file.  See CONTRACTS.md section 2
for the exact schema.
"""

from __future__ import annotations

import pandas as pd

from lib.data import UPOS_TAGS
from lib.paths import word_table_path

_UPOS_SET: frozenset[str] = frozenset(UPOS_TAGS)


def build_word_table(sentences: list[dict], lang: str) -> pd.DataFrame:
    """Build a word table from one split of ``load_ud`` output.

    Parameters
    ----------
    sentences : list[dict]
        Records returned by :func:`lib.data.load_ud` for a single split.
        Each record must contain ``sent_id`` (str), ``words`` (list[str])
        and ``upos`` (list[str]).
    lang : str
        Language code (``"en"`` or ``"it"``).

    Returns
    -------
    pd.DataFrame
        Columns: ``sent_id``, ``word_idx``, ``form``, ``upos``,
        ``next_upos``, ``lang``.  Row order matches sentence and word
        order in the input.

    Raises
    ------
    ValueError
        If any UPOS tag is not one of the 17 canonical tags.
    """
    rows: list[dict] = []
    for sentence in sentences:
        sent_id: str = sentence["sent_id"]
        words: list[str] = sentence["words"]
        upos_tags: list[str] = sentence["upos"]

        n_words = len(words)
        for i, (form, upos) in enumerate(zip(words, upos_tags)):
            if upos not in _UPOS_SET:
                raise ValueError(
                    f"Invalid UPOS tag {upos!r} for word {form!r} "
                    f"in sentence {sent_id!r}. "
                    f"Valid tags: {', '.join(sorted(_UPOS_SET))}"
                )
            next_upos: str | None = upos_tags[i + 1] if i + 1 < n_words else None
            rows.append(
                {
                    "sent_id": sent_id,
                    "word_idx": i,
                    "form": form,
                    "upos": upos,
                    "next_upos": next_upos,
                    "lang": lang,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        # Return an empty frame with the correct columns and dtypes.
        df = pd.DataFrame(
            columns=["sent_id", "word_idx", "form", "upos", "next_upos", "lang"]
        )

    # Enforce dtypes per CONTRACTS.md section 2.
    df["sent_id"] = df["sent_id"].astype("string")
    df["word_idx"] = df["word_idx"].astype("int32")
    df["form"] = df["form"].astype("string")
    df["upos"] = df["upos"].astype("string")
    df["next_upos"] = df["next_upos"].astype("string")  # pd.NA for None
    df["lang"] = df["lang"].astype("string")

    return df


def load_word_table(treebank: str, split: str) -> pd.DataFrame:
    """Read a word table from its parquet file.

    Parameters
    ----------
    treebank : str
        Treebank registry key.
    split : str
        Split name.

    Returns
    -------
    pd.DataFrame
        The word table with the schema of :func:`build_word_table`.

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist (with a hint to run the build
        script).
    """
    path = word_table_path(treebank, split)
    if not path.exists():
        raise FileNotFoundError(
            f"Word table not found: {path}\n"
            f"Run: PYTHONPATH=src python scripts/02_build_word_tables.py"
        )
    return pd.read_parquet(path)


def table_to_sentences(table: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Convert a word table back to a list of ``(sent_id, words)`` pairs.

    The output preserves the row order of the table so that concatenating
    all word lists reproduces the ``form`` column.

    Parameters
    ----------
    table : pd.DataFrame
        A word table produced by :func:`build_word_table`.

    Returns
    -------
    list[tuple[str, list[str]]]
        Each element is ``(sent_id, [word_form, ...])``.
    """
    result: list[tuple[str, list[str]]] = []
    for sent_id, group in table.groupby("sent_id", sort=False):
        words = group["form"].tolist()
        result.append((str(sent_id), words))
    return result
