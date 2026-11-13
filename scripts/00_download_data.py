#!/usr/bin/env python3
"""Download Universal Dependencies treebanks to ``data/raw/{key}/``.

Usage
-----
    PYTHONPATH=src python scripts/00_download_data.py
    PYTHONPATH=src python scripts/00_download_data.py --treebank ewt pud_en
    PYTHONPATH=src python scripts/00_download_data.py --force
    PYTHONPATH=src python scripts/00_download_data.py --wikitext
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
from pathlib import Path

from lib.data import load_ud
from lib.registry import get_treebank_spec

try:
    from lib.paths import raw_dir
except ImportError:
    def raw_dir(treebank: str) -> Path:
        return Path(__file__).resolve().parents[1] / "data" / "raw" / treebank

DEFAULT_TREEBANKS: tuple[str, ...] = ("ewt", "isdt", "pud_en", "pud_it")

# ------------------------------------------------------------------ #
# URL helpers (testable without network)
# ------------------------------------------------------------------ #

def build_url(spec: dict, split: str) -> str:
    """Build the raw GitHub URL for a CoNLL-U file.

    Parameters
    ----------
    spec : dict
        Treebank specification from :func:`lib.registry.get_treebank_spec`.
    split : str
        Split name (``"train"``, ``"dev"`` or ``"test"``).

    Returns
    -------
    str
        Full URL to the raw file on GitHub at the pinned UD release tag.
    """
    repo = spec["repo"]
    if not repo.startswith("UniversalDependencies/"):
        repo = f"UniversalDependencies/{repo}"
    release = spec.get("ud_release", "r2.19")
    filename = spec["file_pattern"].format(split=split)
    return f"https://raw.githubusercontent.com/{repo}/{release}/{filename}"


def build_filename(spec: dict, split: str) -> str:
    """Return the local filename for a treebank split.

    Parameters
    ----------
    spec : dict
        Treebank specification.
    split : str
        Split name.

    Returns
    -------
    str
        The file name, e.g. ``"en_ewt-ud-train.conllu"``.
    """
    return spec["file_pattern"].format(split=split)


# ------------------------------------------------------------------ #
# Download logic
# ------------------------------------------------------------------ #

def download_treebank(key: str, *, force: bool = False) -> Path:
    """Download all splits of a treebank.

    Parameters
    ----------
    key : str
        Treebank registry key.
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Directory where files were saved.
    """
    spec = get_treebank_spec(key)
    out_dir = raw_dir(key)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in spec["splits"]:
        filename = build_filename(spec, split)
        dest = out_dir / filename

        if dest.exists() and not force:
            print(f"  [skip] {dest} (already exists)")
            continue

        url = build_url(spec, split)
        print(f"  [download] {url}")

        # Download to a temporary file then rename, so an interrupted
        # download never leaves a half-written file.
        try:
            with tempfile.NamedTemporaryFile(
                dir=out_dir, suffix=".tmp", delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.replace(dest)
        except Exception as exc:
            # Clean up the temp file on failure.
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"  [error] Failed to download {url}: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

        print(f"  [saved]    {dest}")

    return out_dir


def print_counts(key: str) -> None:
    """Load a downloaded treebank and print sentence/word counts per split."""
    spec = get_treebank_spec(key)
    out_dir = raw_dir(key)
    splits = tuple(spec["splits"])
    dataset = load_ud(out_dir, splits=splits)

    for split in splits:
        sentences = dataset[split]
        n_sents = len(sentences)
        n_words = sum(len(s["words"]) for s in sentences)
        print(f"    {key}/{split}: {n_sents} sentences, {n_words} words")


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Universal Dependencies treebanks."
    )
    parser.add_argument(
        "--treebank",
        nargs="*",
        default=None,
        help="Treebank keys to download (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist.",
    )
    parser.add_argument(
        "--wikitext",
        action="store_true",
        help="Also cache the WikiText-103 dataset via Hugging Face.",
    )
    args = parser.parse_args()

    keys = args.treebank if args.treebank else DEFAULT_TREEBANKS

    for key in keys:
        print(f"[{key}]")
        download_treebank(key, force=args.force)
        print_counts(key)

    if args.wikitext:
        print("[wikitext]")
        from datasets import load_dataset

        load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
        print("  WikiText-103 cached.")

    print("Done.")


if __name__ == "__main__":
    main()
