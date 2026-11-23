"""Download WS4 lens checkpoints and evaluation results from Hugging Face.

Examples:
    PYTHONPATH=src python scripts/15_pull_ws4_lens.py
    PYTHONPATH=src python scripts/15_pull_ws4_lens.py --checkpoints-only

Reads the private dataset ``hlt-group11/ws4-lens`` (see its README) and writes
``checkpoints/`` and ``results/lens/`` at the repository root, matching
``docs/CONTRACTS.md`` sections 9.3–9.5. Authentication uses the saved Hugging
Face login or ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from lib.paths import ROOT

HF_REPO_ID = "hlt-group11/ws4-lens"


def main(argv: list[str] | None = None) -> None:
    """Download checkpoints and optional lens result parquets."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--checkpoints-only",
        action="store_true",
        help="Download only checkpoints/tuned_lens and checkpoints/jacobian_lens.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Dataset revision (branch, tag, or commit). Default: main.",
    )
    args = parser.parse_args(argv)

    patterns = ["checkpoints/**"]
    if not args.checkpoints_only:
        patterns.append("results/lens/**")

    print(f"Downloading from {HF_REPO_ID}@{args.revision} ...", flush=True)
    path = snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(ROOT),
        allow_patterns=patterns,
    )
    print(f"Done. Files are under {ROOT} ({path})")


if __name__ == "__main__":
    main()