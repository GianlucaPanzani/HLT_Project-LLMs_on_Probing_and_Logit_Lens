#!/bin/bash

# Run from the repository root with the project's Python environment activated.
# Usage: bash tests/test.sh <test_file>
# Example:
#   bash tests/test.sh tests/test_probes.py
# Runs pytest on the given file and Ruff on src, tests, scripts and app.
# Ruff runs even if pytest fails.

if [ "$#" -ne 1 ]; then
    echo "Usage from the repository root: bash tests/test.sh tests/<testfile.py>" >&2
    exit 1
fi

if [ ! -f "$1" ]; then
    echo "Test file not found: $1" >&2
    exit 1
fi

# Run both checks, even if pytest fails
PYTHONPATH=src python -m pytest -q "$1"
ruff check src tests scripts app

