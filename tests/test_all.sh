#!/bin/bash

# Run from the repository root with the project's Python environment activated.
# Usage: bash tests/test_all.sh [--slow|--no-slow]
# Examples:
#   bash tests/test_all.sh             # All tests, including slow (pytest -q).
#   bash tests/test_all.sh --slow      # Same as the default.
#   bash tests/test_all.sh --no-slow   # Exclude tests marked slow.
# Slow tests may download models and require prepared local datasets.
# Returns pytest's exit status; this script does not run Ruff.

if [ "$#" -gt 1 ]; then
    echo "Usage from the repository root: bash tests/test_all.sh [--slow|--no-slow]" >&2
    exit 1
fi

case "${1:---slow}" in
    --slow)
        PYTHONPATH=src python -m pytest -q
        ;;
    --no-slow)
        PYTHONPATH=src python -m pytest -q -m "not slow"
        ;;
    *)
        echo "Usage from the repository root: bash tests/test_all.sh [--slow|--no-slow]" >&2
        exit 1
        ;;
esac
