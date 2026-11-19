#!/usr/bin/env bash
# =============================================================================
# CLEAN step entry point — produces D0.5 and D0.6
#
#   data/processed/eda_7043.parquet    50 cols, Joined dropped (EDA only)
#   data/processed/clean_7043.parquet  31 cols = ID + 29 features + target
#
# Idempotent: safe to re-run. Called by the Airflow DAG in W9.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Prefer the project venv when present
if [[ -x ".venv/bin/python" ]]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

echo "=============================================================="
echo "CLEAN STEP — IBM Telco EXTENDED"
echo "  repo   : $REPO_ROOT"
echo "  python : $PYTHON"
echo "=============================================================="

"$PYTHON" -m src.data.clean

echo
echo "Artifacts:"
ls -lh data/processed/*.parquet
