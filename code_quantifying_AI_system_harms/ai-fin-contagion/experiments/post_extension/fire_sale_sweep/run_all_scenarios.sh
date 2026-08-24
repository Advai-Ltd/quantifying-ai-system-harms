#!/usr/bin/env bash
# Run fire_sale_sweep/run.py for every CSV in a folder.
#
# Usage:
#   ./run_all_scenarios.sh [FOLDER]
#
# FOLDER defaults to:
#   experiments/post_extension/fire_sale_sweep/firesale_ai_component_table/updated_escalation_mapping_firesale_sceanarios
#
# All other run.py flags can be edited in the ARGS block below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

FOLDER="${1:-$SCRIPT_DIR/firesale_ai_component_table/updated_escalation_mapping_firesale_sceanarios}"
# Resolve relative paths against cwd (where the user invoked the script from)
[[ "$FOLDER" != /* ]] && FOLDER="$(pwd)/$FOLDER"

# ── run.py arguments (edit here) ─────────────────────────────────────────────
MIN_SHOCK="-0.05"
MAX_SHOCK="-0.25"
NUM_RUNS=30
SKIP_MARKET_STRUCTURE="skewed oligopoly"
# ─────────────────────────────────────────────────────────────────────────────

shopt -s nullglob
csv_files=("$FOLDER"/*.csv)

if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "ERROR: No CSV files found in: $FOLDER"
    exit 1
fi

echo "========================================"
echo "Fire Sale Sweep — batch run"
echo "Folder : $FOLDER"
echo "Files  : ${#csv_files[@]}"
echo "========================================"

for csv in "${csv_files[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running: $(basename "$csv")"
    echo "----------------------------------------"
    uv run python3 "$SCRIPT_DIR/run.py" \
        --scenario-table "$csv" \
        --min-shock "$MIN_SHOCK" \
        --max-shock "$MAX_SHOCK" \
        --no-run-output \
        --num-runs "$NUM_RUNS" \
        --skip-market-structure "$SKIP_MARKET_STRUCTURE" \
        --yes
done

echo ""
echo "========================================"
echo "All scenarios complete."
echo "========================================"
