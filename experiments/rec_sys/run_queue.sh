#!/bin/bash
# Sequential experiment queue — runs each config in order.
# Ctrl+C will stop the current run and exit. Completed runs are safe.

set -e

cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"

RUNS=30
DEFAULT_CONCURRENT=5
MODEL="${1:-}"  # optional: pass as first arg, e.g. ./run_queue.sh claude-opus-4-6

# Format: "config_name", "config_name:max_concurrent", or "config_name:max_concurrent:runs"
# Omitted fields use RUNS and DEFAULT_CONCURRENT.
CONFIGS=(
    # --- Opus 4.6 ---
    single_base_o46:8
    multi8_base_o46
    single_ownership_o46:8
    multi8_ownership_o46
    multi8_ownershipmedium_o46
    multi8_ownershipheavy_o46
    multi8_escalate_o46
    multi8_escalatemedium_o46
    multi8_escalateheavy_o46
    multi8_dri_o46
    multi8_safetydoc_o46
    multi8_safetydocdri_o46
    multi8_comms_o46
    multi8_reviewpeer_o46
    multi8_safetyreviewpeer_o46
    multi8_reviewpm_o46
    multi8_safetyreviewpm_o46
    multi8_reviewindep_o46
    multi8_safetyreviewindep_o46
    # --- Opus 4.1 ---
    # single_base_o41:8
    # multi8_base_o41
    # single_ownership_o41:8
    # multi8_ownership_o41
    # multi8_ownershipmedium_o41
    # multi8_ownershipheavy_o41
    # multi8_escalate_o41
    # multi8_escalatemedium_o41
    # multi8_escalateheavy_o41
    # multi8_dri_o41
    # multi8_safetydoc_o41
    # multi8_safetydocdri_o41
    # multi8_comms_o41
    # multi8_reviewpeer_o41
    # multi8_safetyreviewpeer_o41
    # multi8_reviewpm_o41
    # multi8_safetyreviewpm_o41
    # multi8_reviewindep_o41
    # multi8_safetyreviewindep_o41
    # --- Sonnet 4.6 ---
    # single_base_s46:8
    # multi8_base_s46
    # single_ownership_s46:8
    # multi8_ownership_s46
    # multi8_ownershipmedium_s46
    # multi8_ownershipheavy_s46
    # multi8_escalate_s46
    # multi8_escalatemedium_s46
    # multi8_escalateheavy_s46
    # multi8_dri_s46
    # multi8_safetydoc_s46
    # multi8_safetydocdri_s46
    # multi8_comms_s46
    # multi8_reviewpeer_s46
    # multi8_safetyreviewpeer_s46
    # multi8_reviewpm_s46
    # multi8_safetyreviewpm_s46
    # multi8_reviewindep_s46
    # multi8_safetyreviewindep_s46
)

for entry in "${CONFIGS[@]}"; do
    IFS=':' read -r config concurrent runs <<< "$entry"
    concurrent="${concurrent:-$DEFAULT_CONCURRENT}"
    runs="${runs:-$RUNS}"
    echo ""
    echo "=========================================="
    echo "Starting: $config ($runs runs, max-concurrent $concurrent)"
    echo "Time: $(date)"
    echo "=========================================="
    uv run python "$SCRIPT_DIR/run.py" --config "configs/${config}.yaml" --runs "$runs" --max-concurrent "$concurrent" --docker --api-key ${MODEL:+--model "$MODEL"}
    echo "Finished: $config at $(date)"
done

echo ""
echo "All queued experiments complete."
