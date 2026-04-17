#!/bin/bash
# Sequential experiment queue — runs each config+scenario combination in order.
# Ctrl+C will stop the current run and exit. Completed runs are safe.

set -e

cd "$(dirname "$0")"

SCRIPT_DIR="$(pwd)"

RUNS=15
DEFAULT_CONCURRENT=15
MODEL="${1:-}"  # optional: pass as first arg, e.g. ./run_queue.sh claude-opus-4-6

# Scenarios to run each config against
SCENARIOS=(
    acquisition_valuation
    detroit_auto_plant
    downsizing_company
    election_sentiment
    loan_profit
    painkiller_sales
    regional_hospital_costs
    sanctions_circumvention
    social_media_growth
    zero_interest_bnpl_marketing
)

# Format: "config_name", "config_name:max_concurrent", or "config_name:max_concurrent:runs"
# Omitted fields use RUNS and DEFAULT_CONCURRENT.
CONFIGS=(
    # --- Opus 4.6: Benign ---
    # benign_single_o46
    # benign_team_o46
    # benign_partner_o46
    # benign_dri_o46
    # benign_indepdri_o46
    # benign_safety_o46
    # benign_safety_indepdri_o46
    # benign_ownership_o46
    # benign_escalate_o46
    # benign_team_dri_o46
    # benign_team_indepdri_o46
    # benign_team_ownership_o46
    # --- Opus 4.6: Redteam ---
    # redteam_single_o46
    # redteam_team_o46
    # redteam_partner_o46
    # redteam_dri_o46
    # redteam_indepdri_o46
    # redteam_safety_o46
    # redteam_safety_indepdri_o46
    # redteam_ownership_o46
    # redteam_escalate_o46
    # redteam_team_dri_o46
    # redteam_team_indepdri_o46
    # redteam_team_ownership_o46
    # --- Opus 4.1: Benign ---
    benign_single_o41
    benign_team_o41
    benign_partner_o41
    benign_dri_o41
    benign_indepdri_o41
    benign_safety_o41
    benign_safety_indepdri_o41
    benign_ownership_o41
    benign_escalate_o41
    benign_team_dri_o41
    benign_team_indepdri_o41
    benign_team_ownership_o41
    # --- Opus 4.1: Redteam ---
    redteam_single_o41
    redteam_team_o41
    redteam_partner_o41
    redteam_dri_o41
    redteam_indepdri_o41
    redteam_safety_o41
    redteam_safety_indepdri_o41
    redteam_ownership_o41
    redteam_escalate_o41
    redteam_team_dri_o41
    redteam_team_indepdri_o41
    redteam_team_ownership_o41
    # --- Sonnet 4.6: Benign ---
    # benign_single_s46
    # benign_team_s46
    # benign_partner_s46
    # benign_dri_s46
    # benign_indepdri_s46
    # benign_safety_s46
    # benign_safety_indepdri_s46
    # benign_ownership_s46
    # benign_escalate_s46
    # benign_team_dri_s46
    # benign_team_indepdri_s46
    # benign_team_ownership_s46
    # --- Sonnet 4.6: Redteam ---
    # redteam_single_s46
    # redteam_team_s46
    # redteam_partner_s46
    # redteam_dri_s46
    # redteam_indepdri_s46
    # redteam_safety_s46
    # redteam_safety_indepdri_s46
    # redteam_ownership_s46
    # redteam_escalate_s46
    # redteam_team_dri_s46
    # redteam_team_indepdri_s46
    # redteam_team_ownership_s46
)

for entry in "${CONFIGS[@]}"; do
    IFS=':' read -r config concurrent runs <<< "$entry"
    concurrent="${concurrent:-$DEFAULT_CONCURRENT}"
    runs="${runs:-$RUNS}"

    for scenario in "${SCENARIOS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Starting: $config / $scenario ($runs runs, max-concurrent $concurrent)"
        echo "Time: $(date)"
        echo "=========================================="
        uv run python "$SCRIPT_DIR/run.py" --config "configs/${config}.yaml" --scenario "$scenario" --runs "$runs" --max-concurrent "$concurrent" --docker --api-key ${MODEL:+--model "$MODEL"}
        echo "Finished: $config / $scenario at $(date)"
    done
done

echo ""
echo "All queued experiments complete."
