#!/bin/bash

RESPONSES_CSV="simulation_outputs/llm_response_data/main_run/extension/combined_neutral_bearish_results.csv"
ADVERSARIAL_CSV="simulation_outputs/llm_response_data/main_run/extension/04_03_neutral_bearish_adversarial/updated_combined_adversarial_attack_results.csv"
ATTACK_TYPE="escalation"

# Firesale scaler (kappa)
SCALER_MIN=0.5
SCALER_MAX=8.1
SCALER_STEP=0.25

# Human firesale term (f_H)
FH_MIN=0.01
FH_MAX=0.3
FH_STEP=0.025

folder_name=$(date +"%Y_%m_%d_%H-%M-%S_firesale_scenarios")

mkdir $folder_name
touch "$folder_name/run_details.txt"
echo "RESPONSES_CSV: $RESPONSES_CSV">> "$folder_name/run_details.txt"
echo "ADVERSARIAL_CSV $ADVERSARIAL_CSV" >> "$folder_name/run_details.txt"
echo "ATTACK_TYPE $ATTACK_TYPE" >> "$folder_name/run_details.txt"
echo "FH_MIN $FH_MIN ; FH_MAX $FH_MAX ; FH_STEP $FH_STEP" >> "$folder_name/run_details.txt"
echo "SCALER_MIN $SCALER_MIN ; SCALER_MAX $SCALER_MAX ; SCALER_STEP $SCALER_STEP" >> "$folder_name/run_details.txt"


for scaler in $(seq $SCALER_MIN $SCALER_STEP $SCALER_MAX); do
    for fh in $(seq $FH_MIN $FH_STEP $FH_MAX); do
        echo "Running: scaler=${scaler}, human_firesale_term=${fh}"
        python3 src/analysis/stats_mapping_analysis.py \
            "$RESPONSES_CSV" \
            --adversarial-responses-csv "$ADVERSARIAL_CSV" \
            --firesale-mapping \
            --adversarial-attack-type "$ATTACK_TYPE" \
            --firesale-scaler "$scaler" \
            --human-firesale-term "$fh" \
            --output-dir "$folder_name"
    done
done
