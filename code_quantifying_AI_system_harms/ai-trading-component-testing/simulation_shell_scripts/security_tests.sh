#!/bin/bash

# security_tests.sh
# Loops over all attack types and target assets, running main.py for each attack config.

ATTACK_CONFIG_DIR="simulation_inputs/adversarial_attack_configs"
NEWS_DIR="simulation_inputs/input_article_data/serious_run/2025_12_01-12_25_31"
OUTPUT_BASE="simulation_outputs/llm_response_data/DUMMY_2_11_02_instruction_attack_on_bearish_neutral"
RESULTS_FILENAME="results.csv" 
EXPERIMENT_TYPE="fixed-bearish-neutral"
NUM_EPISODES=30
BATCH_MODE="--batch-mode"
BATCH_SIZE=10
INFERENCE_MODELS="gemini/gemini-flash-lite-latest"

# Define your attack types and target assets explicitly
ATTACK_TYPES=("system_instruction" "alignment_association" "model_address" "appeal_to_authority")
TARGET_ASSETS=("Mortgages_Backed_Securities" "Corporate_Bonds" "Equities" "Government_Bonds")

for ASSET in "${TARGET_ASSETS[@]}"; do
    for ATTACK_TYPE in "${ATTACK_TYPES[@]}"; do
        # Construct the expected filename
        # Example: asset_Mortgages_Backed_Securities_(MBS)--attack_alignment_association--insertion_by_asset.json
        ATTACK_CONFIG="$ATTACK_CONFIG_DIR/asset_${ASSET}--attack_${ATTACK_TYPE}--insertion_by_asset.json"

        if [[ -f "$ATTACK_CONFIG" ]]; then
            BASENAME=$(basename "$ATTACK_CONFIG")
            ASSET_DISPLAY=$(echo "$ASSET" | sed 's/_/ /g' | sed 's/\\//g')
            echo "Asset Target, $ASSET ; Asset display, $ASSET_DISPLAY"


            echo "Running attack config: $ATTACK_CONFIG"
            echo "Target asset: $ASSET_DISPLAY"
            echo "Output dir: $OUTPUT_BASE"

            python3 src/main.py \
                "$EXPERIMENT_TYPE" \
                "$NEWS_DIR" \
                --inference-models "$INFERENCE_MODELS" \
                --fixed-assets "$ASSET_DISPLAY" \
                --num-episodes "$NUM_EPISODES" \
                $BATCH_MODE \
                --batch-size "$BATCH_SIZE" \
                --attack-config "$ATTACK_CONFIG" \
                --output-dir "$OUTPUT_BASE" \
                --mitigations "$MITIGATIONS" \
                --results-file-basename "$RESULTS_FILENAME" \

        else
            echo "Config file not found: $ATTACK_CONFIG"
        fi

        sleep 10
    done
done
