#!/bin/bash

# comprehensive_experiment_run.sh
# Runs all-neutral, fixed-bearish-neutral, and escalation attack experiments
# Only on GPT-5, for all 4 assets, 30 runs each

# Configuration
NEWS_DIR="simulation_inputs/input_article_data/serious_run/2025_12_01-12_25_31"
OUTPUT_BASE="simulation_outputs/llm_response_data/comprehensive_run_2026_03_05-14_45_53"
RESULTS_FILENAME="prompt_hardened_all_neutral_bearish_adversarial.csv"
NUM_EPISODES=30
BATCH_MODE="--batch-mode"
BATCH_SIZE=10
INFERENCE_MODELS="openai/gpt-5-mini"
ATTACK_CONFIG_DIR="simulation_inputs/adversarial_attack_configs"
# ATTACK_TYPES=("escalation" "system_instruction" "alignment_association" "model_address" "appeal_to_authority")
ATTACK_TYPES=("system_instruction" "alignment_association" "model_address" "appeal_to_authority")
# Define target assets
TARGET_ASSETS=("Mortgages Backed Securities" "Corporate Bonds" "Equities" "Government Bonds")
TARGET_ASSETS_FILES=("Mortgages_Backed_Securities" "Corporate_Bonds" "Equities" "Government_Bonds")


# TARGET_ASSETS=("Mortgages Backed Securities")
# TARGET_ASSETS_FILES=("Mortgages_Backed_Securities")

# echo "Starting comprehensive experiment run..."
# echo "Output directory: $OUTPUT_BASE"
# echo "Number of episodes per experiment: $NUM_EPISODES"
# echo "Model: $INFERENCE_MODELS"
# echo "=================================================="

# # Create output directory
# mkdir -p "$OUTPUT_BASE"

# #########################
# # 1. ALL NEUTRAL EXPERIMENT
# #########################
# echo ""
# echo "1. Running ALL-NEUTRAL experiment..."
# echo "=================================================="

# python3 src/main.py \
#     "all-neutral" \
#     "$NEWS_DIR" \
#     --inference-models "$INFERENCE_MODELS" \
#     --num-episodes "$NUM_EPISODES" \
#     $BATCH_MODE \
#     --batch-size "$BATCH_SIZE" \
#     --output-dir "$OUTPUT_BASE" \
#     --mitigations "prompt-hardening" \
#     --results-file-basename "$RESULTS_FILENAME" \
#     --experiment-tag "all-neutral"

# echo "All-neutral experiment completed."
# sleep 10

# #########################
# # 2. FIXED-BEARISH-NEUTRAL EXPERIMENTS
# #########################
# echo ""
# echo "2. Running FIXED-BEARISH-NEUTRAL experiments for each asset..."
# echo "=================================================="

# for i in "${!TARGET_ASSETS[@]}"; do
#     ASSET="${TARGET_ASSETS[$i]}"
#     echo ""
#     echo "Running fixed-bearish-neutral for asset: $ASSET"
    
#     python3 src/main.py \
#         "fixed-bearish-neutral" \
#         "$NEWS_DIR" \
#         --inference-models "$INFERENCE_MODELS" \
#         --fixed-assets "$ASSET" \
#         --num-episodes "$NUM_EPISODES" \
#         $BATCH_MODE \
#         --batch-size "$BATCH_SIZE" \
#         --output-dir "$OUTPUT_BASE" \
#         --mitigations "prompt-hardening" \
#         --results-file-basename "$RESULTS_FILENAME" \
#         --experiment-tag "fixed-bearish-neutral-$(echo $ASSET | sed 's/ /-/g')"
    
#     echo "Completed fixed-bearish-neutral for $ASSET"
#     sleep 5
# done

#########################
# 3. ATTACK EXPERIMENTS (all attack types x all assets)
#########################
echo ""
echo "3. Running ATTACK experiments for all attack types and assets..."
echo "=================================================="

for ATTACK_TYPE in "${ATTACK_TYPES[@]}"; do
    echo ""
    echo "--- Attack type: $ATTACK_TYPE ---"
    
    for i in "${!TARGET_ASSETS[@]}"; do
        ASSET="${TARGET_ASSETS[$i]}"
        ASSET_FILE="${TARGET_ASSETS_FILES[$i]}"
        
        # Construct the expected attack config filename
        ATTACK_CONFIG="$ATTACK_CONFIG_DIR/asset_${ASSET_FILE}--attack_${ATTACK_TYPE}--insertion_by_asset.json"
        
        echo ""
        echo "Running $ATTACK_TYPE attack for asset: $ASSET"
        echo "Attack config: $ATTACK_CONFIG"
        
        if [[ -f "$ATTACK_CONFIG" ]]; then
            python3 src/main.py \
                "fixed-bearish-neutral" \
                "$NEWS_DIR" \
                --inference-models "$INFERENCE_MODELS" \
                --fixed-assets "$ASSET" \
                --num-episodes "$NUM_EPISODES" \
                $BATCH_MODE \
                --batch-size "$BATCH_SIZE" \
                --attack-config "$ATTACK_CONFIG" \
                --output-dir "$OUTPUT_BASE" \
                --mitigations "prompt-hardening" \
                --results-file-basename "$RESULTS_FILENAME" \
                --experiment-tag "attack-${ATTACK_TYPE}-$(echo $ASSET | sed 's/ /-/g')"

            
            echo "Completed $ATTACK_TYPE attack for $ASSET"
        else
            echo "WARNING: Attack config file not found: $ATTACK_CONFIG"
            echo "Skipping $ATTACK_TYPE attack for $ASSET"
        fi
        
        sleep 5
    done
done

echo ""
echo "=================================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_BASE"
echo "=================================================="

# Summary
echo ""
echo "EXPERIMENT SUMMARY:"
echo "==================="
echo "1. All-neutral experiment: 1 run with $NUM_EPISODES episodes"
echo "2. Fixed-bearish-neutral: 4 runs (one per asset) with $NUM_EPISODES episodes each"
echo "3. Attack experiments: $((${#ATTACK_TYPES[@]} * ${#TARGET_ASSETS[@]})) runs (${#ATTACK_TYPES[@]} attack types x ${#TARGET_ASSETS[@]} assets) with $NUM_EPISODES episodes each"
echo "Total episodes: $((NUM_EPISODES * (1 + ${#TARGET_ASSETS[@]} + ${#ATTACK_TYPES[@]} * ${#TARGET_ASSETS[@]}))) episodes"
echo "Model used: $INFERENCE_MODELS"
echo "Output directory: $OUTPUT_BASE"