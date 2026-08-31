#!/usr/bin/env bash
set -euo pipefail

# Default Parameters
MODEL="${1:-facebook/opt-125m}"
DATASET="${2:-coqa}"
FRACTION="${3:-0.01}"
NUM_GEN="${4:-5}"
DEVICE="${5:-cuda:0}"
PROJECT_IND="${6:-0}"
TEMPERATURE="${7:-0.5}"
TOP_P="${8:-0.99}"
TOP_K="${9:-10}"

# 1. Locate Python / Virtualenv
PYTHON_EXE=""
if [ -f "../.venv/Scripts/python.exe" ]; then
    PYTHON_EXE="../.venv/Scripts/python.exe"
elif [ -f "../.venv/bin/python" ]; then
    PYTHON_EXE="../.venv/bin/python"
elif [ -f "./.venv/bin/python" ]; then
    PYTHON_EXE="./.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_EXE="$VIRTUAL_ENV/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -f "$VIRTUAL_ENV/Scripts/python.exe" ]; then
    PYTHON_EXE="$VIRTUAL_ENV/Scripts/python.exe"
elif command -v python3 &>/dev/null; then
    PYTHON_EXE="python3"
elif command -v python &>/dev/null; then
    PYTHON_EXE="python"
fi

SAFE_MODEL=$(echo "$MODEL" | tr '/' '_')
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FOLDER="./data/output"
mkdir -p "$OUTPUT_FOLDER"
LOG_FILE="$OUTPUT_FOLDER/run_${SAFE_MODEL}_${DATASET}_${TIMESTAMP}.log"
TARGET_DIR="$OUTPUT_FOLDER/${SAFE_MODEL}_${DATASET}_${PROJECT_IND}"

echo "=========================================================="
echo " INSIDE / EigenScore Pipeline: Generation & Evaluation"
echo "=========================================================="
echo " Model:       $MODEL"
echo " Dataset:     $DATASET"
echo " Fraction:    $FRACTION"
echo " NumGen:      $NUM_GEN"
echo " Device:      $DEVICE"
echo " ProjectInd:  $PROJECT_IND"
echo " Log File:    $LOG_FILE"
echo "=========================================================="

GEN_ARGS=(-m pipeline.generate \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --fraction_of_data_to_use "$FRACTION" \
    --num_generations_per_prompt "$NUM_GEN" \
    --device "$DEVICE" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --project_ind "$PROJECT_IND")

echo -e "\n[1/2] Running Generation and EigenScore computation..."
"$PYTHON_EXE" "${GEN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

echo -e "\n[2/2] Running Evaluation & Computing AUROC Metrics..."
PKL_FILE=$(ls -t "$TARGET_DIR"/*.pkl 2>/dev/null | grep -v "_partial" | head -n 1 || true)

if [ -n "$PKL_FILE" ] && [ -f "$PKL_FILE" ]; then
    echo "Evaluating output: $PKL_FILE"
    "$PYTHON_EXE" -m func.evalFunc "$PKL_FILE" 2>&1 | tee -a "$LOG_FILE"
else
    echo "Warning: No .pkl file found in $TARGET_DIR for evaluation."
fi

echo -e "\n=========================================================="
echo " Pipeline Finished!"
echo " Full execution log saved to: $LOG_FILE"
echo " ROC Curves exported to:      ./Figure/AUROC_${DATASET}.png"
echo "=========================================================="
