#!/bin/bash

# Set default parameters
MODEL_NAME_AND_PATH="google/gemma-2-9b-it"
MODE="nl"         # Options: truth_table, code, nl
DATASET_NAME="yale-nlp/FOLIO"
OUTPUT_DIR="outputs"
OUTPUT_FILE="rationales.json"
N_SAMPLES=200
BATCH_SIZE=16
MAX_TOKENS=512
TEMPERATURE=1.0
TOP_P=0.9
TOP_K=50
SEED=42
USE_FEWSHOT="--use_fewshot"  # Comment out if fewshot is not needed

# Print runtime parameters
echo "Running with the following parameters:"
echo "MODEL_NAME_AND_PATH: $MODEL_NAME_AND_PATH"
echo "MODE: $MODE"
echo "DATASET_NAME: $DATASET_NAME"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "OUTPUT_FILE: $OUTPUT_FILE"
echo "N_SAMPLES: $N_SAMPLES"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_TOKENS: $MAX_TOKENS"
echo "TEMPERATURE: $TEMPERATURE"
echo "TOP_P: $TOP_P"
echo "TOP_K: $TOP_K"
echo "SEED: $SEED"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the Python script
python ../generate_rationale.py \
  --model_name_and_path "$MODEL_NAME_AND_PATH" \
  --mode "$MODE" \
  --dataset_name "$DATASET_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --output_file "$OUTPUT_FILE" \
  --n_samples "$N_SAMPLES" \
  --batch_size "$BATCH_SIZE" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --seed "$SEED" \
  $USE_FEWSHOT

