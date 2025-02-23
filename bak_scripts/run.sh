#!/usr/bin/env bash

BASE_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  #"meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
MODE="truth_table"
DATASET_NAME='yale-nlp/FOLIO'
OUTPUT_DIR="outputs"
N_SAMPLES=1000
N_OUTER_LOOPS=3
N_EPOCHS=4
BATCH_SIZE=16
LEARNING_RATE=5e-6
LORA="--lora" 
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.0
MAX_TOKENS=5140
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
SEED=42

python star_pipeline_together_ai.py \
    --base_model "$BASE_MODEL" \
    --mode "$MODE" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --n_samples "$N_SAMPLES" \
    --n_outer_loops "$N_OUTER_LOOPS" \
    --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    $LORA \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --seed "$SEED"
