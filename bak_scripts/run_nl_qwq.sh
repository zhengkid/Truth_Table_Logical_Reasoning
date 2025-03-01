#!/usr/bin/env bash

BASE_MODEL="Qwen/QwQ-32B-Preview"  #"meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
MODE="nl"
DATASET_NAME='yale-nlp/FOLIO'
OUTPUT_DIR="outputs_nl/Qwen/QwQ-32B-Preview"
N_SAMPLES=1000
N_OUTER_LOOPS=3
N_EPOCHS=5
BATCH_SIZE=128
MACRO_BATCH_SIZE=1
LEARNING_RATE=5e-6
LORA="--lora" 
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.0
MAX_TOKENS=2048
TEMPERATURE=1.0
TEST_TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
SEED=42

python star_pipeline.py \
    --model_name_and_path "$BASE_MODEL" \
    --mode "$MODE" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --n_samples "$N_SAMPLES" \
    --n_outer_loops "$N_OUTER_LOOPS" \
    --n_epochs "$N_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --micro_batch_size "$MACRO_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    $LORA \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout "$LORA_DROPOUT" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --test_temperature "$TEST_TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --seed "$SEED"
