#!/usr/bin/env bash

BASE_MODEL="unsloth/gemma-2-2b-it" #"google/gemma-2-2b-it"  #"meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
MODE="nl"
DATASET_NAME='yale-nlp/FOLIO'
OUTPUT_DIR="/beacon-scratch/tongzh24/outputs_nl/gemma-2-2b-it-bs-64-max-length-4096-epoch-5-wo-fewshot-new"
N_SAMPLES=1000
N_OUTER_LOOPS=2
N_EPOCHS=5
BATCH_SIZE=64
MACRO_BATCH_SIZE=4
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
