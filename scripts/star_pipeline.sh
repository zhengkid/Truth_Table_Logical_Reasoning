#!/bin/bash
# star_pipeline_all_flags.sh

# General parameters
BASE_MODEL="google/gemma-2-9b-it"
DATASET="yale-nlp/FOLIO"
OUTPUT_DIR="star_pipeline_outputs/gemma-2-9b-it/nl"
SAVE_RAW_DATA_PATH='Eval_Rationale_Raw_Data_round_'
SAVE_RESULT_PATH='Result_round_'
RATIONALE_FILE_PATH='Rationale_data_round_'
N_SAMPLES=1000
N_ROUNDS=10
N_EPOCHS=4
BATCH_SIZE=64
INFERENCE_BATCH_SIZE=16
MICRO_BATCH_SIZE=4
LEARNING_RATE=1e-5
SEED=42
MAX_TOKENS=1024
TEMP=1.0
TEST_TEMP=0.7
TOP_P=0.9
TOP_K=50
MODE="nl"     # Options: truth_table, code, nl
IS_CHAT="true"        # "true" or "false"

# Recipe file location (used only for --parse)
MODEL_SHORTNAME=$(basename "$BASE_MODEL")
RECIPE="recipes/${MODEL_SHORTNAME}/${MODE}/config_full.yaml"

export PYTHONHASHSEED=$SEED
mkdir -p "$OUTPUT_DIR"

# Phase -1: Initial evaluation using base model (few-shot)
echo "Phase -1: Evaluating few-shot performance with base model..."
python ../eval/eval.py \
  --model_name_and_path "$BASE_MODEL" \
  --dataset_name "$DATASET" \
  --seed "$SEED" \
  --output_dir "$OUTPUT_DIR" \
  --save_raw_data_path "${SAVE_RAW_DATA_PATH}0.txt" \
  --save_result_path "${SAVE_RESULT_PATH}0.txt" \
  --batch_size "$INFERENCE_BATCH_SIZE" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEST_TEMP" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --mode "$MODE" \
  --use_fewshot

sleep 5

# Set CURRENT_MODEL to the base model initially
CURRENT_MODEL="$BASE_MODEL"

# Iterative STaR Pipeline Loop
for (( round=1; round<=N_ROUNDS; round++ ))
do
  echo "===== Round $round ====="
  
  # Stage 1: Rationale Generation using CURRENT_MODEL
  if [ $round -eq 1 ]; then
      FEWSHOT="--use_fewshot"
  else
      FEWSHOT=""
  fi
  
  echo "Stage 1: Generating rationales for round $round using model: $CURRENT_MODEL"
  python ../generate_rationale.py \
    --model_name_and_path "$CURRENT_MODEL" \
    --dataset_name "$DATASET" \
    --mode "$MODE" \
    --output_dir "$OUTPUT_DIR" \
    --output_file "${RATIONALE_FILE_PATH}${round}.json" \
    --seed "$SEED" \
    --n_samples "$N_SAMPLES" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEMP" \
    --batch_size "$INFERENCE_BATCH_SIZE" \ 
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --mode "$MODE" \
    --use_fewshot 
  
  sleep 5
  
  # Stage 2: Fine-tuning with generated rationales
  FT_SAVE_PATH="fine_tuning_${MODE}_${BATCH_SIZE}_${LEARNING_RATE}_round_${round}"
  
  echo "Stage 2: Fine-tuning base model with rationales (round $round)..."
  ACCELERATE_LOG_LEVEL=info accelerate launch --parse "$RECIPE" scripts/run_finetune.py \
    --model_name_or_path "$BASE_MODEL" \
    --model_revision "main" \
    --tokenizer_name_or_path "$BASE_MODEL" \
    --torch_dtype "bfloat16" \
    --attn_implementation "flash_attention_2" \
    --dataset_mixer "${OUTPUT_DIR}/${RATIONALE_FILE_PATH}${round}.json" \
    --dataset_splits "train_sft,test_sft" \
    --preprocessing_num_workers 12 \
    --bf16 true \
    --add_special_tokens false \
    --append_concat_token false \
    --do_eval false \
    --eval_strategy "no" \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --use_reentrant false \
    --learning_rate 5.0e-06 \
    --log_level info \
    --logging_steps 5 \
    --logging_strategy steps \
    --lr_scheduler_type cosine \
    --max_seq_length 2048 \
    --max_steps -1 \
    --num_train_epochs 3 \
    --output_dir "$OUTPUT_DIR/$FT_SAVE_PATH" \
    --overwrite_output_dir true \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 4 \
    --push_to_hub true \
    --remove_unused_columns true \
    --report_to "tensorboard wandb" \
    --save_strategy no \
    --seed 42 \
    --warmup_ratio 0.1 \
    --batch_size "$BATCH_SIZE" \
    --micro_batch_size "$MICRO_BATCH_SIZE" \
    --learning_rate_ft "$LEARNING_RATE"
  
  sleep 20
  
  # Update CURRENT_MODEL for the next round
  CURRENT_MODEL="$OUTPUT_DIR/$FT_SAVE_PATH"
  
  # Stage 3: Evaluation using the fine-tuned model
  echo "Stage 3: Evaluating fine-tuned model for round $round using model: $CURRENT_MODEL"
  python evaluate_batch.py \
    --model_name_and_path "$CURRENT_MODEL" \
    --dataset_name "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --save_raw_data_path "" \
    --save_result_path "" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEST_TEMP" \
    --batch_size "$INFERENCE_BATCH_SIZE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --mode "$MODE" \
    --is_chat_model "$IS_CHAT" \
    --use_fewshot false
  
  echo "===== Round $round complete ====="
  echo ""
done

echo "STaR pipeline completed."

