#!/bin/bash
# star_pipeline_all_flags.sh
# General parameters
N_SAMPLES=1000
N_ROUNDS=3
INFERENCE_BATCH_SIZE=32
SEED=42
GPU_COUNT=4
MAX_TOKENS=2048
NUM_CANDIDATES_EVAL=1
NUM_CANDIDATES_GENERATE=10
EPOCHS=2
TEMP=1.0
PROMPT_MODE='final_v2'
TEST_TEMP=0.7
TOP_P=0.9
TOP_K=50
MODE="nl"     # Options: truth_table, code, nl
STRATEGY="OP_DIS_new"
# Base model for fine-tuning and initial generation
BASE_MODEL="google/gemma-2-2b-it"
MODEL_NAME=${BASE_MODEL##*/}
DATASET="yale-nlp/FOLIO"
OUTPUT_DIR="star_pipeline_outputs/${MODEL_NAME}/${MODE}/${STRATEGY}_${PROMPT_MODE}_${NUM_CANDIDATES_GENERATE}_${EPOCHS}_${N_ROUNDS}Rounds"
MODEL_OUTPUT_DIR="/beacon-scratch/tzheng24/"
SRC_YAML="alignment-handbook/recipes/star_training/sft/config_full.yaml"
HF_USER="TongZheng1999"
HF_DATA_PREFIX="${MODEL_NAME}_${MODE}_${STRATEGY}_rationale_${N_SAMPLES}_${PROMPT_MODE}_${NUM_CANDIDATES_GENERATE}_${EPOCHS}_${N_ROUNDS}Rounds_round_"
INITIAL_DATASET="TongZheng1999/o4-FL-data-transformed"
SAVE_RAW_DATA_PATH='Eval_Rationale_Raw_Data_round_'
SAVE_RESULT_PATH='Result_round_'
RECIPE_DIR='alignment-handbook/recipes/'
ACC_PATH='alignment-handbook/recipes/accelerate_configs/deepspeed_zero3.yaml'
PATTERN='TongZheng1999/nl_rationale_1000_round_1'

export PYTHONHASHSEED=$SEED
mkdir -p "$OUTPUT_DIR"

# Phase -1: Initial evaluation using base model (few-shot)
echo "Phase -1: Evaluating few-shot performance with base model..."
python eval/eval.py \
  --model_name_and_path "$BASE_MODEL" \
  --dataset_name "$DATASET" \
  --seed "$SEED" \
  --prompt_mode ${PROMPT_MODE} \
  --output_dir "$OUTPUT_DIR" \
  --save_raw_data_path "${SAVE_RAW_DATA_PATH}0.txt" \
  --save_result_path "${SAVE_RESULT_PATH}0.txt" \
  --batch_size "$INFERENCE_BATCH_SIZE" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEST_TEMP" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --mode "$MODE" \
  --gpu_count ${GPU_COUNT} \
  --number_candidates ${NUM_CANDIDATES_EVAL} \
  --use_fewshot

sleep 5

# Iterative STaR Pipeline Loop
# CURRENT_MODEL will point to the model used for generation and finetuning each round
CURRENT_MODEL="$BASE_MODEL"
for (( round=1; round<=N_ROUNDS; round++ ))
do
  # Stage 1: Decide rationale data source
  if [ $round -eq 1 ]; then
    echo "Round 1: skip rationale generation, use existing data: $INITIAL_DATASET"
    DATA_REPO="$INITIAL_DATASET"
  else
    echo "Stage 1: Generating rationales for round $round using model: $CURRENT_MODEL"
    python generate_rationale.py \
      --model_name_and_path "$CURRENT_MODEL" \
      --dataset_name "$DATASET" \
      --mode "$MODE" \
      --seed "$SEED" \
      --prompt_mode ${PROMPT_MODE} \
      --n_samples "$N_SAMPLES" \
      --huggingface_repo "${HF_USER}/${HF_DATA_PREFIX}${round}" \
      --max_tokens "$MAX_TOKENS" \
      --temperature "$TEMP" \
      --batch_size "$INFERENCE_BATCH_SIZE" \
      --top_p "$TOP_P" \
      --top_k "$TOP_K" \
      --gpu_count ${GPU_COUNT} \
      --number_candidates ${NUM_CANDIDATES_GENERATE}

    sleep 5
    DATA_REPO="${HF_USER}/${HF_DATA_PREFIX}${round}"
  fi

  # Stage 2: Fine-tuning with chosen data
  echo "Stage 2: Fine-tuning $CURRENT_MODEL with data from $DATA_REPO"
  ITER_YAML_DIR="$RECIPE_DIR/${MODEL_NAME}_${PROMPT_MODE}_${MODE}_star_training"
  mkdir -p "$ITER_YAML_DIR"
  ITER_YAML="$ITER_YAML_DIR/iter_${round}_config.yaml"
  cp "$SRC_YAML" "$ITER_YAML"

  # Fine-tune the model from previous round (CURRENT_MODEL)
  python utils/utils.py $ITER_YAML \
    ${MODEL_OUTPUT_DIR}/${MODEL_NAME}/${MODE}/${STRATEGY}_${PROMPT_MODE}_${NUM_CANDIDATES_GENERATE}_${EPOCHS}_${N_ROUNDS}Rounds/ft_iter_${round} \
    ${MODEL_NAME}-star-${MODE}-${STRATEGY}-${PROMPT_MODE}_${NUM_CANDIDATES_GENERATE}-${EPOCHS}-${N_ROUNDS}Rounds-iter-${round} \
    $CURRENT_MODEL $CURRENT_MODEL $PATTERN \
    $DATA_REPO

  ACCELERATE_LOG_LEVEL=info accelerate launch --config_file $ACC_PATH alignment-handbook/scripts/run_sft.py $ITER_YAML

  SFT_SAVE_PATH=$(grep '^output_dir:' "$ITER_YAML" | cut -d':' -f2 | tr -d ' ')
  CURRENT_MODEL="$SFT_SAVE_PATH"
  echo "Finished fine-tune round $round, new model at $CURRENT_MODEL"

  # Stage 3: Evaluation using the fine-tuned model
  echo "Stage 3: Evaluating fine-tuned model for round $round using model: $CURRENT_MODEL"
  python eval/eval.py \
    --model_name_and_path "$CURRENT_MODEL" \
    --dataset_name "$DATASET" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR" \
    --save_raw_data_path "${SAVE_RAW_DATA_PATH}${round}.txt" \
    --save_result_path "${SAVE_RESULT_PATH}${round}.txt" \
    --batch_size "$INFERENCE_BATCH_SIZE" \
    --max_tokens "$MAX_TOKENS" \
    --temperature "$TEST_TEMP" \
    --prompt_mode ${PROMPT_MODE} \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --gpu_count ${GPU_COUNT} \
    --number_candidates ${NUM_CANDIDATES_EVAL} \
    --mode "$MODE"

  echo "===== Round $round complete ====="
  echo ""
done

echo "STaR pipeline completed."

