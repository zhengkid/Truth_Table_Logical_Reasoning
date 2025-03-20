#!/bin/bash
# star_pipeline_all_flags.sh
# General parameters
# Model and Datasets
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct" #"TongZheng1999/gemma-2-9b-it-star-mixed_direct-OP-final_v1_10-2-3Rounds-iter-3" #"google/gemma-2-9b-it"
DATASET="yale-nlp/FOLIO"
MODEL_NAME=${BASE_MODEL##*/}


N_SAMPLES=1000
N_ROUNDS=3
INFERENCE_BATCH_SIZE=32
SEED=42
GPU_COUNR=4
MAX_TOKENS=2048
NUM_CANDIDATES_EVAL=1
NUM_CANDIDATES_GENERATE=10
EPOCHS=2
TEMP=1.0
TEST_TEMP=0.7
TOP_P=0.9
TOP_K=50


MODES=("nl" "code" "truth_table") 
PROMPT_MODE='final_v2'
MIXTURE_MODE='direct'  # direct or other
STRAREGY="OP"


SRC_YAML="alignment-handbook/recipes/star_training/sft/config_full.yaml"
HF_USER="TongZheng1999"
RECIPE_DIR='alignment-handbook/recipes/'
ACC_PATH='alignment-handbook/recipes/accelerate_configs/deepspeed_zero3.yaml'
PATTERN='TongZheng1999/nl_rationale_1000_round_1'

# Save Path 
MODEL_OUTPUT_DIR="/beacon-scratch/tongzh24/" 
SAVE_RAW_DATA_PATH='Eval_Rationale_Raw_Data_round_'
SAVE_RESULT_PATH='Result_round_'

# Recipe file location (used only for --parse)
MODEL_SHORTNAME=$(basename "$BASE_MODEL")

export PYTHONHASHSEED=$SEED

# Phase -1: Initial evaluation using base model (few-shot)
echo "Phase -1: Evaluating few-shot performance with $BASE_MODEL"
for MODE in "${MODES[@]}"; do
    echo "Phase -1: Evaluating few-shot performance with $BASE_MODEL in mode: $MODE"
    OUTPUT_DIR="star_pipeline_outputs/${MODEL_NAME}/MIX_${MIXTURE_MODE}/${MODE}/${STRAREGY}_${PROMPT_MODE}_${NUM_CANDIDATES_GENERATE}_${EPOCHS}_${N_ROUNDS}Rounds"
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Directory does not exist. Creating: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    else
        echo "Directory already exists: $OUTPUT_DIR"
    fi
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
        --gpu_count ${GPU_COUNR} \
        --number_candidates ${NUM_CANDIDATES_EVAL} \
        --use_fewshot

    sleep 5
done


# Set CURRENT_MODEL to the base model initially
CURRENT_MODEL="$BASE_MODEL"

# Iterative STaR Pipeline Loop
for (( round=1; round<=N_ROUNDS; round++ ))
do
  echo "===== Round $round ====="
  
  # Stage 1: Rationale Generation using CURRENT_MODEL (first round use fewshot)
  if [ $round -eq 1 ]; then
      FEWSHOT="--use_fewshot"
  else
      FEWSHOT=""
  fi

  echo "Stage 1: Generating rationales for round $round using model: $CURRENT_MODEL"
  INPUT_DATASETS=()
  for MODE in "${MODES[@]}"; do
    echo "Stage 1: Generating rationales for round $round using model: $CURRENT_MODEL in mode: $MODE"
    HF_DATA_ID="${MODEL_NAME}_mixed_${MIXTURE_MODE}_${MODE}_${STRAREGY}_r_${N_SAMPLES}_${PROMPT_MODE}_${NUM_CANDIDATES_GENERATE}_${EPOCHS}_${N_ROUNDS}R_round_${round}"
    python generate_rationale.py \
        --model_name_and_path "$CURRENT_MODEL" \
        --dataset_name "$DATASET" \
        --mode "$MODE" \
        --seed "$SEED" \
        --prompt_mode ${PROMPT_MODE} \
        --n_samples "$N_SAMPLES" \
        --huggingface_repo "${HF_USER}/${HF_DATA_ID}" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEMP" \
        --batch_size "$INFERENCE_BATCH_SIZE" \
        --top_p "$TOP_P" \
        --top_k "$TOP_K" \
        --gpu_count ${GPU_COUNR} \
        --number_candidates ${NUM_CANDIDATES_GENERATE} \
        $FEWSHOT
    
    sleep 5
    INPUT_DATASETS+=("${HF_USER}/${HF_DATA_ID}")
  done
  
  if [[ "$MIXTURE_MODE" != "direct" && "$MIXTURE_MODE" != "unique_conclusion" ]]; then
    echo "Error: Invalid MIXTURE_MODE: $MIXTURE_MODE"
    exit 1
  fi
  echo "Stage 2: Mix Generated rationales for round $round"
  OUTPUT_DATASETS=${HF_USER}/${MODEL_NAME}_mixed_${MIXTURE_MODE}_${STRAREGY}_rationale_${N_SAMPLES}_${PROMPT_MODE}_${NUM_CANDIDATES_GENERATE}_${EPOCHS}_${N_ROUNDS}Rounds_round_${round}
  python utils/mix_datasets.py \
    --input_datasets "${INPUT_DATASETS[@]}" \
    --output_dataset "${OUTPUT_DATASETS}" \
    --mix_mode ${MIXTURE_MODE} \
    --seed $SEED
  

  # Stage 3: Fine-tuning with generated rationales
  ITER_YAML_DIR="$RECIPE_DIR/${MODEL_NAME}_mixed_${MIXTURE_MODE}_${NUM_CANDIDATES_GENERATE}_star_training"
  if [ ! -d "$ITER_YAML_DIR" ]; then
    echo "Directory does not exist. Creating: $ITER_YAML_DIR"
    mkdir -p "$ITER_YAML_DIR"
  else
    echo "Directory already exists: $ITER_YAML_DIR"
  fi
  ITER_YAML="$ITER_YAML_DIR/iter_${round}_config.yaml"

  cp "$SRC_YAML" "$ITER_YAML"
   
  python utils/utils.py $ITER_YAML ${MODEL_OUTPUT_DIR}/${MODEL_NAME}/mixed_${MIXTURE_MODE}/${STRAREGY}_${PROMPT_MODE}_${NUM_CANDIDATES_GENERATE}_${EPOCHS}_${N_ROUNDS}Rounds/ft_iter_${round} ${MODEL_NAME}-star-mixed_${MIXTURE_MODE}-${STRAREGY}-${PROMPT_MODE}_${NUM_CANDIDATES_GENERATE}-${EPOCHS}-${N_ROUNDS}Rounds-iter-${round} $CURRENT_MODEL $CURRENT_MODEL $PATTERN $OUTPUT_DATASETS
  
  echo "Updated: $ITER_YAML"


  SFT_SAVE_PATH=$(grep  '^output_dir:' "$ITER_YAML" | cut -d':' -f2 | tr -d ' ')
  echo "$SFT_SAVE_PATH"
  echo "Stage 3: Fine-tuning base model with rationales (round $round)..."
  ACCELERATE_LOG_LEVEL=info accelerate launch --config_file $ACC_PATH alignment-handbook/scripts/run_sft.py $ITER_YAML
  
  sleep 20
   
  # Update CURRENT_MODEL for the next round
  CURRENT_MODEL="$SFT_SAVE_PATH"
  
  # Stage 3: Evaluation using the fine-tuned model
  echo "Stage 4: Evaluating fine-tuned model for round $round using model: $CURRENT_MODEL"
  for MODE in "${MODES[@]}"; do
    echo "Phase 4: Evaluating few-shot performance with $CURRENT_MODEL in mode: $MODE"
    OUTPUT_DIR="star_pipeline_outputs/${MODEL_NAME}/MIX_${MIXTURE_MODE}/${MODE}/${STRAREGY}_${PROMPT_MODE}_${NUM_CANDIDATES_GENERATE}_${EPOCHS}_${N_ROUNDS}Rounds"
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Directory does not exist. Creating: $OUTPUT_DIR"
        mkdir -p "$OUTPUT_DIR"
    else
        echo "Directory already exists: $OUTPUT_DIR"
    fi
    python eval/eval.py \
        --model_name_and_path "$CURRENT_MODEL" \
        --dataset_name "$DATASET" \
        --seed "$SEED" \
        --prompt_mode ${PROMPT_MODE} \
        --output_dir "$OUTPUT_DIR" \
        --save_raw_data_path "${SAVE_RAW_DATA_PATH}${round}.txt" \
        --save_result_path "${SAVE_RESULT_PATH}${round}.txt" \
        --batch_size "$INFERENCE_BATCH_SIZE" \
        --max_tokens "$MAX_TOKENS" \
        --temperature "$TEST_TEMP" \
        --top_p "$TOP_P" \
        --top_k "$TOP_K" \
        --mode "$MODE" \
        --gpu_count ${GPU_COUNR} \
        --number_candidates ${NUM_CANDIDATES_EVAL} 
    sleep 5
  done

  echo "===== Round $round complete ====="
  echo ""
done

echo "STaR pipeline completed."

