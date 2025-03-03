#!/bin/bash




MODEL_NAME_AND_PATH="NousResearch/Meta-Llama-3.1-8B-Instruct"
MODE="nl"
DATASET_NAME='yale-nlp/FOLIO'
OUTPUT_DIR="./outputs3/prompt/google/Meta-Llama-3.1-8B-Instruct-star-1/2048/few-shot"
SAVE_RAW_DATA_PATH="raw_data.json"
SAVE_RESULT_PATH="result.txt"
PROMPT_MODE='v1'
NUM_CANDIDATES=1
BATCH_SIZE=24
MAX_TOKENS=2048
GPU_COUNR=8
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
SEED=42

# 运行 Python 脚本
python ./eval/eval.py \
  --model_name_and_path ${MODEL_NAME_AND_PATH} \
  --mode ${MODE} \
  --gpu_count ${GPU_COUNR} \
  --prompt_mode ${PROMPT_MODE} \
  --dataset_name ${DATASET_NAME} \
  --output_dir ${OUTPUT_DIR} \
  --save_raw_data_path ${SAVE_RAW_DATA_PATH} \
  --save_result_path ${SAVE_RESULT_PATH} \
  --batch_size ${BATCH_SIZE} \
  --max_tokens ${MAX_TOKENS} \
  --temperature ${TEMPERATURE} \
  --top_p ${TOP_P} \
  --top_k ${TOP_K} \
  --seed ${SEED} \
  --number_candidates ${NUM_CANDIDATES} \
  --use_fewshot
