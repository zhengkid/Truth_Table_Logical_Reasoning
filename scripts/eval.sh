#!/bin/bash
MODEL_NAME_AND_PATH="google/gemma-2-2b-it" #"Qwen/Qwen2.5-7B-Instruct" or other models
MODE="nl"
DATASET_NAME="yale-nlp/FOLIO" #"TongZheng1999/ProofWriter" #"yale-nlp/FOLIO" #'TongZheng1999/ProverQA-Hard'
OUTPUT_DIR= 
SAVE_RAW_DATA_PATH="raw_data.json"
SAVE_RESULT_PATH="result.txt"
PROMPT_MODE='final_v2'
NUM_CANDIDATES=1
BATCH_SIZE=32
MAX_TOKENS=2048
GPU_COUNR=4
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
SEED=42
SPLIT='validation'

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
  --split ${SPLIT} \
  --number_candidates ${NUM_CANDIDATES} \
  --use_fewshot
