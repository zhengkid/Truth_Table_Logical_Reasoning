#!/bin/bash


export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_NAME_AND_PATH="TongZheng1999/gemma-2-9b-it-star-nl-3Rounds-iter-2"
MODE="nl"
DATASET_NAME='yale-nlp/FOLIO'
OUTPUT_DIR="../outputs/prompting/gemma-2-9b-it-nl-star-inter2/5120/few-shot"
SAVE_RAW_DATA_PATH="raw_data.json"
SAVE_RESULT_PATH="result.txt"
BATCH_SIZE=16
MAX_TOKENS=5120
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
SEED=42

# 运行 Python 脚本
python ./eval/eval.py \
  --model_name_and_path ${MODEL_NAME_AND_PATH} \
  --mode ${MODE} \
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
  --use_fewshot

