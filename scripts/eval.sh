#!/bin/bash




MODEL_NAME_AND_PATH="TongZheng1999/gemma-2-9b-it-star-nl-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-star-truth_table-v1_10-2-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-star-code-v3_10-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-star-nl-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-star-truth_table-v1_10-2-3Rounds-iter-1" #"TongZheng1999/gemma-2-9b-it-star-code-v3_10-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-star-nl-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-mix" #"TongZheng1999/gemma-2-9b-it-star-code-v3_10-3Rounds-iter-2" #"TongZheng1999/gemma-2-9b-it-star-nl-3Rounds-iter-2" #"google/gemma-2-9b-it"  #"TongZheng1999/gemma-2-9b-it-star-truth_table-v1_10-2-3Rounds-iter-3"

MODE="truth_table"
DATASET_NAME='yale-nlp/FOLIO'
OUTPUT_DIR="./final_outputs_all_rounds/gemma-2-9b-it-nl/round3/nl_upper_bound_num_candidate_9/"
SAVE_RAW_DATA_PATH="raw_data.json"
SAVE_RESULT_PATH="result.txt"
PROMPT_MODE='v1'
NUM_CANDIDATES=9
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
  --number_candidates ${NUM_CANDIDATES} 
