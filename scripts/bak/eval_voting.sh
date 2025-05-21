#!/bin/bash




MODEL_NAME_AND_PATH="google/gemma-2-9b-it" #"Qwen/Qwen2.5-7B-Instruct" #"TongZheng1999/PF_gemma-2-9b-it-s1" #"google/gemma-2-2b-it" #"/beacon-scratch/tongzh24/gemma-2-2b-it/FL_mixed_direct/OP_final_v2_10_2_3Rounds/ft_iter_3/" #"TongZheng1999/gemma-2-9b-it-star-mixed_direct-OP-final_v2_10-2-3Rounds-iter-2" #"google/gemma-2-9b-it" #"/beacon-scratch/tongzh24/gemma-2-9b-it/mixed_direct/OP_final_v2_1_2_5Rounds/ft_iter_2/" #"TongZheng1999/gemma-2-9b-it-star-mixed_direct-OP-final_v2_10-2-3Rounds-iter-1" #"google/gemma-2-9b-it"  #"TongZheng1999/gemma-2-9b-it-star-mixed_direct-OP-final_v2_10-2-3Rounds-iter-2" 
#"Qwen/Qwen2.5-7B-Instruct" #"unsloth/Meta-Llama-3.1-8B-Instruct" #"Qwen/Qwen2.5-7B-Instruct" #"/beacon-scratch/tongzh24/gemma-2-9b-it/mixed_direct/OF_final_v2_10_2_3Rounds/ft_iter_2/" #"Qwen/Qwen2.5-7B-Instruct" #"/beacon-scratch/tongzh24/gemma-2-9b-it/mixed_direct/OP_final_v2_10_2_3Rounds/ft_iter_2/" #"Qwen/Qwen2.5-7B-Instruct" #"Qwen/Qwen2.5-7B-Instruct"  #"unsloth/Meta-Llama-3.1-8B-Instruct" #"Qwen/Qwen2.5-7B-Instruct" #"TongZheng1999/gemma-2-9b-it-star-mixed_direct-OP-final_v1_10-2-3Rounds-iter-3" #"/beacon-scratch/tongzh24/gemma-2-9b-it/mix-iter1-new-5-epoch" #"/beacon-scratch/tongzh24/gemma-2-9b-it/nl/final_1_5_3Rounds/ft_iter_3/" #"google/gemma-2-9b-it" #"TongZheng1999/gemma-2-9b-it-star-nl-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-star-truth_table-v1_10-2-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-star-code-v3_10-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-star-nl-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-star-truth_table-v1_10-2-3Rounds-iter-1" #"TongZheng1999/gemma-2-9b-it-star-code-v3_10-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-star-nl-3Rounds-iter-3" #"TongZheng1999/gemma-2-9b-it-mix" #"TongZheng1999/gemma-2-9b-it-star-code-v3_10-3Rounds-iter-2" #"TongZheng1999/gemma-2-9b-it-star-nl-3Rounds-iter-2" #"google/gemma-2-9b-it"  #"TongZheng1999/gemma-2-9b-it-star-truth_table-v1_10-2-3Rounds-iter-3"

MODE="nl"
DATASET_NAME="yale-nlp/FOLIO" #"TongZheng1999/ProofWriter" #"yale-nlp/FOLIO" #"TongZheng1999/ProntoQA" #'yale-nlp/FOLIO' #'TongZheng1999/ProofWriter'
OUTPUT_DIR="./folio_results/gemma-2-9b-it/sot-3-vote/nl/"
SAVE_RAW_DATA_PATH="raw_data.json"
SAVE_RESULT_PATH="result.txt"
PROMPT_MODE='final_v2'
NUM_CANDIDATES=3
BATCH_SIZE=32
MAX_TOKENS=2048
GPU_COUNR=4
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
SEED=42
SPLIT='validation'

# 运行 Python 脚本
python ./eval/eval_multi_candidate.py \
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
  --mixture_mode "multi_candidate" \
  --colab_modes "nl/truth_table" \
  --use_fewshot
  
