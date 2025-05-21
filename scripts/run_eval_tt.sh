#!/bin/bash

# 定义模型数组
MODELS=(
  "TongZheng1999/gemma-2-9b-it-star-mixed_direct-OP-final_v2_10-2-3Rounds-iter-2"
)

# 定义数据集数组
DATASETS=(
  "TongZheng1999/ProverQA-Hard"
  "TongZheng1999/ProverQA-Medium"
  "TongZheng1999/ProverQA-Easy"
)

# 定义模式数组
MODES=("code" "truth_table" "nl")

# 固定参数
NUM_CANDIDATES=1
BATCH_SIZE=32
MAX_TOKENS=4096
GPU_COUNT=4
TEMPERATURE=0.7
TOP_P=0.9
TOP_K=50
SEED=42
SPLIT="validation"

# 循环遍历模型、数据集和模式
for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    # 根据数据集判断 PROMPT_MODE
    if [ "$dataset" == "TongZheng1999/ProntoQA" ]; then
      PROMPT_MODE="final_v2_pronqa"
    else
      PROMPT_MODE="final_v2"
    fi

    for mode in "${MODES[@]}"; do

      # 根据当前参数构造输出目录、原始数据保存路径和结果保存路径
      # 将模型名称中的 '/' 替换为 '-'，数据集名称取最后一部分作为标识
      model_id=${model//\//-}
      dataset_id=$(basename "$dataset")
      OUTPUT_DIR="./final_results/${model_id}/${dataset_id}/${mode}/"
      SAVE_RAW_DATA_PATH="raw_data_${model_id}_${dataset_id}_${mode}.json"
      SAVE_RESULT_PATH="result_${model_id}_${dataset_id}_${mode}.txt"
      
      # 打印当前运行的参数，方便调试
      echo "=============================================="
      echo "Running evaluation with parameters:"
      echo "Model: ${model}"
      echo "Dataset: ${dataset}"
      echo "Mode: ${mode}"
      echo "Prompt mode: ${PROMPT_MODE}"
      echo "Output Dir: ${OUTPUT_DIR}"
      echo "Raw data path: ${SAVE_RAW_DATA_PATH}"
      echo "Result path: ${SAVE_RESULT_PATH}"
      echo "=============================================="
      
      # 调用 Python 脚本进行评估
      python ./eval/eval.py \
        --model_name_and_path "${model}" \
        --mode "${mode}" \
        --gpu_count "${GPU_COUNT}" \
        --prompt_mode "${PROMPT_MODE}" \
        --dataset_name "${dataset}" \
        --output_dir "${OUTPUT_DIR}" \
        --save_raw_data_path "${SAVE_RAW_DATA_PATH}" \
        --save_result_path "${SAVE_RESULT_PATH}" \
        --batch_size "${BATCH_SIZE}" \
        --max_tokens "${MAX_TOKENS}" \
        --temperature "${TEMPERATURE}" \
        --top_p "${TOP_P}" \
        --top_k "${TOP_K}" \
        --seed "${SEED}" \
        --split "${SPLIT}" \
        --number_candidates "${NUM_CANDIDATES}" 
      
      echo "Evaluation finished for ${model} | ${dataset} | ${mode}"
      echo ""
    done
  done
done

