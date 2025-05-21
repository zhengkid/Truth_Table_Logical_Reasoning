#!/usr/bin/env bash
set -euo pipefail

# —— 配置区 —— #
MODEL_NAME_AND_PATH="o4-mini-2025-04-16"
MODE="code"                        # truth_table, code, nl
DATASET_NAME="yale-nlp/FOLIO"
OUTPUT_DIR="o4-data/code/"
OUTPUT_FILE="rationales.jsonl"  # checkpoint 文件名
HUGGINGFACE_REPO="TongZheng1999/o4-FL-code-data"             # 填你的 user/repo，否则推送会跳过
USE_FEWSHOT="--use_fewshot"      # 留空 "" 则不启用 few-shot

# 可选覆盖其他参数：
N_SAMPLES=1001
MAX_TOKENS=3072
TEMPERATURE=1.0
TOP_P=0.9
TOP_K=50
SEED=42
PROMPT_MODE="final_v2"
# —— 检查环境 —— #
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "❌ 请先设置 OPENAI_API_KEY"
  exit 1
fi

# —— 打印参数 —— #
echo "运行配置："
echo "  模型:            $MODEL_NAME_AND_PATH"
echo "  模式:            $MODE"
echo "  数据集:          $DATASET_NAME"
echo "  checkpoint 文件: $OUTPUT_DIR/$OUTPUT_FILE"
echo "  few-shot:        ${USE_FEWSHOT:+enabled}${USE_FEWSHOT:+" ($USE_FEWSHOT)"}"
echo "  HuggingFace Repo: ${HUGGINGFACE_REPO:-<none>}"
echo ""

# —— 准备输出目录 —— #
mkdir -p "$OUTPUT_DIR"

# —— 执行脚本 —— #
python ./generate_rationale_openai.py \
  --model_name_and_path "$MODEL_NAME_AND_PATH" \
  --mode "$MODE" \
  --prompt_mode ${PROMPT_MODE} \
  --dataset_name "$DATASET_NAME" \
  --checkpoint_file "$OUTPUT_DIR/$OUTPUT_FILE" \
  --n_samples "$N_SAMPLES" \
  --max_tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --top_k "$TOP_K" \
  --seed "$SEED" \
  --huggingface_repo "$HUGGINGFACE_REPO" \
  $USE_FEWSHOT

