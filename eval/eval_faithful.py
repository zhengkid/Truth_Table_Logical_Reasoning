import os
import json
import openai
from typing import List, Dict

# -----------------------------------------------------------------------------
# 配置：请先在环境变量中设置你的 OPENAI_API_KEY
# -----------------------------------------------------------------------------
# export OPENAI_API_KEY="your_api_key_here"
openai.api_key = os.getenv("OPENAI_API_KEY")

# -----------------------------------------------------------------------------
# 单条评估函数，与之前示例几乎相同
# -----------------------------------------------------------------------------
def evaluate_rationale(premises: str,
                       conclusion: str,
                       rationale: str,
                       label: str,
                       predict: str,
                       model: str = "gpt-4o") -> Dict:
    # system_prompt = (
    #     "You must determine whether a rationale faithfully justifies the truth value of a conclusion given a set of premises.\n\n"
    #     "“Faithful” means all and only the steps actually used in deriving the conclusion:\n"
    #     "- are grounded in the given premises or prior derived steps,\n"
    #     "- apply valid inference rules (no illicit converse or contraposition),\n"
    #     "- cover every disjunction branch or quantifier case,\n"
    #     "- use no unstated assumptions or background knowledge.\n"
    #     "- and correctly assesses whether the conclusion is supported or contradicted by the premises.\n\n"
    #     "You should also flag where and how badly the rationale fails, but allow trivial unused remarks to be overridden.\n\n"

    #     "Input (JSON):\n"
    #     "{\n"
    #     '  "premises":   "<string>",\n'
    #     '  "conclusion": "<string>",\n'
    #     '  "rationale":  "<string>",\n'
    #     '  "label":      "<string>",\n'
    #     '  "predict":    "<string>"\n'
    #     "}\n\n"
    #     "Output (JSON):\n"
    #     "{\n"
    #     '  "faithful":         true | false,\n'
    #     '  "error_type":       "<missing branches | invalid inference | invalid converse/contraposition | shortcut problem | factual misquote>",\n'
    #     '  "error_location":   "<e.g. Step 3, Clause 2>",\n'
    #     '  "override":         true | false,\n'
    #     '  "analysis":         "<brief summary of which steps were ungrounded, unsound, incomplete, or shortcut>"\n'
    #     "}\n\n"
    #     "Note: If multiple error types apply, list them all in the `error_type` field, separated by commas.\n\n"
    #     "Input:\n"
    # )

    system_prompt = (
        "You must determine whether a rationale faithfully justifies the truth value of a conclusion given a set of premises.\n\n"
        "“Faithful” means all and only the steps actually used in deriving the conclusion:\n"
        "- are grounded in the given premises or prior derived steps,\n"
        "- apply valid inference rules (no illicit converse or contraposition),\n"
        "- cover every disjunction branch or quantifier case,\n"
        "- use no unstated assumptions, external knowledge, or background commonsense,\n"
        "- and correctly assess whether the conclusion is supported or contradicted by the premises.\n\n"
        "You must also diagnose where and how the rationale fails when it is unfaithful, allowing trivial unused remarks to be overridden.\n\n"
        
        "Error Types:\n"
        "- Missing Branch: Failing to exhaustively consider all branches of a disjunction, conditionals, or quantified cases.\n"
        "- Invalid Converse: Illicitly reversing the direction of a conditional (e.g., mistaking 'A → B' for 'B → A').\n"
        "- Commonsense Injection: Using external background knowledge or commonsense not entailed or implied by the premises.\n"
        "- Factual Misquote: Misrepresenting, distorting, or misquoting the explicit content of the premises.\n\n"
        
        "Input (JSON):\n"
        "{\n"
        '  "premises":   "<string>",\n'
        '  "conclusion": "<string>",\n'
        '  "rationale":  "<string>",\n'
        '  "label":      "<string>",\n'
        '  "predict":    "<string>"\n'
        "}\n\n"
        
        "Output (JSON):\n"
        "{\n"
        '  "faithful":         true | false,\n'
        '  "error_type":       "<missing branch | invalid converse | commonsense injection | factual misquote>",\n'
        '  "error_location":   "<e.g., Step 3, Clause 2>",\n'
        '  "override":         true | false,\n'
        '  "analysis":         "<brief summary explaining why the reasoning is faithful or unfaithful, citing specific logical failures>"\n'
        "}\n\n"
        
        "Notes:\n"
        "- If multiple error types apply, list them all separated by commas.\n"
        "- Always identify the first point in the rationale where the faithfulness failure occurs.\n"
        "- Be concise, precise, and consistent in your labeling.\n\n"
        "Input:\n"
    )



    user_input = {
        "premises": premises,
        "conclusion": conclusion,
        "rationale": rationale,
        "label": label,
        "predict": predict
    }

    print(model)

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": json.dumps(user_input, ensure_ascii=False, indent=2)}
        ],
        temperature=1.0,
        max_completion_tokens=4096
    )

    text = resp.choices[0].message.content.strip()
    print(text)
    print(resp.usage)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "error": "JSON parse error",
            "raw_response": text
        }

# # -----------------------------------------------------------------------------
# # 批量处理函数
# # -----------------------------------------------------------------------------
# def batch_evaluate(input_path: str,
#                    output_path: str,
#                    model: str = "gpt-4o"):
#     # 1. 读取输入文件（支持纯 JSON 数组或逐行 JSONL）
#     with open(input_path, 'r', encoding='utf-8') as f:
#         first_char = f.read(1)
#         f.seek(0)
#         if first_char == '[':
#             records = json.load(f)  # JSON 列表
#         else:
#             records = [json.loads(line) for line in f if line.strip()]

#     # records = records[:1]

#     results = []
#     for idx, item in enumerate(records, 1):
#         premises   = item.get("premises", "")
#         # 支持两种字段名：conclusion 或 conclusions
#         conclusion = item.get("conclusion") or item.get("conclusions", "")
#         rationale  = item.get("rationale", "")
#         label      = item.get("label", "")
#         predict    = item.get("predict", "")

#         print(f"Evaluating item {idx}/{len(records)}...")
#         eval_output = evaluate_rationale(
#             premises, conclusion, rationale, label, predict, model=model
#         )

#         # 合并原记录与评估结果
#         merged = {
#             **item,
#             "evaluation": eval_output
#         }
#         results.append(merged)

#     # 2. 将结果写入输出文件（JSON 数组）
#     with open(output_path, 'w', encoding='utf-8') as f_out:
#         json.dump(results, f_out, ensure_ascii=False, indent=2)

#     print(f"Done! Results written to {output_path}")

def batch_evaluate(input_path: str,
                   output_path: str,
                   model: str = "o4-mini"):
    # 1. 读取所有输入条目
    with open(input_path, 'r', encoding='utf-8') as f:
        first = f.read(1)
        f.seek(0)
        if first == '[':
            records = json.load(f)
        else:
            records = [json.loads(line) for line in f if line.strip()]

    

    total = len(records)

    # 2. 如果已经有部分输出，先加载已处理数量
    processed = 0
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as outf:
            # 每行一个 JSON
            for _ in outf:
                processed += 1

    print(f"Total records: {total}, already processed: {processed}")

    # 3. 打开输出文件（追加模式）
    with open(output_path, 'a', encoding='utf-8') as outf:
        # 从 processed 开始，逐条处理
        for idx in range(processed, total):
            item = records[idx]
            premises   = item["premises"]
            conclusion = item.get("conclusion", item.get("conclusions", ""))
            rationale  = item["rationale"]
            label      = item["label"]
            predict    = item["predict"]

            print(f"[{idx+1}/{total}] Processing…", end=' ')
            eval_res = evaluate_rationale(
                premises, conclusion, rationale, label, predict, model=model
            )
            # 合并并写一行
            merged = {**item, "evaluation": eval_res}
            outf.write(json.dumps(merged, ensure_ascii=False) + "\n")
            outf.flush()  # 确保立即写盘
            print("done")

    print("All done.")

# -----------------------------------------------------------------------------
# 脚本入口
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="批量调用 OpenAI API 对推理链进行忠实性评估"
    )
    parser.add_argument(
        "--input", "-i", required=True, default='QWEN_FL.jsonl',
        help="输入文件路径（JSON 列表或 JSONL）"
    )
    parser.add_argument(
        "--output", "-o", default="QWEN_FL_output.json",
        help="输出文件路径（JSON 列表格式）"
    )
    parser.add_argument(
        "--model", "-m", default="o4-mini-2025-04-16",
        help="OpenAI 模型名称"
    )
    args = parser.parse_args()

    batch_evaluate(args.input, args.output, model=args.model)

