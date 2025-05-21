import re
import json
from datasets import load_dataset, DatasetDict

# 把多选项字母映射到对应的 label 字符串
LETTER2LABEL = {
    "A": "True",
    "B": "False",
    "C": "Uncertain"
}

def extract_predict_from_response(response: str) -> str:
    """
    从 response 字段中提取模型的预测字母 (A/B/C)。
    """
    m = re.search(r"<answer>.*?\(([ABC])\)", response, flags=re.DOTALL)
    return m.group(1) if m else None

def main():
    # 1. 加载 HF 数据集
    ds = load_dataset("TongZheng1999/o4-FL-data", split="train")

    # 2. 过滤函数：非空 response，能提取到字母预测，并且映射后与 label 相同
    def filter_fn(example):
        resp = example.get("response", "").strip()
        if not resp:
            return False

        letter = extract_predict_from_response(resp)
        if letter is None or letter not in LETTER2LABEL:
            return False

        pred_label = LETTER2LABEL[letter]
        # 只保留预测标签与 gold label 完全一致的
        return pred_label == example.get("label")

    # 3. 执行过滤（并行）
    ds_filtered = ds.filter(filter_fn, num_proc=4)

    # 4. 写本地 JSONL
    #with open("filtered_data.jsonl", "w", encoding="utf-8") as fout:
    #    for ex in ds_filtered:
    #        fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"原始样本数: {len(ds)}, 过滤后样本数: {len(ds_filtered)}")

    # （可选）推到 HF
    DatasetDict({"train": ds_filtered}).push_to_hub("TongZheng1999/o4-FL-data-filtered", private=True)

if __name__ == "__main__":
    main()

