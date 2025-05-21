import json
import re
from datasets import Dataset, DatasetDict

# 加载 easy.json 数据文件
with open("easy.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def process_sample(sample):
    # 从 question 中提取 conclusion
    pattern = r"Based on the above information, is the following statement true, false, or uncertain\?\s*(.*)"
    match = re.search(pattern, sample["question"])
    conclusion = match.group(1).strip() if match else sample["question"].strip()
    
    # 处理 context：
    # 按句点切分，过滤掉空字符串，每个句子末尾补上句点，然后用换行符连接
    sentences = [s.strip() for s in sample["context"].split('.') if s.strip()]
    premises = "\n".join([s + '.' for s in sentences])
    
    # 从 answer 中提取 label，并映射：A -> True, B -> False, C -> Uncertain
    mapping = {"A": "True", "B": "False", "C": "Uncertain"}
    label = mapping.get(sample["answer"].strip(), sample["answer"].strip())
    
    # 添加新字段
    sample["conclusion"] = conclusion
    sample["premises"] = premises
    sample["label"] = label
    return sample

# 对数据进行预处理
processed_data = [process_sample(sample) for sample in data]

# 构造 Dataset 和 DatasetDict
ds_valid = Dataset.from_list(processed_data)
ds_dict = DatasetDict({"validation": ds_valid})

# 推送到 Hugging Face Hub，仓库将创建为私有仓库
ds_dict.push_to_hub(
    repo_id="TongZheng1999/ProverQA-Easy",  # 替换为您的仓库路径
    private=True
)

