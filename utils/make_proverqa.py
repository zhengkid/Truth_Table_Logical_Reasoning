from datasets import load_dataset
import re
import pandas as pd

# 加载 valid 数据集
dataset = load_dataset("opendatalab/ProverQA", split="validation")
print(dataset)
def process_sample(sample):
    # 从 question 中提取 conclusion
    pattern = r"Based on the above information, is the following statement true, false, or uncertain\?\s*(.*)"
    match = re.search(pattern, sample["question"])
    conclusion = match.group(1).strip() if match else sample["question"].strip()
    
    # 处理 context：
    # 按句点切分，过滤空字符串，每个句子末尾补充句点，再用换行符连接成新的 premises 字段
    sentences = [s.strip() for s in sample["context"].split('.') if s.strip()]
    premises = "\n".join([s + '.' for s in sentences])
    
    # 从 answer 中提取 label，并映射：A -> True, B -> False, C -> Uncertain
    mapping = {"A": "True", "B": "False", "C": "Uncertain"}
    label = mapping.get(sample["answer"].strip(), sample["answer"].strip())
    
    # 添加新字段到 sample 中
    sample["conclusion"] = conclusion
    sample["premises"] = premises
    sample["label"] = label
    return sample

# 对数据集进行 map 处理
#processed_dataset = dataset.map(process_sample)

# 如果需要转换为 pandas DataFrame 查看，可以执行：
#df = processed_dataset.to_pandas()
#print(df.head())

# 推送数据集到您的私有 Hugging Face 仓库
# 请确保已经登录 Hugging Face 或设置了 HUGGINGFACE_HUB_TOKEN 环境变量
#processed_dataset.push_to_hub("TongZheng1999/ProverQA", private=True)

