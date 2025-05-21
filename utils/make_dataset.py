import json
from datasets import Dataset, DatasetDict

def extract_premises(context):
    # 先按 ". " 分割为句子
    sentences = context.split('. ')
    processed_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            # 如果当前句子末尾没有句号，则添加句号
            if not s.endswith('.'):
                s += '.'
            processed_sentences.append(s)
    # 用换行符将句子连接起来
    return "\n".join(processed_sentences)

def extract_conclusion(question):
    # 按问号分割，取问号后的部分（如果存在）
    parts = question.split("?", 1)
    if len(parts) > 1:
        return parts[1].strip()
    return question

def extract_label(options, answer_key):
    # 遍历 options，查找以 answer_key + ")" 开头的选项
    for opt in options:
        if opt.startswith(f"{answer_key})"):
            # 提取选项文本，去除前缀，例如 "A) " 的部分
            label_text = opt.split(")", 1)[1].strip()
            # 如果文本为 Unknown，则替换为 Uncertain
            if label_text.lower() == "unknown":
                label_text = "Uncertain"
            return label_text
    return ""





with open('dev.json', 'r', encoding='utf-8') as file:
    test_data = json.load(file)

# 对每个记录添加新键
for record in test_data:
    # premises：将 context 中的句子用换行符连接，并确保每个句子后有句号
    record['premises'] = extract_premises(record.get('context', ""))
    # conclusion：取 question 中问号后的部分
    record['conclusion'] = extract_conclusion(record.get('question', ""))
    # label：根据 answer 对应的选项内容，Unknown 替换为 Uncertain
    record['label'] = extract_label(record.get('options', []), record.get('answer', ""))










# 构造 Dataset 和 DatasetDict

ds_test = Dataset.from_list(test_data)
ds_dict = DatasetDict({"validation": ds_test})



# 推送到 Hugging Face Hub，仓库将创建为私有仓库
ds_dict.push_to_hub(
    repo_id="TongZheng1999/ProntoQA",  # 替换为你的仓库路径
    private=True
)
