from datasets import load_dataset, Dataset, DatasetDict
import random

# 固定随机种子
random.seed(42)

# 加载完整数据集
full_dataset = load_dataset("TongZheng1999/ProofWriter")
train_dataset = full_dataset["train"]
val_dataset = full_dataset["validation"]

# 三分类标签
LABELS = ["True", "False", "Uncertain"]

# 平衡采样函数（确保总数精确、类别尽量均衡）
def balanced_sample(dataset, n_total):
    base = n_total // 3
    remainder = n_total % 3
    per_class_list = [base + 1 if i < remainder else base for i in range(3)]
    
    sampled = []
    for label, n in zip(LABELS, per_class_list):
        label_subset = dataset.filter(lambda x: x['label'] == label)
        sampled.extend(random.sample(list(label_subset), n))
    
    return Dataset.from_list(sampled)

# 采样+拼合+上传
subset_sizes = [300, 600, 1000]
for size in subset_sizes:
    sampled_train = balanced_sample(train_dataset, size)
    dataset_dict = DatasetDict({
        "train": sampled_train,
        "validation": val_dataset
    })
    dataset_dict.push_to_hub(f"TongZheng1999/ProofWriter-{size}")

