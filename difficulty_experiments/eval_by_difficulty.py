import json
import argparse
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate(preds):
    labels = [x["label"] for x in preds]
    predictions = [x["predict"] for x in preds]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions, average="macro"),
        "count": len(labels)
    }

def main(prediction_path, easy_dataset_name, difficult_dataset_name):
    # 加载预测文件
    with open(prediction_path) as f:
        pred_data = json.load(f)

    # 加载 easy 和 difficult 子集 conclusion 字段
    print(f"📥 Loading easy dataset from: {easy_dataset_name}")
    easy_dataset = load_dataset(easy_dataset_name, split="validation")
    print(f"📥 Loading difficult dataset from: {difficult_dataset_name}")
    difficult_dataset = load_dataset(difficult_dataset_name, split="validation")
    
    # 抽取 conclusions 字段作为唯一标识
    easy_set = set([item["conclusion"] for item in easy_dataset])
    difficult_set = set([item["conclusion"] for item in difficult_dataset])

    # 分配子集
    easy_preds = [x for x in pred_data if x["conclusions"] in easy_set]
    difficult_preds = [x for x in pred_data if x["conclusions"] in difficult_set]

    # 打印评估结果
    print("\\n📊 Evaluation Results")
    print("Easy Subset:")
    print(evaluate(easy_preds))
    print("\\nDifficult Subset:")
    print(evaluate(difficult_preds))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_path", type=str, required=True, help="Path to the prediction JSON file")
    parser.add_argument("--easy_dataset_name", type=str, required=True, help="HuggingFace dataset name for easy subset")
    parser.add_argument("--difficult_dataset_name", type=str, required=True, help="HuggingFace dataset name for difficult subset")
    args = parser.parse_args()

    main(args.prediction_path, args.easy_dataset_name, args.difficult_dataset_name)

