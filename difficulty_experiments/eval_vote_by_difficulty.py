import json
import argparse
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import Counter


def evaluate(preds):
    labels = [x["label"] for x in preds]
    predictions = [x["predict"] for x in preds]
    return {
        "accuracy": accuracy_score(labels, predictions) * 100,
        "f1": f1_score(labels, predictions, average="macro"),
        "precision": precision_score(labels, predictions, average="macro"),
        "recall": recall_score(labels, predictions, average="macro"),
        "count": len(labels)
    }


def voting_accuracy(nl_preds, code_preds, tt_preds):
    # 构建结论到预测的映射
    nl_map = {x["conclusions"]: x["predict"] for x in nl_preds}
    code_map = {x["conclusions"]: x["predict"] for x in code_preds}
    tt_map = {x["conclusions"]: x["predict"] for x in tt_preds}
    label_map = {x["conclusions"]: x["label"] for x in nl_preds}  # 假设label一致

    correct = 0
    total = 0

    for cid in label_map:
        votes = [nl_map.get(cid), code_map.get(cid), tt_map.get(cid)]
        vote_counts = Counter(votes)
        if not vote_counts:
            continue
        majority_vote, _ = vote_counts.most_common(1)[0]
        if majority_vote == label_map[cid]:
            correct += 1
        total += 1

    return 100 *correct / total if total > 0 else 0.0


def main(nl_path, code_path, tt_path, easy_dataset_name, difficult_dataset_name):
    # 加载预测结果
    with open(nl_path) as f:
        nl_data = json.load(f)
    with open(code_path) as f:
        code_data = json.load(f)
    with open(tt_path) as f:
        tt_data = json.load(f)

    # 加载 easy 和 difficult 子集
    print(f"\U0001F4C5 Loading easy dataset from: {easy_dataset_name}")
    easy_dataset = load_dataset(easy_dataset_name, split="validation")
    print(f"\U0001F4C5 Loading difficult dataset from: {difficult_dataset_name}")
    difficult_dataset = load_dataset(difficult_dataset_name, split="validation")

    easy_set = set(item["conclusion"] for item in easy_dataset)
    difficult_set = set(item["conclusion"] for item in difficult_dataset)

    # 分割子集
    def filter_by_conclusion(data, subset_set):
        return [x for x in data if x["conclusions"] in subset_set]

    easy_nl = filter_by_conclusion(nl_data, easy_set)
    easy_code = filter_by_conclusion(code_data, easy_set)
    easy_tt = filter_by_conclusion(tt_data, easy_set)

    diff_nl = filter_by_conclusion(nl_data, difficult_set)
    diff_code = filter_by_conclusion(code_data, difficult_set)
    diff_tt = filter_by_conclusion(tt_data, difficult_set)

    # 评估
    print("\n\U0001F4CA Evaluation Results - Easy Subset")
    print("NL:", evaluate(easy_nl))
    print("Code:", evaluate(easy_code))
    print("Truth Table:", evaluate(easy_tt))
    print("Voting Accuracy:", voting_accuracy(easy_nl, easy_code, easy_tt))

    print("\n\U0001F4CA Evaluation Results - Difficult Subset")
    print("NL:", evaluate(diff_nl))
    print("Code:", evaluate(diff_code))
    print("Truth Table:", evaluate(diff_tt))
    print("Voting Accuracy:", voting_accuracy(diff_nl, diff_code, diff_tt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nl_path", type=str, required=True, help="Path to NL prediction JSON file")
    parser.add_argument("--code_path", type=str, required=True, help="Path to Code prediction JSON file")
    parser.add_argument("--tt_path", type=str, required=True, help="Path to Truth Table prediction JSON file")
    parser.add_argument("--easy_dataset_name", type=str, required=True, help="HuggingFace dataset name for easy subset")
    parser.add_argument("--difficult_dataset_name", type=str, required=True, help="HuggingFace dataset name for difficult subset")
    args = parser.parse_args()

    main(args.nl_path, args.code_path, args.tt_path, args.easy_dataset_name, args.difficult_dataset_name)

