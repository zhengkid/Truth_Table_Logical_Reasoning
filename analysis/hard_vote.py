#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
from collections import Counter

def main():
    if len(sys.argv) != 4:
        print("用法：python hard_vote.py file1.json file2.json file3.json")
        sys.exit(1)

    file1, file2, file3 = sys.argv[1], sys.argv[2], sys.argv[3]

    data1 = read_json(file1)
    data2 = read_json(file2)
    data3 = read_json(file3)

    # 若三份文件的顺序、数量都一致
    if not (len(data1) == len(data2) == len(data3)):
        print("三个文件记录数量不一致，请检查！")
        sys.exit(1)

    results = []
    correct_count = 0  # 统计投票后正确的条数
    total_samples = len(data1)

    for i in range(total_samples):
        record1 = data1[i]
        record2 = data2[i]
        record3 = data3[i]

        # 取三个模型的预测
        pred1 = record1.get("predict", None)
        pred2 = record2.get("predict", None)
        pred3 = record3.get("predict", None)

        # 假设每条记录都有真实标签 label（至少在其中一个或三个文件里）
        # 这里就示例从 file1 里取 label，你也可以看三份文件是否都相同
        ground_truth = record1.get("label", None)

        majority_pred, tie_info = majority_vote(pred1, pred2, pred3)
        print("g", ground_truth)
        # 如果存在真实标签，就可以判断投票是否正确
        if ground_truth is not None:
            if majority_pred == ground_truth:
                correct_count += 1

        result_item = {
            "id": record1.get("id", f"item_{i}"),
            "predictions": [pred1, pred2, pred3],
            "majority_vote": majority_pred,
            "label": ground_truth
        }
        if tie_info:
            result_item["tie_info"] = tie_info

        results.append(result_item)

    # 计算并打印准确率（若 ground_truth 都存在）
    print(correct_count)
    accuracy = correct_count / total_samples if total_samples > 0 else 0
    print(f"投票后的准确率：{accuracy:.2%}")

    # 如需将结果保存为 JSON
    # with open("voting_result.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def majority_vote(pred1, pred2, pred3):
    preds = [pred1, pred2, pred3]
    counter = Counter(preds)
    top_label, top_count = counter.most_common(1)[0]
    print(preds)
    tie_info = ""
    all_sorted = counter.most_common()

    # 三方各不相同（例如 A,B,C）
    print(all_sorted)
    if len(all_sorted) == 3 and all_sorted[0][1] == 1:
        tie_info = f"Three-way tie among {preds}"
        # 自定义打破平局，这里示例：按字母顺序选最小
        majority_label = sorted(preds)[0]
        return (majority_label, tie_info)
    print(top_label)
    return (top_label, tie_info)

if __name__ == "__main__":
    main()

