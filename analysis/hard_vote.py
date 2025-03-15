#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json
from collections import Counter

def main():
    """
    用法：
    python hard_vote.py file1.json file2.json file3.json

    假设：
    1. file1.json, file2.json, file3.json中记录的数量相同，并且顺序一一对应。
    2. 每个文件都是一个列表，列表元素形如：
       {
         "id": ...,
         "predict": ...  # 或者是其他字段名, 如 "label" 或 "prediction"
         ...
       }
    3. 如果数据结构不同，请自行调整代码中获取预测的逻辑
    """

    if len(sys.argv) != 4:
        print("用法：python hard_vote.py file1.json file2.json file3.json")
        sys.exit(1)

    file1, file2, file3 = sys.argv[1], sys.argv[2], sys.argv[3]

    # 读取三个 JSON 文件 (每个都应是一个列表)
    data1 = read_json(file1)
    data2 = read_json(file2)
    data3 = read_json(file3)

    # 简单检查记录数一致性（可以根据需求改成基于id匹配）
    if not (len(data1) == len(data2) == len(data3)):
        print("三个文件记录数量不一致，请确保它们对应相同样本！")
        sys.exit(1)

    # 用于保存最终投票结果
    results = []

    for i in range(len(data1)):
        # 从每个文件中取第i条记录
        record1 = data1[i]
        record2 = data2[i]
        record3 = data3[i]

        # 取出各自的预测，字段名可以自行修改
        pred1 = record1.get("predict", None)
        pred2 = record2.get("predict", None)
        pred3 = record3.get("predict", None)

        # 用 majority_vote 函数得到投票结果
        majority_pred, tie_info = majority_vote(pred1, pred2, pred3)

        # 打印或收集结果
        # 可以把原信息、投票结果放进一个字典再存
        result_item = {
            "id": record1.get("id", f"item_{i}"),
            "predictions": [pred1, pred2, pred3],
            "majority_vote": majority_pred
        }

        if tie_info:  # 如果有平局信息，可以备注
            result_item["tie_info"] = tie_info

        results.append(result_item)

    # 在这里你可以把结果写入新的 JSON 文件，也可以直接打印
    # 下面演示把结果打印到屏幕
    for r in results:
        print(r)


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def majority_vote(pred1, pred2, pred3):
    """
    对三份预测做硬投票。
    返回 (majority_label, tie_info)

    - majority_label: 票数最多的标签 (str)
    - tie_info: 如果出现三方各不相同或有并列，记录一下具体情况。若无平局则返回空字符串。
    """

    preds = [pred1, pred2, pred3]
    counter = Counter(preds)
    # Counter 示例：Counter({'A':2, 'B':1}) 或 Counter({'A':1, 'B':1, 'C':1})

    # 找到出现次数最多的标签
    # most_common(1) 返回 [(label, count)]，取第一个即可
    top_label, top_count = counter.most_common(1)[0]

    # 查看是否有平局
    # 例如：counter = {'A':1, 'B':1, 'C':1}, 三个都出现1次
    # 或 counter = {'A':2, 'B':2}, 这种并列
    tie_info = ""

    # 全部的 (标签, 次数) 已按 count 递减排序
    all_sorted = counter.most_common()

    # 检测“3个预测各不相同” -> 3种标签，各计数1
    if len(all_sorted) == 3 and all_sorted[0][1] == 1:
        # 说明 pred1, pred2, pred3 全都不一样
        tie_info = f"Three-way tie among {preds}"
        # 可以自定义处理，比如选alphabetical最小的
        majority_label = sorted(preds)[0]  # 例如：按字母顺序选最小
        return (majority_label, tie_info)

    # 检测“并列第一”
    # 如果第二名的count和第一名相等，就说明有并列
    if len(all_sorted) >= 2 and all_sorted[1][1] == top_count:
        tie_info = f"Multi-way tie for top: {all_sorted}"
        # 自定义如何打破平局，这里也可选alphabetical最小
        # 也可根据对各模型信任度进行加权处理
        # 这里演示简单做法：取字母顺序排最小的label
        candidates = [t[0] for t in all_sorted if t[1] == top_count]
        majority_label = sorted(candidates)[0]
        return (majority_label, tie_info)

    # 如果无平局，则直接返回
    return (top_label, tie_info)


if __name__ == "__main__":
    main()

