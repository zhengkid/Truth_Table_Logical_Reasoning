#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import json

def main():
    """
    用法：
    python overlap_correct_conclusions.py file1.json file2.json file3.json

    说明：
    - 假设每个 JSON 文件的最外层结构是一个列表。
    - 列表中的每个元素至少包含 "conclusions", "label", "predict" 这几个字段。
    - 我们将用 'conclusions' 作为唯一标识，收集“预测正确”的记录，做交并差分析。
    """

    # 检查命令行参数
    if len(sys.argv) != 4:
        print("用法：python overlap_correct_conclusions.py file1.json file2.json file3.json")
        sys.exit(1)

    file1, file2, file3 = sys.argv[1], sys.argv[2], sys.argv[3]

    # 读取三个 JSON 文件
    data_list1 = read_json(file1)
    data_list2 = read_json(file2)
    data_list3 = read_json(file3)
    print(len(data_list1))
    # 提取“预测正确”且用 'conclusions' 作为“标识”的集合
    set_correct_1 = get_correct_conclusions(data_list1)
    set_correct_2 = get_correct_conclusions(data_list2)
    set_correct_3 = get_correct_conclusions(data_list3)

    # 计算交集（同时出现在 1、2、3 中）
    intersect_123 = set_correct_1 & set_correct_2 & set_correct_3
    
    intersect_12 = set_correct_1 & set_correct_2 
    
    intersect_13 = set_correct_1 & set_correct_3
    
    intersect_23 = set_correct_2 & set_correct_3

    print("1 in 2",len(intersect_12))
    print("1 in 3",len(intersect_13))
    print("2 in 3",len(intersect_23))

    # 计算并集（只要出现在 1、2、3 中任意一个就算）
    union_123 = set_correct_1 | set_correct_2 | set_correct_3
    
    union_12 = set_correct_1 | set_correct_2

    union_13 = set_correct_1 | set_correct_3

    union_23 = set_correct_2 | set_correct_3

    # 计算差集：只在 file1 中正确、但不在 file2 或 file3 中
    only_in_file1 = set_correct_1 - (set_correct_2 | set_correct_3)
    # 也可以依次计算 only_in_file2, only_in_file3 等
    
    # 计算差集：只在 file1 中正确、但不在 file2 或 file3 中
    only_in_file2 = set_correct_2 - (set_correct_1 | set_correct_3)
    # 也可以依次计算 only_in_file2, only_in_file3 等

    # 计算差集：只在 file1 中正确、但不在 file2 或 file3 中
    only_in_file3 = set_correct_3 - (set_correct_1 | set_correct_2)
    # 也可以依次计算 only_in_file2, only_in_file3 等

    print("=== 统计信息 ===")
    print(f"file1 中 预测正确(conclusions) 个数: {len(set_correct_1)}")
    print(f"file2 中 预测正确(conclusions) 个数: {len(set_correct_2)}")
    print(f"file3 中 预测正确(conclusions) 个数: {len(set_correct_3)}")

    print("\n=== 交集：同时在3个文件中预测正确的 'conclusions' ===")
    #print(intersect_123)
    print("个数:", len(intersect_123))

    print("\n=== 并集：只要在任意1个文件中预测正确的 'conclusions' ===")
    #print(union_123)
    print("个数:", len(union_123))
    
    
    print("\n=== 并集：只要在任意1和2文件中预测正确的 'conclusions' ===")
    #print(union_123)
    print("个数:", len(union_12))

    print("\n=== 并集：只要在任意2和3文件中预测正确的 'conclusions' ===")
    #print(union_123)
    print("个数:", len(union_23))

    print("\n=== 并集：只要在任意1和3文件中预测正确的 'conclusions' ===")
    #print(union_123)
    print("个数:", len(union_13))


    print("\n=== 只在 file1 中正确 (而不在 file2 或 file3) 的 'conclusions' ===")
    #print(only_in_file1)
    print("个数:", len(only_in_file1))
   

    print("\n=== 只在 file2 中正确 (而不在 file1 或 file3) 的 'conclusions' ===")
    #print(only_in_file2)
    print("个数:", len(only_in_file2))

    print("\n=== 只在 file3 中正确 (而不在 file1 或 file2) 的 'conclusions' ===")
    #print(only_in_file3)
    print("个数:", len(only_in_file3))
    print(only_in_file3)
def read_json(file_path):
    """读取并解析 JSON 文件，返回一个 Python 列表或字典。"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_correct_conclusions(data_list):
    """
    遍历列表中每条记录：
    - 若 label == predict， 则把 record['conclusions'] 放入集合。
    """
    correct_conclusion_set = set()
    for record in data_list:
        label = record.get("label", "").strip()
        predict = record.get("predict", "").strip()
        conclusion_text = record.get("conclusions", "").strip()
        premise = record.get("premises", "").strip()
        conclusion_text = conclusion_text + "\nPremises:\n" + premise
        # 只在 label == predict 且 conclusion_text 不为空时，将其加入集合
        if label == predict and conclusion_text:
            correct_conclusion_set.add(conclusion_text)
    
    return correct_conclusion_set


if __name__ == "__main__":
    main()

