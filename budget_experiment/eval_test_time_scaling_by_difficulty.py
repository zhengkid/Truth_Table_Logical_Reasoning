# 整合了多模态一致性检查、Huggingface子集支持、评估与保存的完整脚本

import json
import random
import numpy as np
import argparse
from datasets import load_dataset

def evaluate_prediction(prediction, label):
    return prediction.strip().lower() == label.strip().lower()

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def sample_pass_metric_baseline(responses, label, sample_size):
    if len(responses) < sample_size:
        raise ValueError("响应数量不足以进行抽样")
    sample = random.sample(responses, sample_size)
    return int(any(evaluate_prediction(response, label) for response in sample))

def run_evaluation_baseline(data, sample_size, num_trials):
    trial_results = []
    for _ in range(num_trials):
        correct_count = sum(
            sample_pass_metric_baseline(item['predictions'], item['label'], sample_size)
            for item in data
        )
        trial_results.append((correct_count / len(data)) * 100)
    return np.mean(trial_results), np.std(trial_results), trial_results

def sample_pass_metric_multimode(r1, r2, r3, label, sample_size_total):
    s = sample_size_total // 3
    combined = random.sample(r1, s) + random.sample(r2, s) + random.sample(r3, s)
    return int(any(evaluate_prediction(r, label) for r in combined))

def run_evaluation_multimode(d1, d2, d3, sample_size_total, num_trials):
    trial_results = []
    for _ in range(num_trials):
        correct_count = 0
        for i in range(len(d1)):
            r1, r2, r3 = d1[i]['predictions'], d2[i]['predictions'], d3[i]['predictions']
            label = d1[i]['label']
            correct_count += sample_pass_metric_multimode(r1, r2, r3, label, sample_size_total)
        trial_results.append((correct_count / len(d1)) * 100)
    return np.mean(trial_results), np.std(trial_results), trial_results

def run_evaluation_two_mode(d1, d2, sample_size_total, num_trials):
    s = sample_size_total // 2
    trial_results = []
    for _ in range(num_trials):
        correct_count = 0
        for i in range(len(d1)):
            r1, r2 = random.sample(d1[i]['predictions'], s), random.sample(d2[i]['predictions'], s)
            label = d1[i]['label']
            correct_count += int(any(evaluate_prediction(r, label) for r in (r1 + r2)))
        trial_results.append((correct_count / len(d1)) * 100)
    return np.mean(trial_results), np.std(trial_results), trial_results

def save_results_to_file(filename, results):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("% sample_size mean_accuracy std_accuracy\n")
        for s, m, sd in results:
            f.write(f"{s} {m:.3f} {sd:.3f}\n")
    print(f"[Saved] {filename}")

def load_subset_conclusions_from_hf(dataset_name, split="validation"):
    ds = load_dataset(dataset_name, split=split)
    return set(example['conclusion'] for example in ds)

def filter_data_by_conclusion(data, conclusion_set):
    return [item for item in data if item.get("conclusions") in conclusion_set]

def check_conclusion_alignment(datasets, name=""):
    n = len(datasets[0])
    for i in range(n):
        conclusions = [ds[i].get("conclusion") for ds in datasets]
        if not all(c == conclusions[0] for c in conclusions):
            raise ValueError(f"[错误] {name} 第 {i} 条样本 conclusion 不一致: {conclusions}")
    print(f"[检查通过] {name}: 所有样本结论一致，共 {n} 条")

def evaluate_on_subset(data_code, data_nl, data_tt, sample_sizes, num_trials, output_prefix):
    results = {
        'code': [], 'nl': [], 'truth_table': [],
        'mix3': [], 'code+nl': [], 'code+truth_table': [], 'nl+truth_table': []
    }
    for s in sample_sizes:
        m1, s1, _ = run_evaluation_baseline(data_code, s, num_trials)
        m2, s2, _ = run_evaluation_baseline(data_nl, s, num_trials)
        m3, s3, _ = run_evaluation_baseline(data_tt, s, num_trials)
        m_mix3, s_mix3, _ = run_evaluation_multimode(data_code, data_nl, data_tt, s, num_trials)
        m12, s12, _ = run_evaluation_two_mode(data_code, data_nl, s, num_trials)
        m13, s13, _ = run_evaluation_two_mode(data_code, data_tt, s, num_trials)
        m23, s23, _ = run_evaluation_two_mode(data_nl, data_tt, s, num_trials)
        results['code'].append((s, m1, s1))
        results['nl'].append((s, m2, s2))
        results['truth_table'].append((s, m3, s3))
        results['mix3'].append((s, m_mix3, s_mix3))
        results['code+nl'].append((s, m12, s12))
        results['code+truth_table'].append((s, m13, s13))
        results['nl+truth_table'].append((s, m23, s23))
    for k, v in results.items():
        save_results_to_file(f"{output_prefix}_{k}.dat", v)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method_code', type=str)
    parser.add_argument('--method_nl', type=str)
    parser.add_argument('--method_truth_table', type=str)
    parser.add_argument('--subset_easy_hf', type=str)
    parser.add_argument('--subset_hard_hf', type=str)
    parser.add_argument('--num_trials', type=int, default=10)
    args = parser.parse_args()

    sample_sizes = [3, 6, 12, 24, 48, 72, 96, 120]
    
    random.seed(42)
    np.random.seed(42)


    # 加载 method 模式数据
    method_code = load_data(args.method_code)
    method_nl = load_data(args.method_nl)
    method_tt = load_data(args.method_truth_table)
    check_conclusion_alignment([method_code, method_nl, method_tt], "method 全体数据")

    # 加载子集
    easy_set = load_subset_conclusions_from_hf(args.subset_easy_hf)
    hard_set = load_subset_conclusions_from_hf(args.subset_hard_hf)

    # 筛选子集
    easy_code = filter_data_by_conclusion(method_code, easy_set)
    easy_nl = filter_data_by_conclusion(method_nl, easy_set)
    easy_tt = filter_data_by_conclusion(method_tt, easy_set)
    hard_code = filter_data_by_conclusion(method_code, hard_set)
    hard_nl = filter_data_by_conclusion(method_nl, hard_set)
    hard_tt = filter_data_by_conclusion(method_tt, hard_set)

    check_conclusion_alignment([easy_code, easy_nl, easy_tt], "easy 子集")
    check_conclusion_alignment([hard_code, hard_nl, hard_tt], "hard 子集")

    # 评估
    print("[评估] Easy 子集")
    evaluate_on_subset(easy_code, easy_nl, easy_tt, sample_sizes, args.num_trials, "method_easy")
    print("[评估] Hard 子集")
    evaluate_on_subset(hard_code, hard_nl, hard_tt, sample_sizes, args.num_trials, "method_hard")

if __name__ == "__main__":
    main()

