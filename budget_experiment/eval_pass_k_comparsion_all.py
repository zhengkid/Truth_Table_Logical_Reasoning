import json
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

def evaluate_prediction(prediction, label):
    """
    判断单个预测是否正确，这里简单采用大小写不敏感的字符串相等。
    根据实际任务，可以替换为更复杂的评估逻辑。
    """
    return prediction.strip().lower() == label.strip().lower()

def load_data(filepath):
    """
    加载数据文件，假设文件是一个 JSON 列表，
    每个元素为一个字典，包含 "predictions" 和 "label" 字段。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def sample_pass_metric_baseline(responses, label, sample_size):
    """
    基线方法：从 responses 中随机采样 sample_size 个响应，
    如果采样结果中至少有一个正确，则返回 1，否则返回 0。
    """
    if len(responses) < sample_size:
        raise ValueError("响应数量不足以进行抽样")
    sample = random.sample(responses, sample_size)
    return int(any(evaluate_prediction(response, label) for response in sample))

def run_evaluation_baseline(data, sample_size, num_trials):
    """
    基线方法：对所有问题进行 num_trials 次试验，
    每次试验中对每个问题随机采样 sample_size 个响应，
    并统计 pass@sample_size 的准确率。
    
    返回：平均准确率、标准差、以及各次试验的准确率列表。
    """
    trial_results = []
    for trial in range(num_trials):
        correct_count = 0
        for item in data:
            responses = item['predictions']
            label = item['label']
            correct_count += sample_pass_metric_baseline(responses, label, sample_size)
        trial_accuracy = correct_count / len(data)
        trial_accuracy = trial_accuracy * 100
        trial_results.append(trial_accuracy)
    return np.mean(trial_results), np.std(trial_results), trial_results

def sample_pass_metric_multimode(responses1, responses2, responses3, label, sample_size_total):
    """
    Multi-mode 方法：
      - 从每个 mode 的响应列表中分别采样 sample_size_mode 个响应，
      - 合并得到总共 k 个响应（其中 k = sample_size_total），
      - 若这 k 个响应中至少有一个正确，则返回 1，否则返回 0。
      
    这里简单地令 sample_size_mode = sample_size_total // 3。
    """
    sample_size_mode = sample_size_total // 3
    sample1 = random.sample(responses1, sample_size_mode)
    sample2 = random.sample(responses2, sample_size_mode)
    sample3 = random.sample(responses3, sample_size_mode)
    combined_sample = sample1 + sample2 + sample3
    return int(any(evaluate_prediction(response, label) for response in combined_sample))

def run_evaluation_multimode(data1, data2, data3, sample_size_total, num_trials):
    """
    Multi-mode 方法：对所有问题进行 num_trials 次试验，
    每次试验中对每个问题分别从三个 mode 的数据中采样，
    合并后判断是否至少有一个响应正确；
    返回平均准确率、标准差及各次试验的准确率列表。
    """
    num_questions = len(data1)
    trial_results = []
    for trial in range(num_trials):
        correct_count = 0
        for i in range(num_questions):
            responses1 = data1[i]['predictions']
            responses2 = data2[i]['predictions']
            responses3 = data3[i]['predictions']
            label = data1[i]['label']  # 假设三个文件中同一问题的 label 一致
            correct_count += sample_pass_metric_multimode(responses1, responses2, responses3, label, sample_size_total)
        trial_accuracy = correct_count / num_questions
        trial_accuracy = trial_accuracy * 100
        trial_results.append(trial_accuracy)
    return np.mean(trial_results), np.std(trial_results), trial_results

def save_results_to_file(filename, results):
    """
    将结果保存为适用于 LaTeX pgfplots 的数据格式。
    文件每行格式：sample_size mean_accuracy std_accuracy
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("% sample_size mean_accuracy std_accuracy\n")
        for sample_size, mean_acc, std_acc in results:
            f.write(f"{sample_size} {mean_acc} {std_acc}\n")
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Baseline 和 Multi-mode Pass@k 评估 Pipeline, 自动运行多个采样数")
    parser.add_argument('--data_path_baseline', type=str, default=None,
                        help='基线方法的数据文件路径，JSON 列表，每个元素包含 "predictions" 和 "label"')
    parser.add_argument('--data_path_mode1', type=str, default=None,
                        help='Multi-mode 方法 mode1 数据文件路径')
    parser.add_argument('--data_path_mode2', type=str, default=None,
                        help='Multi-mode 方法 mode2 数据文件路径')
    parser.add_argument('--data_path_mode3', type=str, default=None,
                        help='Multi-mode 方法 mode3 数据文件路径')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='重复采样试验次数，用于统计平均准确率和标准差')
    args = parser.parse_args()

    # 预设要测试的采样数
    sample_sizes = [3, 6, 12, 24, 48, 72, 96, 120]

    baseline_results = []
    multimode_results = []

    # 如果提供了基线数据，则加载并运行基线评估
    if args.data_path_baseline:
        baseline_data = load_data(args.data_path_baseline)
        print(f"基线数据加载了 {len(baseline_data)} 个问题。")
        for s in sample_sizes:
            mean_acc, std_acc, _ = run_evaluation_baseline(baseline_data, s, args.num_trials)
            baseline_results.append((s, mean_acc, std_acc))
            print(f"Baseline Pass@{s}: Mean = {mean_acc:.3f}, Std = {std_acc:.3f}")
        save_results_to_file("baseline_results.dat", baseline_results)

    # 如果提供了多模式数据，则加载并运行 multi-mode 评估
    if args.data_path_mode1 and args.data_path_mode2 and args.data_path_mode3:
        data_mode1 = load_data(args.data_path_mode1)
        data_mode2 = load_data(args.data_path_mode2)
        data_mode3 = load_data(args.data_path_mode3)
        if not (len(data_mode1) == len(data_mode2) == len(data_mode3)):
            raise ValueError("三个多模式文件中的问题数量不一致")
        print(f"Multi-mode 数据加载了 {len(data_mode1)} 个问题（每个文件）。")
        for s in sample_sizes:
            mean_acc, std_acc, _ = run_evaluation_multimode(data_mode1, data_mode2, data_mode3, s, args.num_trials)
            multimode_results.append((s, mean_acc, std_acc))
            print(f"Multi-mode Pass@{s}: Mean = {mean_acc:.3f}, Std = {std_acc:.3f}")
        save_results_to_file("multimode_results.dat", multimode_results)

if __name__ == '__main__':
    main()

