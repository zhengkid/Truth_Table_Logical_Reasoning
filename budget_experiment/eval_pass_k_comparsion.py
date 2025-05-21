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
    对单个问题，基线方法：从 responses 中随机采样 sample_size 个响应，
    如果采样的结果中至少有一个正确，则返回 1，否则返回 0。
    """
    if len(responses) < sample_size:
        raise ValueError("响应数量不足以进行抽样")
    sample = random.sample(responses, sample_size)
    return int(any(evaluate_prediction(response, label) for response in sample))

def run_evaluation_baseline(data, sample_size, num_trials):
    """
    基线方法：对所有问题进行 num_trials 次试验，
    每次试验中对每个问题随机采样 sample_size 个响应，并统计 pass@sample_size 的准确率。
    
    返回：平均准确率、标准差、以及每次试验的准确率列表。
    """
    trial_results = []
    for trial in range(num_trials):
        correct_count = 0
        for item in data:
            responses = item['predictions']
            label = item['label']
            correct_count += sample_pass_metric_baseline(responses, label, sample_size)
        trial_accuracy = correct_count / len(data)
        trial_results.append(trial_accuracy)
    return np.mean(trial_results), np.std(trial_results), trial_results

def sample_pass_metric_multimode(responses1, responses2, responses3, label, sample_size_total):
    """
    对单个问题，多模式方法：
      - 从每个 mode 的响应列表中采样 sample_size_mode 个响应，
      - 合并采样结果（总共 k 个响应），
      - 如果这 k 个响应中至少有一个正确，则返回 1，否则返回 0。
    """
    sample_size_mode = sample_size_total // 3
    # 若 sample_size_total 不能被 3 整除，可选择均摊或额外采样；这里简单采用整除的结果
    sample1 = random.sample(responses1, sample_size_mode)
    sample2 = random.sample(responses2, sample_size_mode)
    sample3 = random.sample(responses3, sample_size_mode)
    combined_sample = sample1 + sample2 + sample3
    return int(any(evaluate_prediction(response, label) for response in combined_sample))

def run_evaluation_multimode(data1, data2, data3, sample_size_total, num_trials):
    """
    多模式方法：对所有问题进行 num_trials 次试验，
    每次试验中对每个问题分别从三个 mode 的数据中采样，然后合并采样结果判断是否至少有一个正确；
    返回平均准确率、标准差以及各次试验的准确率列表。
    """
    num_questions = len(data1)
    trial_results = []
    for trial in range(num_trials):
        correct_count = 0
        for i in range(num_questions):
            responses1 = data1[i]['predictions']
            responses2 = data2[i]['predictions']
            responses3 = data3[i]['predictions']
            label = data1[i]['label']  # 假定三个文件中同一问题的 label 相同
            correct_count += sample_pass_metric_multimode(responses1, responses2, responses3, label, sample_size_total)
        trial_accuracy = correct_count / num_questions
        trial_results.append(trial_accuracy)
    return np.mean(trial_results), np.std(trial_results), trial_results

def main():
    parser = argparse.ArgumentParser(description="Baseline 和 Multi-mode Pass@k 评估 Pipeline")
    parser.add_argument('--data_path_baseline', type=str, default=None,
                        help='基线方法的数据文件路径，JSON 列表，每个元素包含 "predictions" 和 "label"')
    parser.add_argument('--data_path_mode1', type=str, default=None,
                        help='Multi-mode 方法 mode1 数据文件路径')
    parser.add_argument('--data_path_mode2', type=str, default=None,
                        help='Multi-mode 方法 mode2 数据文件路径')
    parser.add_argument('--data_path_mode3', type=str, default=None,
                        help='Multi-mode 方法 mode3 数据文件路径')
    parser.add_argument('--sample_size', type=int, default=3,
                        help='采样总数 k，对于基线方法，直接采样 k 个响应；对于多模式方法，从每个 mode 采样 k//3 个响应')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='重复采样试验次数，用于统计平均准确率和标准差')
    args = parser.parse_args()

    results = {}
    # 如果提供了基线数据，则运行基线评估
    if args.data_path_baseline:
        baseline_data = load_data(args.data_path_baseline)
        print(f"基线数据加载了 {len(baseline_data)} 个问题。")
        mean_acc_base, std_acc_base, trial_results_base = run_evaluation_baseline(
            baseline_data, args.sample_size, args.num_trials)
        results['Baseline'] = (mean_acc_base, std_acc_base)
        print(f"Baseline Pass@{args.sample_size} Accuracy: Mean = {mean_acc_base:.3f}, Std = {std_acc_base:.3f}")
        print("各次试验的准确率（Baseline）：", trial_results_base)

    # 如果提供了多模式数据，则运行多模式评估
    if args.data_path_mode1 and args.data_path_mode2 and args.data_path_mode3:
        data_mode1 = load_data(args.data_path_mode1)
        data_mode2 = load_data(args.data_path_mode2)
        data_mode3 = load_data(args.data_path_mode3)
        # 检查三个文件中问题数量是否一致
        if not (len(data_mode1) == len(data_mode2) == len(data_mode3)):
            raise ValueError("三个多模式文件中的问题数量不一致")
        print(f"Multi-mode 数据加载了 {len(data_mode1)} 个问题（每个文件）。")
        mean_acc_multi, std_acc_multi, trial_results_multi = run_evaluation_multimode(
            data_mode1, data_mode2, data_mode3, args.sample_size, args.num_trials)
        results['Multi-mode'] = (mean_acc_multi, std_acc_multi)
        print(f"Multi-mode Pass@{args.sample_size} Accuracy: Mean = {mean_acc_multi:.3f}, Std = {std_acc_multi:.3f}")
        print("各次试验的准确率（Multi-mode）：", trial_results_multi)

    # 可视化比较结果（如果同时提供了两种评估方式，则在同一图中比较）
    if results:
        labels = list(results.keys())
        means = [results[label][0] for label in labels]
        stds = [results[label][1] for label in labels]
        
        # 简单的 bar plot 加误差条
        x = np.arange(len(labels))
        fig, ax = plt.subplots()
        ax.bar(x, means, yerr=stds, align='center', alpha=0.7, capsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Pass@{args.sample_size} Accuracy over {args.num_trials} Trials")
        plt.show()

if __name__ == '__main__':
    main()

