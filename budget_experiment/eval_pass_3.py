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
    每个元素为一个字典，包含 "predicts" 和 "label" 字段。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def sample_pass_metric(responses, label, sample_size):
    """
    对单个问题，随机从 responses 中采样 sample_size 个响应，
    检查这 sample_size 个中是否至少有一个正确，返回 1（通过）或 0（未通过）。
    """
    if len(responses) < sample_size:
        raise ValueError("响应数量不足以进行抽样")
    sample = random.sample(responses, sample_size)
    # 如果任意一个采样的响应正确，则认为该问题通过评估
    return int(any(evaluate_prediction(response, label) for response in sample))

def run_evaluation(data, sample_size, num_trials):
    """
    对所有问题进行多次试验，每次试验中对于每个问题，
    随机采样 sample_size 个响应，统计 pass@sample_size 的准确率。
    
    返回：平均准确率、标准差、以及每次试验的准确率列表。
    """
    trial_results = []
    for trial in range(num_trials):
        correct_count = 0
        for item in data:
            predictions = item['predictions']
            label = item['label']
            correct_count += sample_pass_metric(predictions, label, sample_size)
        trial_accuracy = correct_count / len(data)
        trial_results.append(trial_accuracy)
    mean_accuracy = np.mean(trial_results)
    std_accuracy = np.std(trial_results)
    return mean_accuracy, std_accuracy, trial_results

def main():
    parser = argparse.ArgumentParser(description="Pass@N 测试 Pipeline")
    parser.add_argument('--data_path', type=str, required=True,
                        help='输入数据的 JSON 文件路径，每个元素包含 "predicts" 和 "label"')
    parser.add_argument('--sample_size', type=int, default=3,
                        help='每个问题中采样的响应数量（例如 pass@3 则 sample_size=3）')
    parser.add_argument('--num_trials', type=int, default=10,
                        help='重复采样试验次数，用于统计平均准确率和标准差')
    args = parser.parse_args()

    # 加载数据
    data = load_data(args.data_path)
    print(f"加载了 {len(data)} 个问题。")
    
    # 运行多次采样评估
    mean_acc, std_acc, trial_results = run_evaluation(data, args.sample_size, args.num_trials)
    print(f"Pass@{args.sample_size} Accuracy: Mean = {mean_acc:.3f}, Std = {std_acc:.3f}")
    print("各次试验的准确率：", trial_results)
    
    # 绘制误差条图
    plt.errorbar(x=[1], y=[mean_acc], yerr=[std_acc], fmt='o', capsize=5)
    plt.xticks([1], [f'Pass@{args.sample_size}'])
    plt.ylabel("Accuracy")
    plt.title(f"Pass@{args.sample_size} Accuracy over {args.num_trials} Trials")
    plt.show()

if __name__ == '__main__':
    main()

