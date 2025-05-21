import json
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import Counter
import random
import numpy as np

def set_seed(seed: int):
    """
    固定全局随机种子，确保 random 和 numpy 的采样可复现。
    """
    random.seed(seed)
    np.random.seed(seed)
def evaluate_prediction(prediction, label):
    """
    判断单个预测是否正确，简单采用大小写不敏感的字符串相等。
    根据实际任务，可替换为更复杂的评估逻辑。
    """
    return prediction.strip().lower() == label.strip().lower()

def load_data(filepath):
    """
    加载数据文件，假设文件为 JSON 列表，
    每个元素为一个字典，包含 "predictions" 和 "label" 字段。
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def sample_pass_metric_baseline(responses, label, sample_size):
    """
    单一模式下的基线方法：
    从 responses 中随机采样 sample_size 个响应，
    如果采样结果中至少有一个正确，则返回 1，否则返回 0。
    """
    if len(responses) < sample_size:
        raise ValueError("响应数量不足以进行抽样")
    sample = random.sample(responses, sample_size)
    return int(any(evaluate_prediction(response, label) for response in sample))


def vote_prediction(predictions):
    """
    对一组预测做投票，返回出现次数最多的预测。
    若平票，则随机选一个。
    """
    cnt = Counter(predictions)
    most_common = cnt.most_common()
    max_votes = most_common[0][1]
    # 所有得票 = max_votes 的候选
    candidates = [pred for pred, v in most_common if v == max_votes]
    return random.choice(candidates)

def sample_acc_metric_baseline(responses, label, sample_size):
    """
    单一模式下的基线方法（Accuracy）：
      1. 从 responses 中随机采样 sample_size 个响应；
      2. 对采样结果做投票，选出最终预测；
      3. 比对最终预测与真实 label，正确返回 1，否则返回 0。
    """
    if len(responses) < sample_size:
        raise ValueError("响应数量不足以进行抽样")
    sample = random.sample(responses, sample_size)
    pred = vote_prediction(sample)
    return int(evaluate_prediction(pred, label))

def run_evaluation_baseline(data, sample_size, num_trials):
    """
    对所有问题进行 num_trials 次试验，每次试验中：
      - 单一模式下随机采样 sample_size 个响应，
      - 判断是否至少有一个正确。
    返回平均准确率、标准差以及各次试验的准确率列表。
    """
    trial_results = []
    for trial in range(num_trials):
        correct_count = 0
        for item in data:
            responses = item['predictions']
            label = item['label']
            correct_count += sample_acc_metric_baseline(responses, label, sample_size)
        trial_accuracy = (correct_count / len(data)) * 100
        trial_results.append(trial_accuracy)
    return np.mean(trial_results), np.std(trial_results), trial_results

def sample_pass_metric_multimode(responses1, responses2, responses3, label, sample_size_total):
    """
    三模态混合方法：
      - 分别从每个 mode 的响应列表中采样 sample_size_mode 个响应，
        其中 sample_size_mode = sample_size_total // 3，
      - 合并成总共 sample_size_total 个响应，
      - 若至少一个响应正确，返回 1，否则返回 0。
    """
    sample_size_mode = sample_size_total * 1 // 3
    sample_size_mode1 = sample_size_total* 1 // 3
    sample_size_mode2 = sample_size_total * 1 // 3
    sample1 = random.sample(responses1, sample_size_mode1)
    sample2 = random.sample(responses2, sample_size_mode)
    sample3 = random.sample(responses3, sample_size_mode2)
    combined_sample = sample1 + sample2 + sample3
    return int(any(evaluate_prediction(response, label) for response in combined_sample))

def sample_acc_metric_multimode(responses1, responses2, responses3, label, sample_size_total):
    """
    三模态混合方法（Accuracy）：
      1. 从每个 mode 的响应列表中随机采样 sample_size_mode = sample_size_total // 3 条；
      2. 合并成一个总样本列表；
      3. 对合并样本做投票选出最终预测；
      4. 比对最终预测与真实 label，正确返回 1，否则返回 0。
    """
    #sample_size_mode = sample_size_total // 3
    sample_size_mode1 = sample_size_total * 1 // 3
    sample_size_mode2 = sample_size_total* 1 // 3
    sample_size_mode3 = sample_size_total * 1 // 3
    if len(responses1) < sample_size_mode1 or len(responses2) < sample_size_mode2 or len(responses3) < sample_size_mode3:
        raise ValueError("某个模式的响应数量不足以进行抽样")
    sample1 = random.sample(responses1, sample_size_mode1)
    sample2 = random.sample(responses2, sample_size_mode2)
    sample3 = random.sample(responses3, sample_size_mode3)
    combined = sample1 + sample2 + sample3
    pred = vote_prediction(combined)
    return int(evaluate_prediction(pred, label))

def run_evaluation_multimode(data1, data2, data3, sample_size_total, num_trials):
    """
    对所有问题进行 num_trials 次试验，每次试验中：
      - 分别从三个 mode 的数据中采样，
      - 合并后判断是否至少有一个响应正确。
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
            label = data1[i]['label']  # 假定三个文件中同一问题 label 一致
            correct_count += sample_acc_metric_multimode(responses1, responses2, responses3, label, sample_size_total)
        trial_accuracy = (correct_count / num_questions) * 100
        trial_results.append(trial_accuracy)
    return np.mean(trial_results), np.std(trial_results), trial_results

def sample_pass_metric_two_modes(responses1, responses2, label, sample_size_total):
    """
    两模态混合方法：
      - 从每个 mode 中分别采样 sample_size_mode 个响应，
        其中 sample_size_mode = sample_size_total // 2，
      - 合并后判断至少有一个响应正确则返回 1，否则返回 0。
    """
    sample_size_mode = sample_size_total // 2
    sample1 = random.sample(responses1, sample_size_mode)
    sample2 = random.sample(responses2, sample_size_mode)
    combined_sample = sample1 + sample2
    return int(any(evaluate_prediction(response, label) for response in combined_sample))

def sample_acc_metric_two_modes(responses1, responses2, label, sample_size_total):
    """
    两模态混合方法（Accuracy）：
      1. 计算每个 mode 的采样大小 sample_size_mode = sample_size_total // 2；
      2. 从每个 mode 的响应列表中随机采样 sample_size_mode 条；
      3. 合并为一个列表；
      4. 对合并列表做投票，选出票数最多的预测作为最终预测；
      5. 与真实 label 比对，正确返回 1，否则返回 0。
    """
    sample_size_mode = sample_size_total // 2
    if len(responses1) < sample_size_mode or len(responses2) < sample_size_mode:
        raise ValueError("某个模式的响应数量不足以进行抽样")
    sample1 = random.sample(responses1, sample_size_mode)
    sample2 = random.sample(responses2, sample_size_mode)
    combined = sample1 + sample2
    pred = vote_prediction(combined)
    return int(evaluate_prediction(pred, label))

def run_evaluation_two_mode(data1, data2, sample_size_total, num_trials):
    """
    两模态混合评估：
      - 对所有问题进行 num_trials 次试验，
      - 每个问题分别从两个 mode 的数据中采样，
      - 合并后判断是否至少有一个响应正确。
    返回平均准确率、标准差以及各次试验的准确率列表。
    """
    num_questions = len(data1)
    trial_results = []
    for trial in range(num_trials):
        correct_count = 0
        for i in range(num_questions):
            responses1 = data1[i]['predictions']
            responses2 = data2[i]['predictions']
            label = data1[i]['label']
            correct_count += sample_acc_metric_two_modes(responses1, responses2, label, sample_size_total)
        trial_accuracy = (correct_count / num_questions) * 100
        trial_results.append(trial_accuracy)
    return np.mean(trial_results), np.std(trial_results), trial_results

def save_results_to_file(filename, results):
    """
    将结果保存为适用于 LaTeX pgfplots 的数据格式。
    每行格式：sample_size mean_accuracy std_accuracy
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("% sample_size mean_accuracy std_accuracy\n")
        for sample_size, mean_acc, std_acc in results:
            f.write(f"{sample_size} {mean_acc} {std_acc}\n")
    print(f"Results saved to {filename}")

def main():
    parser = argparse.ArgumentParser(description="评估 Pipeline：基线和我们方法的单模态及混合模式 Pass@k 测试")
    # 基线方法数据文件（单模态，对三个模式分别评估）
    parser.add_argument('--baseline_code', type=str, default=None,
                        help='基线方法 mode1 数据文件路径')
    parser.add_argument('--baseline_nl', type=str, default=None,
                        help='基线方法 mode2 数据文件路径')
    parser.add_argument('--baseline_tt', type=str, default=None,
                        help='基线方法 mode3 数据文件路径')
    # 我们方法数据文件（单模态，对三个模式分别评估）
    parser.add_argument('--method_code', type=str, default=None,
                        help='我们方法 mode1 数据文件路径')
    parser.add_argument('--method_nl', type=str, default=None,
                        help='我们方法 mode2 数据文件路径')
    parser.add_argument('--method_tt', type=str, default=None,
                        help='我们方法 mode3 数据文件路径')
    # 采样试验次数
    parser.add_argument('--num_trials', type=int, default=10,
                        help='重复采样试验次数，用于统计平均准确率和标准差')
    args = parser.parse_args()

    sample_sizes = [3, 6, 12, 24, 48, 72, 96, 120]
    num_trials = args.num_trials
    set_seed(42)
    # --------------------------- #
    # 基线方法评估
    # --------------------------- #
    if args.baseline_code and args.baseline_nl and args.baseline_tt:
        baseline_data_mode1 = load_data(args.baseline_code)
        baseline_data_mode2 = load_data(args.baseline_nl)
        baseline_data_mode3 = load_data(args.baseline_tt)
        if not (len(baseline_data_mode1) == len(baseline_data_mode2) == len(baseline_data_mode3)):
            raise ValueError("三个基线文件中的问题数量不一致")
        num_questions = len(baseline_data_mode1)
        print(f"基线数据加载了 {num_questions} 个问题（每个文件）。")
        
        # 单模态评估
        baseline_single_mode_results = {'code': [], 'nl': [], 'tt': []}
        # 混合评估（3 模式结合）
        baseline_mix_results = []
        for s in sample_sizes:
            mean1, std1, _ = run_evaluation_baseline(baseline_data_mode1, s, num_trials)
            mean2, std2, _ = run_evaluation_baseline(baseline_data_mode2, s, num_trials)
            mean3, std3, _ = run_evaluation_baseline(baseline_data_mode3, s, num_trials)
            baseline_single_mode_results['code'].append((s, mean1, std1))
            baseline_single_mode_results['nl'].append((s, mean2, std2))
            baseline_single_mode_results['tt'].append((s, mean3, std3))
            # 三模态混合
            mean_mix, std_mix, _ = run_evaluation_multimode(baseline_data_mode1, baseline_data_mode2, baseline_data_mode3, s, num_trials)
            baseline_mix_results.append((s, mean_mix, std_mix))
            print(f"[Baseline] Pass@{s}: code = {mean1:.3f}, nl = {mean2:.3f}, tt = {mean3:.3f}, mix3 = {mean_mix:.3f}")
        # 保存结果
        save_results_to_file("baseline_code_vote_results.dat", baseline_single_mode_results['code'])
        save_results_to_file("baseline_nl_vote_results.dat", baseline_single_mode_results['nl'])
        save_results_to_file("baseline_tt_vote_results.dat", baseline_single_mode_results['tt'])
        # save_results_to_file("baseline_mix3_vote_results.dat", baseline_mix_results)
    else:
        print("未提供完整的基线方法数据（需要 --baseline_mode1, --baseline_mode2, --baseline_mode3）。")

    # --------------------------- #
    # 我们的方法评估
    # --------------------------- #
    if args.method_code and args.method_nl and args.method_tt:
        method_data_mode1 = load_data(args.method_code)
        method_data_mode2 = load_data(args.method_nl)
        method_data_mode3 = load_data(args.method_tt)
        if not (len(method_data_mode1) == len(method_data_mode2) == len(method_data_mode3)):
            raise ValueError("三个我们方法文件中的问题数量不一致")
        num_questions = len(method_data_mode1)
        print(f"我们方法数据加载了 {num_questions} 个问题（每个文件）。")
        
        # 单模态评估
        method_single_mode_results = {'code': [], 'nl': [], 'tt': []}
        # 混合评估（3 模式结合）
        method_mix3_results = []
        # 两模式混合评估（任意两种组合，有三种组合）
        method_mix2_results = {'code+nl': [], 'code+tt': [], 'nl+tt': []}
        
        for s in sample_sizes:
            m_mean1, m_std1, _ = run_evaluation_baseline(method_data_mode1, s, num_trials)
            m_mean2, m_std2, _ = run_evaluation_baseline(method_data_mode2, s, num_trials)
            m_mean3, m_std3, _ = run_evaluation_baseline(method_data_mode3, s, num_trials)
            method_single_mode_results['code'].append((s, m_mean1, m_std1))
            method_single_mode_results['nl'].append((s, m_mean2, m_std2))
            method_single_mode_results['tt'].append((s, m_mean3, m_std3))
            # 三模态混合
            m_mean_mix3, m_std_mix3, _ = run_evaluation_multimode(method_data_mode1, method_data_mode2, method_data_mode3, s, num_trials)
            method_mix3_results.append((s, m_mean_mix3, m_std_mix3))
            # 两模态混合
            m_mean_12, m_std_12, _ = run_evaluation_two_mode(method_data_mode1, method_data_mode2, s, num_trials)
            m_mean_13, m_std_13, _ = run_evaluation_two_mode(method_data_mode1, method_data_mode3, s, num_trials)
            m_mean_23, m_std_23, _ = run_evaluation_two_mode(method_data_mode2, method_data_mode3, s, num_trials)
            method_mix2_results['code+nl'].append((s, m_mean_12, m_std_12))
            method_mix2_results['code+tt'].append((s, m_mean_13, m_std_13))
            method_mix2_results['nl+tt'].append((s, m_mean_23, m_std_23))
            
            print(f"[Our Method] Pass@{s}: code = {m_mean1:.3f}, nl = {m_mean2:.3f}, tt = {m_mean3:.3f}, mix3 = {m_mean_mix3:.3f}, mixcodenl = {m_mean_12:.3f}, mixcodett = {m_mean_13:.3f}, mixnltt = {m_mean_23:.3f}")
            
        # 保存结果
        # save_results_to_file("method_code_results.dat", method_single_mode_results['code'])
        # save_results_to_file("method_nl_results.dat", method_single_mode_results['nl'])
        # save_results_to_file("method_tt_results.dat", method_single_mode_results['tt'])
        save_results_to_file("method_mix3_results.dat", method_mix3_results)
        # save_results_to_file("method_mix2_code+nl_results.dat", method_mix2_results['code+nl'])
        # save_results_to_file("method_mix2_code+tt_results.dat", method_mix2_results['code+tt'])
        # save_results_to_file("method_mix2_nl+tt_results.dat", method_mix2_results['nl+tt'])
    else:
        print("未提供完整的我们方法数据（需要 --method_mode1, --method_mode2, --method_mode3）。")

if __name__ == '__main__':
    main()

