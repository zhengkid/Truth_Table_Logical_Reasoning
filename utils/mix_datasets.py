import argparse
import random
import datasets
import re
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Mix multiple Hugging Face datasets with quality checks")
    parser.add_argument("--input_datasets", type=str, nargs="+", required=True,
                        help="List of Hugging Face datasets to mix (e.g., hf_user/dataset_nl_cot hf_user/dataset_code hf_user/dataset_answer)")
    parser.add_argument("--output_dataset", type=str, required=True,
                        help="Output Hugging Face dataset ID (e.g., hf_user/mixed_dataset)")
    parser.add_argument("--mix_mode", type=str, choices=["direct", "uniq"], default="uniq",
                        help="Mixing mode: 'direct' (add all data) or 'unique_conclusion' (filter by unique conclusions)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def check_validity(sample, mode):
    for message in sample["messages"]:
        if message["role"] == "assistant":
            content = message["content"]

            if mode == "nl":
                return "<end_of_nl_cot>" in content 

            elif mode == "code":
                return ("<end_of_code>" in content and
                        ("def " in content or "class " in content or "import " in content))

            elif mode == "truth_table":
                return "<end_of_truth_table>" in content

    return False

def extract_conclusion(sample):
    for message in sample["messages"]:
        if message["role"] == "user":
            match = re.search(r"<conclusion>(.*?)</conclusion>", message["content"], re.DOTALL)
            if match:
                return match.group(1).strip()
    return None

def load_all_datasets(dataset_paths):
    data = []
    for dataset_path in dataset_paths:
        print(f"Loading dataset: {dataset_path}")
        dataset = load_dataset(dataset_path)
        if 'nl' in dataset_path:
            mode = 'nl'
        elif 'code' in dataset_path:
            mode = 'code'
        elif 'truth_table' in dataset_path:
            mode = 'truth_table'
            
        for sample in dataset["train"]:
            if check_validity(sample, mode):
                data.append(sample)
    return data

def mix_datasets_direct(data):
    print(f"Using direct mixing mode: {len(data)} samples retained (no filtering)")
    return data

def mix_datasets_unique_conclusion(data):
    conclusion_map = {}

    for sample in data:
        conclusion = extract_conclusion(sample)
        if conclusion:
            if conclusion not in conclusion_map:
                conclusion_map[conclusion] = []
            conclusion_map[conclusion].append(sample)

    mixed_data = []
    for conclusion, samples in conclusion_map.items():
        chosen_sample = random.choice(samples)
        mixed_data.append(chosen_sample)

    return mixed_data

def ensure_dataset_format(data):
    return datasets.Dataset.from_list(data)

def save_dataset(data, output_path):
    dataset = ensure_dataset_format(data)
    dataset.push_to_hub(output_path)
    print(f"Dataset successfully uploaded to: {output_path}")

def main():
    args = parse_args()
    
    random.seed(args.seed)

    print(f"Using random seed: {args.seed}")
    print("Loading datasets...")
    all_data = load_all_datasets(args.input_datasets)

    print("Mixing datasets...")
    if args.mix_mode == "direct":
        mixed_data = mix_datasets_direct(all_data)
    else:
        mixed_data = mix_datasets_unique_conclusion(all_data)

    print(f"Final dataset size: {len(mixed_data)}")
    save_dataset(mixed_data, args.output_dataset)

if __name__ == "__main__":
    main()

