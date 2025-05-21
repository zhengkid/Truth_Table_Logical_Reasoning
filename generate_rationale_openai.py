import os
import json
import random
import argparse

import numpy as np
import torch
import tqdm
import openai
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi




def obtain_seed_dataset(dataset_name, num_samples, seed=42):
    """
    Load a seed dataset from a Hugging Face dataset.

    Args:
        dataset_name (str): Name of the Hugging Face dataset (e.g., "glue").
        num_samples (int): Number of samples to include in the seed dataset.
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        Dataset: A subset of the dataset containing the specified number of samples.
    """
    # Load the dataset
    print(f"Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name)
    train_dataset = dataset['train']

    # Shuffle and select a subset
    print(f"Selecting {num_samples} samples from the dataset...")
    seed_dataset = train_dataset.select(range(num_samples))
    #seed_dataset = train_dataset.shuffle(seed=seed).select(range(num_samples))
    print(f"Seed dataset obtained with {len(seed_dataset)} samples.")
    return seed_dataset
def check_huggingface_repo_exists(repo_id: str) -> bool:
    """
    检查 Hugging Face 上的 repo 是否存在。
    """
    if not repo_id:
        return False
    api = HfApi()
    try:
        api.repo_info(repo_id)
        return True
    except Exception:
        return False


def generate_responses_batch(
    model,
    user_prompts,
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
):
    """
    Generate batch responses using OpenAI API, supporting both chat and completion models.
    （当前未在单条生成模式中使用，可保留以备他用。）
    """
    responses = []
    for prompts in user_prompts:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=prompts,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=top_p,
        )
        text = resp.choices[0].message.content.strip()
        print(text)
        responses.append(text)
    return responses


def get_prompt(mode: str, prompt_mode: str, use_fewshot: bool = False):
    """
    Load system prompt and few-shot examples according to modes (truth_table, code, nl).
    """
    sys_prompt_path = os.path.join('./Prompts', f'sys_prompt_star_{prompt_mode}.txt')
    example_path    = os.path.join('./Prompts', f'example_{mode}_star_{prompt_mode}.txt')

    with open(sys_prompt_path, encoding="utf-8") as f:
        sys_prompt = f.read()
    with open(example_path, encoding="utf-8") as f:
        example = f.read()

    if use_fewshot:
        fewshot_path = os.path.join('./Prompts', f'prompt_{mode}_star_{prompt_mode}.txt')
        with open(fewshot_path, encoding="utf-8") as f:
            fewshot_example = f.read()
        return sys_prompt + '\n\n', fewshot_example + '\n\n' + example
    else:
        return sys_prompt + '\n\n', example


def generate_rationales(
    model_path: str,
    dataset,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    mode: str = 'truth_table',
    use_fewshot: bool = True,
    huggingface_repo: str = "",
    prompt_mode: str = 'v1',
    checkpoint_file: str = 'rationales.jsonl'
):
    """
    逐条生成 rationale，不并行，每生成一条就写入 checkpoint；
    如果中断，下次运行会从最后写入的条目开始继续。
    """
    # ——— 1. 读取已生成的条数，确定起始索引 ———
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for _ in f:
                start_idx += 1
    print(f"[INFO] 从第 {start_idx} 条开始生成 rationales")

    # ——— 2. 打开 checkpoint 文件（追加模式） ———
    fout = open(checkpoint_file, 'a', encoding='utf-8')

    # ——— 3. 加载 prompts ———
    sys_prompt, user_prompt = get_prompt(
        mode=mode,
        prompt_mode=prompt_mode,
        use_fewshot=use_fewshot
    )

    # ——— 4. 逐条生成 ———
    for idx in tqdm.tqdm(range(start_idx, len(dataset))):
        item       = dataset[idx]
        premise    = item['premises']
        conclusion = item['conclusion']
        label      = item.get('label', None)

        user_example = user_prompt.format(Premises=premise, Conclusions=conclusion)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": user_example}
        ]
        print(messages) 
        resp = openai.ChatCompletion.create(
            model=model_path,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,


        )
        print(resp.usage)
        text = resp.choices[0].message.content.strip()
        print(text)

        # ——— 5. 写入 checkpoint 并 flush ———
        record = {
            "premises":   premise,
            "conclusion": conclusion,
            "label":      label,
            "response":   text,
            "messages":   messages
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()

    # ——— 6. 读取所有记录，构造 Dataset 并推送到 Hugging Face Hub ———
    all_rationales = []
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            all_rationales.append(json.loads(line))

    ds = Dataset.from_list(all_rationales)
    ds_dict = DatasetDict({'train': ds})
    ds_dict.push_to_hub(
        repo_id=huggingface_repo,
        private=True
    )
    print(f"[INFO] 已成功推送到 Hugging Face Hub: {huggingface_repo} (train split, private=True)")


################################################# Star Pipeline #############################################################

def generate_rationale_data(
    model_name_and_path: str,
    dataset_name: str,
    n_samples: int = 200,
    seed: int = 42,
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50,
    stop=None,
    mode: str = 'truth_table',
    is_chat_model: bool = False,
    use_fewshot: bool = False,
    huggingface_repo: str = "",
    gpu_count: int = 4,
    prompt_mode: str = 'v1',
    number_candidates: int = 10,
    checkpoint_file: str = 'rationales.jsonl'
):
    """
    Implements the STaR pipeline where each fine-tuning starts from the initial base model.
    """
    dataset = obtain_seed_dataset(dataset_name, n_samples, seed)

    generate_rationales(
        model_path=model_name_and_path,
        dataset=dataset,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        mode=mode,
        use_fewshot=use_fewshot,
        huggingface_repo=huggingface_repo,
        prompt_mode=prompt_mode,
        checkpoint_file=checkpoint_file
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Run the STaR pipeline with fine-tuning.")

    parser.add_argument("--model_name_and_path", type=str, required=True,
                        help="Base pre-trained model (e.g., 'meta-llama/Meta-Llama-3.1-8B').")
    parser.add_argument("--mode", type=str, required=True,
                        help="truth_table, code, nl")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Name of the Hugging Face dataset to use (e.g., 'glue').")
    parser.add_argument("--prompt_mode", type=str, default="v1",
                        help="Prompt 模式 (e.g., 'v1').")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples to use from the dataset.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (保留参数，但实际不并行调用).")
    parser.add_argument("--use_fewshot", action="store_true",
                        help="Enable few-shot examples.")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Maximum number of tokens for generated responses.")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--gpu_count", type=int, default=4,
                        help="Number of GPUs for inference.")
    parser.add_argument("--huggingface_repo", type=str, default="",
                        help="目标 Hugging Face repo ID，用于推送数据集。")
    parser.add_argument("--checkpoint_file", type=str, default="rationales.jsonl",
                        help="断点续存文件名，默认为 rationales.jsonl。")

    args = parser.parse_args()

    print("Running with the following arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    set_seed(args.seed)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # 如果 Hugging Face repo 不存在，则执行生成
    if not check_huggingface_repo_exists(args.huggingface_repo):
        generate_rationale_data(
            model_name_and_path=args.model_name_and_path,
            dataset_name=args.dataset_name,
            n_samples=args.n_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            mode=args.mode,
            use_fewshot=args.use_fewshot,
            prompt_mode=args.prompt_mode,
            huggingface_repo=args.huggingface_repo,
            gpu_count=args.gpu_count,
            checkpoint_file=args.checkpoint_file
        )
    else:
        print(f"[INFO] Dataset {args.huggingface_repo} already exists. Skipping generation.")


if __name__ == "__main__":
    main()

