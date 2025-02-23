import random
import numpy as np
import os
import json
import time
import torch
import tqdm
import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
from datasets import Dataset, DatasetDict


def get_prompt(mode, use_fewshot=False):
    """
    Load sys_prompt and few-shot examples according to modes(truth_table、code、nl)
    """
    sys_prompt_path = os.path.join('../Prompts', f'sys_prompt_{mode}_star.txt')
    example_path = os.path.join('../Prompts', f'example_{mode}_star.txt')
    with open(sys_prompt_path, encoding="utf-8") as f:
        sys_prompt = f.read()
    with open(example_path, encoding="utf-8") as f:
        example = f.read()
    if use_fewshot:
        fewshot_path = os.path.join('../Prompts', f'prompt_{mode}_star.txt')
        with open(fewshot_path, encoding="utf-8") as f:
            fewshot_example = f.read()
        full_prompt = sys_prompt + '\n\n' + fewshot_example + '\n\n' + example
        full_prompt_wo_fewshot_example = sys_prompt + '\n\n' + example
        return full_prompt, full_prompt_wo_fewshot_example
    else:
        full_prompt = sys_prompt + '\n\n' + example
        return full_prompt, full_prompt


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
    seed_dataset = train_dataset.shuffle(seed=seed).select(range(num_samples))
    print(f"Seed dataset obtained with {len(seed_dataset)} samples.")
    return seed_dataset

def load_model_inference(model_name_or_path='gemma-2-9b'):
    gpu_count = torch.cuda.device_count()
    model = LLM(model=model_name_or_path, tensor_parallel_size=gpu_count)
    return model

##########################################################Code for Generating Response##########################################################################

def generate_responses_batch(model, user_prompts, max_tokens, temperature, top_p, top_k, stop, is_chat_model):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens
    )
    if is_chat_model:
        outputs = model.chat(
            user_prompts,
            sampling_params,
        )
    else:
        outputs = model.generate(
            user_prompts,
            sampling_params,
        )
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text
        responses.append(generated_text)
    return responses

def generate_rationales(model, dataset, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None, mode='truth_table', is_chat_model=False, batch_size=16,use_fewshot=True, huggingface_repo=""):
    """
    Generate rationales for each data point in the dataset.
    """
    rationales = []
    total_num = 0   
    rationale_prompt = get_prompt(mode=mode, use_fewshot=use_fewshot)
    full_prompt = rationale_prompt[0]
    full_prompt_only_example = rationale_prompt[1]
    prompts = []
    prompts_only_example = []
    dataset_list = []
    for item in dataset:
        premises = item.get("premises", "")
        conclusions = item.get("conclusion", "")
        prompt = full_prompt.format(Premises=premises, Conclusions=conclusions)
        prompt_only_example = full_prompt_only_example.format(Premises=premises, Conclusions=conclusions)
        if is_chat_model:
            prompt = [{"role": "user","content": prompt}]
        prompts.append(prompt)
        prompts_only_example.append(prompt_only_example)
        dataset_list.append(item)

    for batch_start in tqdm.tqdm(range(0, len(dataset_list), batch_size)):
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        batch_prompts_only_example = prompts_only_example[batch_start: batch_start + batch_size]
        batch_items = dataset_list[batch_start: batch_start + batch_size]
        try:
            with torch.no_grad():
                batch_responses = generate_responses_batch(
                    model=model,
                    user_prompts=batch_prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop,
                    is_chat_model=is_chat_model
                )
        except Exception as e:
            print(f"Error generating responses for batch starting at index {batch_start}: {e}")
            continue 
        if mode != 'code':
            for prompt, prompt_only_example, item, rationale_response in zip(batch_prompts, batch_prompts_only_example, batch_items, batch_responses):            
                rationale_response = rationale_response.split("<Reasoning>")[-1]
                rationale_response = rationale_response.split("</Answer>")[0] + "</Answer>"
                answer_response = rationale_response.split("<Answer>")[-1]
                if "(A)" in answer_response:
                    predict = "True"
                elif "(B)" in answer_response:
                    predict = "False"
                elif "(C)" in answer_response:
                    predict = "Uncertain"
                else:
                    predict = "Unknown"
                label = item['label']
                if predict == label:
                    rationales.append({
                        'prompt_id': str(total_num),
                        'prompt': prompt_only_example, #prompt[0]['content'],
                        'messages': [{"role": "user","content": prompt_only_example}, { "content":rationale_response.strip(), "role": "assistant" }],
                    })
                    print(f"Generated rationale for data point {total_num + 1}/{len(dataset)}")
                else:
                    print(f"Filter out the data point due to poor quality.") 
                total_num += 1

        else:
            for prompt, prompt_only_example, item, code_response in zip(batch_prompts, batch_prompts_only_example, batch_items, batch_responses):            
                code_response = code_response.split("<PYTHON>")[-1]
                code_response = code_response.split("</PYTHON")[0]
                exec(code_response)
                predict = locals().get("result")
                label = item['label']
                if predict == label:
                    rationales.append({
                        'prompt_id': str(total_num),
                        'prompt': prompt_only_example, #prompt[0]['content'],
                        'messages': [{"role": "user","content": prompt_only_example}, { "content":code_response.strip(), "role": "assistant" }],
                    })
                    print(f"Generated rationale for data point {total_num + 1}/{len(dataset)}")
                else:
                    print(f"Filter out the data point due to poor quality.") 
                total_num += 1
    # with open(os.path.join(output_dir, output_file), 'w') as f:
    #     json.dump(rationales, f, indent=4)
    # print(f"Rationales saved to {os.path.join(output_dir, output_file)}")

    
    ds = Dataset.from_list(rationales)
    ds_dict = DatasetDict({'train': ds})
    ds_dict.push_to_hub(
        repo_id=huggingface_repo,
        private=True
    )
    
    print(
        f"Successfully pushed dataset to Hugging Face Hub: {huggingface_repo} "
        f"(train split, private={True})."
    )
   

################################################# Star Pipeline #############################################################

def generate_rationale_data(model_name_and_path, dataset_name, n_samples=200, batch_size=16, seed=42, max_tokens=512, temperature=1.0, top_p=0.9, top_k=50, stop=None, mode='truth_table', is_chat_model=False, use_fewshot=False, huggingface_repo=""):
    """
    Implements the STaR pipeline where each fine-tuning starts from the initial base model.

    Args:
        client (Together): An initialized Together API client instance.
        base_model (str): Base pre-trained model (e.g., "meta-llama/Meta-Llama-3.1-8B").
        dataset (str): Path to the dataset (e.g., JSONL file).
        validation_file (str): Optional validation dataset.
        n_outer_loops (int): Number of outer-loop iterations.
        n_epochs (int): Epochs for inner-loop fine-tuning. Default is 4.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate.
        lora (bool): Whether to enable LoRA fine-tuning. Default is False.
        lora_params (dict): LoRA parameters (if LoRA is enabled).

    Returns:
        dict: Responses of all stages in the STaR pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    # Obtain seed training set and test set
    dataset = obtain_seed_dataset(dataset_name, n_samples, seed)
    model = load_model_inference(model_name_or_path=model_name_and_path)

    if os.path.exists(os.path.join(output_dir, output_file)):
            pass
    else:
        generate_rationales(
            model=model,
            dataset=dataset,
            max_tokens=max_tokens,
            temperature=temperature,
            batch_size=batch_size,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            mode=mode,
            is_chat_model=is_chat_model, 
            use_fewshot=use_fewshot,
            huggingface_repo=huggingface_repo,
        )

   
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run the STaR pipeline with fine-tuning.")

    # Add arguments
    parser.add_argument("--model_name_and_path", type=str, required=True, 
                        help="Base pre-trained model (e.g., 'meta-llama/Meta-Llama-3.1-8B').")
    parser.add_argument("--mode", type=str, required=True, 
                        help="truth_table, code, nl")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Name of the Hugging Face dataset to use (e.g., 'glue').")
    parser.add_argument("--huggingface_repo", type=str, default="", 
                        help="huggingface_repo.")
    parser.add_argument("--n_samples", type=int, default=200, 
                        help="Number of samples to use from the dataset.")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for fine-tuning.")
    parser.add_argument("--use_fewshot", action="store_true", 
                    help="Enable the fewshot. If not provided, defaults to False.")
    parser.add_argument("--max_tokens", type=int, default=512, 
                        help="Maximum number of tokens for generated responses.")
    parser.add_argument("--temperature", type=float, default=1, 
                        help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--top_k", type=int, default=50, 
                        help="Top-k sampling parameter.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility.")

    # Parse arguments
    args = parser.parse_args()

    # Print arguments for verification
    print("Running with the following arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")


    set_seed(args.seed)
    comp_models = {
                "NousResearch/Meta-Llama-3-8B",
                "NousResearch/Meta-Llama-3.1-8B",
                "unsloth/gemma-2-9b",
                'mistralai/Mistral-7B-v0.3',
                'google/gemma-2-9b',
                "Qwen/Qwen2.5-7B",

            }
    chat_models = {
        "NousResearch/Meta-Llama-3-8B-Instruct",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        'NousResearch/Meta-Llama-3.1-8B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
        'google/gemma-2-9b-it',
        'Qwen/Qwen2.5-7B-Instruct',
        'unsloth/gemma-2-2b-it',
        'Qwen/QwQ-32B-Preview',
        'Qwen/Qwen2.5-14B-Instruct-1M',
    }

    if args.model_name_and_path in chat_models:
        is_chat = True
    elif args.model_name_and_path in comp_models:
        is_chat = False
    else:
        is_chat = True

    # Run the pipeline
    generate_rationale_data(
        model_name_and_path=args.model_name_and_path,
        dataset_name=args.dataset_name,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mode=args.mode,
        is_chat_model=is_chat,
        use_fewshot=args.use_fewshot,
        huggingface_repo=args.huggingface_repo,
    )

if __name__ == "__main__":
    main()






