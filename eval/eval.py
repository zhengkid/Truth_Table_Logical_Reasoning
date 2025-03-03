import random
import numpy as np
import os
import json
import torch
import tqdm
import argparse
from datasets import load_dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils_function import (
    get_prompt,
    execute_with_timeout,
    load_model_inference,
    generate_responses_batch,
    is_executable,
    remove_incorrect_code_symbols,
    post_process_batch_data_eval,
)

def evaluation_batch(model, dataset, output_dir, raw_data_path, accuracy_path,
               max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None,
               mode='truth_table', is_chat_model=False, batch_size=16, use_fewshot=True,prompt_mode='v1', number_candidates=10):
    rationales = []
    correct_num = 0
    total_num = 0   
    rationale_prompt, _ = get_prompt(mode=mode, prompt_mode=prompt_mode, use_fewshot=use_fewshot)
    for batch_start in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch_dataset = dataset[batch_start: batch_start + batch_size]
        batch_prompts = []
        batch_items = []
        # Accumulate batch data 
        batch_premises = batch_dataset['premises']
        batch_conclusions = batch_dataset['conclusion']
        batch_labels = batch_dataset['label']
        for premise, conclusion, label  in zip(batch_premises, batch_conclusions, batch_labels):
            prompt = rationale_prompt.format(Premises=premise, Conclusions=conclusion)
            if is_chat_model:
                prompt = [{"role": "user","content": prompt}]
            batch_prompts.append(prompt)
            batch_items.append({'premises':premise, 'conclusion': conclusion, 'label': label})
        # Process batch data via LLM
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
                    is_chat_model=is_chat_model,
                    number_candidates=number_candidates,
                )
        except Exception as e:
            print(f"Error generating responses for batch starting at index {batch_start}: {e}")
            tqdm.tqdm.update(1)
            continue 
        batch_rationales, correct_num, total_num, accuracy = post_process_batch_data_eval(batch_prompts, batch_items, batch_responses, mode, total_num, correct_num)
        rationales.extend(batch_rationales)

    # for batch_start in tqdm.tqdm(range(0, len(dataset_list), batch_size)):
    #     batch_prompts = prompts[batch_start: batch_start + batch_size]
    #     batch_items = dataset_list[batch_start: batch_start + batch_size]
    #     print(batch_start, len(batch_items))


    #     if mode != 'code':
    #         for prompt, item, rationale_response in zip(batch_prompts, batch_items, batch_responses):
    #             label = item['label']
    #             for j in range(len(rationale_response)):
    #                 rationale_response_sample_j = rationale_response[j]
    #                 rationale_response_sample_j = rationale_response_sample_j.split("<Reasoning>")[-1]
    #                 rationale_response_sample_j = rationale_response_sample_j.split("</Answer>")[0] + "</Answer>"
    #                 answer_response_sample_j = rationale_response_sample_j.split("<Answer>")[-1]
    #                 if "(A)" in answer_response_sample_j:
    #                     predict = "True"
    #                 elif "(B)" in answer_response_sample_j:
    #                     predict = "False"
    #                 elif "(C)" in answer_response_sample_j:
    #                     predict = "Uncertain"
    #                 else:
    #                     predict = "Unknown"
                    
    #                 if predict == label:
    #                     correct_num += 1
    #                     break
    #             rationales.append({
    #                     "premises": item['premises'],
    #                     "conclusions": item['conclusion'],
    #                     "rationale": rationale_response.strip(),
    #                     "label": item['label'],
    #                     "predict": predict,
    #                     "user_prompt": prompt
    #                 })
    #             total_num += 1
    #             print(f"{correct_num} out of {total_num} is correct!")
    #             accuracy = correct_num / total_num if total_num > 0 else 0.0
    #     else:
    #         for prompt, item, code_response in zip(batch_prompts, batch_items, batch_responses):            
    #             try:
    #                 label = item['label']
    #                 if code_response:
    #                     total_num += 1
    #                 for j in range(len(code_response)):
    #                     code_response_sample_j = code_response[j]
    #                     code_response_sample_j = code_response_sample_j.split("<PYTHON>")[-1]
    #                     code_response_sample_j = code_response_sample_j.split("</PYTHON>")[0]
    #                     code_response_sample_j = remove_incorrect_code_symbols(code_response_sample_j)
    #                     code_response_sample_j = code_response_sample_j.split("result = 'Unknown'")[0] + "result = 'Unknown'"
    #                     if not code_response_sample_j:
    #                         print("Warning: Empty code response! Counting as incorrect.")
    #                         predict = "Unknown"
    #                     else:
    #                         if is_executable(code_response_sample_j):
    #                             print("Executing code!")
    #                             predict = execute_with_timeout(code_response_sample_j, timeout_seconds=3)
    #                             num_exec += 1
    #                         else:
    #                             print("Warning: the code is not executable")
    #                             predict = "Unexecutable"
    #                     print(predict)
    #                     if str(predict) == str(label):
    #                         correct_num += 1
    #                         break
    #                 rationales.append({
    #                     "premises": item['premises'],
    #                     "conclusions": item['conclusion'],
    #                     "rationale": code_response_sample_j.strip(),
    #                     "label": label,
    #                     "predict": predict,
    #                     "user_prompt": prompt,
    #                 })

    #             except Exception as e:
    #                 print(f"Unexpected error in processing item: {e}")
    #                 predict = "Unknown"
    #             finally:
    #                 accuracy = correct_num / total_num if total_num > 0 else 0.0
    #                 print(f"{correct_num} out of {total_num} is correct! Accuracy: {accuracy:.2%}")

    with open(os.path.join(output_dir, raw_data_path), 'w') as f:
        json.dump(rationales, f, indent=4)
    print(f"Rationales saved to {os.path.join(output_dir, raw_data_path)}")

    with open(os.path.join(output_dir, accuracy_path), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Total samples: {total_num}\n")
        f.write(f"Correct predictions: {correct_num}\n")

    print(f"Accuracy: {accuracy:.4f}")

    print(f"Total samples: {total_num}")
    print(f"Correct predictions: {correct_num}")
    print(f"Accuracy report saved to {accuracy_path}")

def eval_performance(model_name_and_path, dataset_name, output_dir, save_raw_data_path, save_result_path, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None, mode='truth_table', is_chat_model=False, batch_size=16, use_fewshot=False, gpu_count=4, prompt_mode="v1", number_candidates=10):
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
    # Load test set
    print(f"Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name)
    test_dataset = dataset['validation'] 

    # Load Model  
    base_model = load_model_inference(model_name_or_path=model_name_and_path, gpu_count=gpu_count)

    if os.path.exists(os.path.join(output_dir, save_raw_data_path)):
            pass
    else:
        evaluation_batch(
                model=base_model,  # Always use the base model
                dataset=test_dataset,
                output_dir=output_dir,
                raw_data_path=save_raw_data_path, 
                accuracy_path=save_result_path,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                mode=mode,
                is_chat_model=is_chat_model, 
                batch_size=batch_size,
                use_fewshot=use_fewshot,
                prompt_mode=prompt_mode,
                number_candidates=number_candidates,
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
    parser.add_argument("--prompt_mode", type=str, default="v1", 
                        help="mode of prompts.")
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Name of the Hugging Face dataset to use (e.g., 'glue').")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Directory to save generated files and responses.")
    parser.add_argument("--save_raw_data_path", type=str, default="outputs", 
                        help="data path to save generated raw data.")
    parser.add_argument("--save_result_path", type=str, default="outputs", 
                        help="data path to save final results.")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for Inference.")
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
    parser.add_argument("--gpu_count", type=int, default=4, 
                        help="the number of gpus for inference.")
    parser.add_argument("--number_candidates", type=int, default=10, 
                        help="the number of candidates.")

    # Parse arguments
    args = parser.parse_args()
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

    eval_performance(
        model_name_and_path=args.model_name_and_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        save_raw_data_path=args.save_raw_data_path,
        save_result_path=args.save_result_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mode=args.mode,
        is_chat_model=is_chat,  
        batch_size=args.batch_size,
        use_fewshot=args.use_fewshot,
        gpu_count=args.gpu_count,
        prompt_mode=args.prompt_mode,
        number_candidates=args.number_candidates,
    )

if __name__ == "__main__":
    main()






