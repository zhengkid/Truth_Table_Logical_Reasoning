import random
import numpy as np
import torch
import tqdm
import argparse
from datasets import Dataset, DatasetDict
from utils.utils_function import (
    get_prompt,
    load_model_inference,
    generate_responses_batch,
    post_process_batch_data,
    check_huggingface_repo_exists,
    obtain_seed_dataset,
)

def generate_rationales(model, dataset, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None, mode='truth_table', is_chat_model=False, batch_size=16,use_fewshot=True, huggingface_repo="", prompt_mode='v1', number_candidates=10):
    """
    Generate rationales for each data point in the dataset.
    """
    rationales = []
    total_num, correct, dataset_len = 0, 0, len(dataset)
    # load prompts 0: full prompt with few shot 1: prompt without few shot
    full_prompt, full_prompt_only_example = get_prompt(mode=mode, prompt_mode=prompt_mode, use_fewshot=use_fewshot)
    # Prepare prompts for datasets
    for batch_start in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch_dataset = dataset[batch_start: batch_start + batch_size]
        batch_prompts = []
        batch_prompts_only_example = []
        batch_items = []
        # Accumulate batch data 
        for item in batch_dataset:
            premises = item.get("premises", "")
            conclusions = item.get("conclusion", "")
            prompt = full_prompt.format(Premises=premises, Conclusions=conclusions)
            prompt_only_example = full_prompt_only_example.format(Premises=premises, Conclusions=conclusions)
            if is_chat_model:
                prompt = [{"role": "user","content": prompt}]
            batch_prompts.append(prompt)
            batch_prompts_only_example.append(prompt_only_example)
            batch_items.append(item)

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
        # Post-process batch data
        batch_rationales, correct, total_num = post_process_batch_data(batch_prompts_only_example, batch_items, batch_responses, mode, total_num, correct, dataset_len)
        rationales.extend(batch_rationales)

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

    # for item in dataset:
    #     premises = item.get("premises", "")
    #     conclusions = item.get("conclusion", "")
    #     prompt = full_prompt.format(Premises=premises, Conclusions=conclusions)
    #     prompt_only_example = full_prompt_only_example.format(Premises=premises, Conclusions=conclusions)
    #     if is_chat_model:
    #         prompt = [{"role": "user","content": prompt}]
    #     prompts.append(prompt)
    #     prompts_only_example.append(prompt_only_example)
    #     dataset_list.append(item)

    # for batch_start in tqdm.tqdm(range(0, len(dataset_list), batch_size)):
    #     batch_prompts = prompts[batch_start: batch_start + batch_size]
    #     batch_prompts_only_example = prompts_only_example[batch_start: batch_start + batch_size]
    #     batch_items = dataset_list[batch_start: batch_start + batch_size]
    #     try:
    #         with torch.no_grad():
    #             batch_responses = generate_responses_batch(
    #                 model=model,
    #                 user_prompts=batch_prompts,
    #                 max_tokens=max_tokens,
    #                 temperature=temperature,
    #                 top_p=top_p,
    #                 top_k=top_k,
    #                 stop=stop,
    #                 is_chat_model=is_chat_model
    #             )
    #     except Exception as e:
    #         print(f"Error generating responses for batch starting at index {batch_start}: {e}")
    #         tqdm.tqdm.update(1)
    #         continue 
    #     if mode != 'code':
    #         for prompt, prompt_only_example, item, rationale_response in zip(batch_prompts, batch_prompts_only_example, batch_items, batch_responses):            
    #             label = item['label']
    #             for j in range(len(rationale_response)):
    #                 rationale_response_sample_j = rationale_response[j]
    #                 rationale_response_sample_j, predict_j = parse_answer(rationale_response_sample_j, mode)
    #                 if predict_j == label:
    #                     rationales.append({
    #                         'prompt_id': str(total_num),
    #                         'prompt': prompt_only_example, #prompt[0]['content'],
    #                         'messages': [{"role": "user","content": prompt_only_example}, { "content":rationale_response_sample_j.strip(), "role": "assistant" }],
    #                     })
    #                     print(f"Generated rationale for data point {total_num + 1}/{len(dataset)}")
    #                     correct += 1
    #                     print("correct_number:", correct)
    #                     break
    #                 else:
    #                     print(f"Filter out the data point due to poor quality.") 
    #             total_num += 1
    #     else:
    #         for prompt, prompt_only_example, item, code_response in zip(batch_prompts, batch_prompts_only_example, batch_items, batch_responses):            
    #             try:
    #                 label = item['label']
    #                 code_response, predict = parse_answer(code_response, mode)
    #                 if predict == label:
    #                     rationales.append({
    #                         'prompt_id': str(total_num),
    #                         'prompt': prompt_only_example, #prompt[0]['content'],
    #                         'messages': [{"role": "user","content": prompt_only_example}, { "content":code_response.strip(), "role": "assistant" }],
    #                     })
    #                     print(f"Generated rationale for data point {total_num + 1}/{len(dataset)}")
    #                     correct += 1
    #                     print("correct_number:", correct)
    #                 else:
    #                     print(f"Filter out the data point due to poor quality.")
    #                 total_num += 1
    #             except Exception as e:
    #                 print(f"Unexpected error in processing item: {e}")
    #                 print(f"Filter out the data point due to poor quality.")
    #                 total_num += 1
   

################################################# Star Pipeline #############################################################

def generate_rationale_data(model_name_and_path, dataset_name, n_samples=200, batch_size=16, seed=42, max_tokens=512, temperature=1.0, top_p=0.9, top_k=50, stop=None, mode='truth_table', is_chat_model=False, use_fewshot=False, huggingface_repo="", gpu_count=4, prompt_mode='v1', number_candidates=10):
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

    # Obtain seed training set and test set
    dataset = obtain_seed_dataset(dataset_name, n_samples, seed)
    model = load_model_inference(model_name_or_path=model_name_and_path,  gpu_count=gpu_count)

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
    parser.add_argument("--dataset_name", type=str, required=True, 
                        help="Name of the Hugging Face dataset to use (e.g., 'glue').")
    parser.add_argument("--huggingface_repo", type=str, default="", 
                        help="huggingface_repo.")
    parser.add_argument("--prompt_mode", type=str, default="v1", 
                        help="mode of prompts.")
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
    parser.add_argument("--gpu_count", type=int, default=4, 
                        help="the number of gpus for inference.")
    parser.add_argument("--number_candidates", type=int, default=10, 
                        help="the number of candidates.")
    

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

    if not check_huggingface_repo_exists(args.huggingface_repo):
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
            gpu_count=args.gpu_count,
            prompt_mode=arg.prompt_mode,
            number_candidates=args.number_candidates,
        )
    else:
        print(f"Dataset {args.huggingface_repo} already exists. Skipping generation.")

if __name__ == "__main__":
    main()






