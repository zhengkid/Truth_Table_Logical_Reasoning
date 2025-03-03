import multiprocessing
from vllm import LLM, SamplingParams
import os
import requests
from huggingface_hub import HfApi
from datasets import load_dataset

import re

def remove_incorrect_code_symbols(text):
    """
    Removes incorrect ‘’‘python and ’‘’ symbols used for code blocks.

    Args:
        text (str): The input text containing incorrectly formatted code blocks.

    Returns:
        str: Cleaned text with proper formatting.
    """
    # Replace incorrect opening ‘’‘python with correct triple backticks ```
    text = re.sub(r"[‘’`]{3}python", "", text)

    # Replace incorrect closing ’‘’ with correct triple backticks ```
    text = re.sub(r"[‘’`]{3}", "", text)

    return text

class TimeoutException(Exception):
    pass

def run_code(code_str, return_dict):
    """
    Executes the given code string in a controlled environment and stores the result.

    Args:
        code_str (str): The code to be executed.
        return_dict (dict): A shared dictionary to store the execution result.

    Returns:
        None
    """
    try:
        exec(code_str, globals())
        return_dict["result"] = globals().get("result", "Unknown")
    except Exception as e:
        print("Error during execution:", e)
        return_dict["result"] = "Unknown"

def execute_with_timeout(code_str, timeout_seconds=3):
    """
    Executes code with a timeout to prevent infinite loops or excessive execution time.

    Args:
        code_str (str): The code to be executed.
        timeout_seconds (int): Maximum execution time before termination.

    Returns:
        str: The execution result or "Unknown" if timed out.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    proc = multiprocessing.Process(target=run_code, args=(code_str, return_dict))
    proc.start()
    proc.join(timeout_seconds)  

    if proc.is_alive():
        print("Timeout reached! Terminating process...")
        proc.terminate()
        proc.join() 
        return "Unknown" 

    return return_dict.get("result", "Unknown") 

def load_model_inference(model_name_or_path='gemma-2-9b', gpu_count=4):
    """
    Loads the specified model for inference using vLLM.

    Args:
        model_name_or_path (str): The model name or local path.

    Returns:
        LLM: Loaded model instance.
    """
    model = LLM(model=model_name_or_path, tensor_parallel_size=gpu_count)
    return model

def generate_responses_batch(model, user_prompts, max_tokens, temperature, top_p, top_k, stop, is_chat_model, number_candidates):
    """
    Generates responses in batch from the model based on user prompts.

    Args:
        model (LLM): The loaded language model.
        user_prompts (list): A list of user prompts.
        max_tokens (int): Maximum number of tokens for generated responses.
        temperature (float): Sampling temperature for response generation.
        top_p (float): Nucleus sampling parameter.
        top_k (int): Top-k sampling parameter.
        stop (list or None): Optional stop words for generation.
        is_chat_model (bool): Whether the model is a chat model.

    Returns:
        list: A list of generated responses.
    """
    sampling_params = SamplingParams(
        n=number_candidates,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens
    )
    if is_chat_model:
        outputs = model.chat(user_prompts, sampling_params)
    else:
        outputs = model.generate(user_prompts, sampling_params)

    responses = [[candidate.text for candidate in output.outputs] for output in outputs]
    return responses

def get_prompt(mode, prompt_mode, use_fewshot=False):
    """
    Load system prompt and few-shot examples according to modes (truth_table, code, nl).
    """
    sys_prompt_path = os.path.join('./Prompts', f'sys_prompt_{mode}_star_{prompt_mode}.txt') 
    example_path = os.path.join('./Prompts', f'example_{mode}_star.txt')

    with open(sys_prompt_path, encoding="utf-8") as f:
        sys_prompt = f.read()
    with open(example_path, encoding="utf-8") as f:
        example = f.read()
 

    if use_fewshot:
        fewshot_path = os.path.join('./Prompts', f'prompt_{mode}_star_{prompt_mode}.txt')
        with open(fewshot_path, encoding="utf-8") as f:
            fewshot_example = f.read()
        full_prompt = sys_prompt + '\n\n' + fewshot_example + '\n\n' + example
    else:
        full_prompt = sys_prompt + '\n\n' + example
    full_prompt_without_few_shot = sys_prompt + '\n\n' + example
    return full_prompt, full_prompt_without_few_shot

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

def is_executable(code):
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False
    

def check_huggingface_repo_exists(huggingface_repo: str) -> bool:
    api = HfApi()
    try:
        api.repo_info(repo_id=huggingface_repo, repo_type="dataset")  # Change to "model" if checking a model
        return True
    except requests.exceptions.HTTPError:
        return False
    

def parse_answer(rationale_response, mode):
    predict = None
    if mode != 'code':
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
        return rationale_response, predict, None
    else:
        error_message = None
        rationale_response = rationale_response.split("<PYTHON>")[-1]
        rationale_response = rationale_response.split("</PYTHON>")[0]
        if not rationale_response.strip():
            print("Warning: Empty code response! Counting as incorrect.")
            error_message = "Warning: Empty code response! Counting as incorrect."
            return rationale_response, "Unknown", error_message
        try:
            if is_executable(rationale_response):
                print("Executing code!")
                predict = execute_with_timeout(rationale_response, timeout_seconds=3)
            else:
                print("Warning: the code is not executable")
                predict = "Unexecutable"
        except Exception as e:
            print(f"Error executing code: {e}")
            predict = "Execution Error"
            error_message = f"Error executing code: {e}"
        return rationale_response, predict, error_message
    

def post_process_batch_data_generate_rationale(batch_prompts_only_example, batch_items, batch_responses, mode, total_num, correct, dataset_len):
    rationales = []
    for prompt_only_example, item, rationale_response in zip(batch_prompts_only_example, batch_items, batch_responses):
        label = item['label']
        for j in range(len(rationale_response)):
            rationale_response_sample_j = rationale_response[j]
            rationale_response_sample_j, predict_j, error_message = parse_answer(rationale_response_sample_j, mode)
            if predict_j == label:
                rationales.append({
                    'prompt_id': str(total_num),
                    'prompt': prompt_only_example, #prompt[0]['content'],
                    'messages': [{"role": "user","content": prompt_only_example}, { "content":rationale_response_sample_j.strip(), "role": "assistant" }],
                })
                print(f"Generated rationale for data point {total_num + 1}/{dataset_len}")
                correct += 1
                print("correct_number:", correct)
                break
            else:
                print(f"Filter out the data point due to poor quality.") 
        total_num += 1
    return rationales, correct, total_num

def post_process_batch_data_eval(batch_prompts, batch_items, batch_responses, mode, total_num, correct):
    rationales = []
    for prompt, item, rationale_response in zip(batch_prompts, batch_items, batch_responses):
        label = item['label']
        predict_j = None
        for j in range(len(rationale_response)):
            rationale_response_sample_j = rationale_response[j]
            rationale_response_sample_j, predict_j, error_message = parse_answer(rationale_response_sample_j, mode)
            if predict_j == label:
                correct += 1
                break
        rationales.append({
                        "premises": item['premises'],
                        "conclusions": item['conclusion'],
                        "rationale": rationale_response_sample_j.strip(),
                        "label": item['label'],
                        "predict": predict_j,
                        "user_prompt": prompt,
                    })
        total_num += 1
        print(f"{correct} out of {total_num} is correct!")
        accuracy = correct / total_num if total_num > 0 else 0.0
    return rationales, correct, total_num, accuracy
