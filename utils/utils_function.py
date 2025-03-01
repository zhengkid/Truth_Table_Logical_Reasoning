import multiprocessing
from vllm import LLM, SamplingParams
import os
import requests
from huggingface_hub import HfApi
from datasets import load_dataset


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

def generate_responses_batch(model, user_prompts, max_tokens, temperature, top_p, top_k, stop, is_chat_model):
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
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens
    )
    if is_chat_model:
        outputs = model.chat(user_prompts, sampling_params)
    else:
        outputs = model.generate(user_prompts, sampling_params)

    responses = [output.outputs[0].text for output in outputs]
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
    
    return full_prompt

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
    

