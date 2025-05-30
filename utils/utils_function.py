import multiprocessing
from vllm import LLM, SamplingParams
import os
import requests
from huggingface_hub import HfApi
from datasets import load_dataset
import traceback
import re
import random
from collections import Counter

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

import re

def clean_markdown_code(response: str) -> str:
    response = re.sub(r"```python\s*", "", response)
    response = re.sub(r"```", "\n</code>", response)

    return response

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

        local_vars = {}
        print(repr(code_str))
        exec(code_str, local_vars, local_vars)

        return_dict["result"] = local_vars.get("result", "Unknown")
        return_dict["error"] = None
        print(return_dict["result"])
    except Exception as e:

        print("Error during execution:", e)

        return_dict["error"] = traceback.format_exc()  
        return_dict["result"] = "Unknown"

def execute_with_timeout(code_str, timeout_seconds=3):
    """
    Executes code with a timeout to prevent infinite loops or excessive execution time.

    Args:
        code_str (str): The code to be executed.
        timeout_seconds (int): Maximum execution time before termination.

    Returns:
        tuple: (result, error_message)
               - result: The execution result or "Unknown" if timed out or error.
               - error_message: None if no error, or the error traceback string.
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

        return "Unknown", "Timeout"


    result = return_dict.get("result", "Unknown")
    error_message = return_dict.get("error", None)
    return result, error_message

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
    print(outputs)
    responses = [[candidate.text for candidate in output.outputs] for output in outputs]
    return responses

def get_prompt(model, mode, prompt_mode, use_fewshot=False):
    """
    Load system prompt and few-shot examples according to modes (truth_table, code, nl).
    """
    sys_prompt_path = os.path.join('./Prompts', f'sys_prompt_star_{prompt_mode}.txt') 
    example_path = os.path.join('./Prompts', f'example_{mode}_star_{prompt_mode}.txt')

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
    if False: # "gemma" not in model:
        return (sys_prompt, fewshot_example + "\n\n" + example), (sys_prompt, example)
    else:
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
    

def parse_answer(rationale_response, mode, prompt_mode):
    predict = 'Unknown'
    if mode != 'code' or (mode == 'code' and 'final' in prompt_mode):
        tag = mode if mode!='nl' else "nl_cot"
        rationale_response = clean_markdown_code(rationale_response)
        rationale_response = rationale_response.split(f"<{tag}>")[-1]
        rationale_response = rationale_response.split("<end_of_answer>")[0] + "<end_of_answer>"
        answer_match = re.search(r'<answer>(.*?)<end_of_answer>', rationale_response, re.DOTALL)
        answer_response = answer_match.group(1).strip() if answer_match else ""
        print(answer_response) 
        match = re.search(r'\(?([A-D])\)?', answer_response)
        if match:
            extracted_answer = match.group(1)
            predict_mapping = {
                "A": "True",
                "B": "False",
                "C": "Uncertain",
            }
            predict = predict_mapping.get(extracted_answer, "Unknown")
        elif "true" in answer_response.lower() or "false" in answer_response.lower() or "uncertain" in answer_response.lower():
            if "true" in answer_response.lower():
                predict = "True"
            elif "false" in answer_response.lower():
                predict = "False"
            elif "uncertain" in answer_response.lower():
                predict = "Uncertain"
            else:
                predict = 'Unknown'

        #answer_response = rationale_response.split("<Answer>")[-1]
        #if "(A)" in answer_response:
        #    predict = "True"
        #elif "(B)" in answer_response:
        #    predict = "False"
        #elif "(C)" in answer_response:
        #    predict = "Uncertain"
        #else:
        #    predict = "Unknown"
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
            #if is_executable(rationale_response):
            print("Executing code!")
            predict, err = execute_with_timeout(rationale_response, timeout_seconds=3)
            #else:
            #print("Warning: the code is not executable")
            #predict = "Unexecutable"
            #error_message = traceback.format_exc()
            error_message = err
        except Exception as e:
            print(f"Error executing code: {e}")
            predict = "Execution Error"
            error_message = f"Error executing code: {e}"

        return rationale_response, predict, error_message
    



def post_process_batch_data_generate_rationale(
    batch_prompts_only_example,
    batch_items,
    batch_responses,
    mode,
    total_num,
    correct,
    dataset_len,
    model,
    max_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    top_k=40,
    stop=None,
    is_chat_model=False,
    max_refine=0,  # <-- New parameter for maximum refine attempts
    prompt_mode='v1'
):
    """
    For each sample, iterate over multiple candidate rationales. If any candidate
    predicts the label correctly, we record it. If all candidates are incorrect,
    we collect their error info and try self_refine multiple times.
    """
    rationales = []

    for prompt_only_example, item, rationale_response in zip(batch_prompts_only_example, batch_items, batch_responses):
        label = item['label']
        found_correct = False

        # Collect all error messages and failed responses for potential refinement
        all_error_messages = []
        all_error_programs = []
        print(label)
        max_count = 0
        # 1) Try each candidate response
        for j in range(len(rationale_response)):
            response_text = rationale_response[j]
            parsed_rationale, predict_j, error_message = parse_answer(response_text, mode, prompt_mode)
            print(predict_j)
            if max_count == 1:
                    break
            if predict_j == label:
                found_correct = True
                correct += 1
                print(f"Generated rationale for data point {total_num + 1}/{dataset_len}")
                print("correct_number:", correct)

                # Store the successful rationale
                rationales.append({
                    'prompt_id': str(total_num),
                    'prompt': prompt_only_example,
                    'messages': [
                        {"role": "user", "content": prompt_only_example},
                        {"role": "assistant", "content": parsed_rationale.strip()}
                    ],
                })
                max_count += 1

            else:
                # Store errors and incorrect candidates for potential refinement
                all_error_messages.append(error_message)
                all_error_programs.append(response_text)
                print("Filter out the data point due to poor quality.")

        # 2) If no candidate was correct, try self-refine up to max_refine times
        if not found_correct:
            refined_rationale = None
            refined_predict = None
            refined_error_message = None

            for refine_count in range(max_refine):
                refined_code = self_refine(
                    error_messages=all_error_messages,
                    error_programs=all_error_programs,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop,
                    is_chat_model=is_chat_model
                )

                # Parse the refined output
                refined_rationale, refined_predict, refined_error_message = parse_answer(refined_code, mode, prompt_mode)

                if refined_predict == label:
                    correct += 1
                    found_correct = True
                    print(f"Generated rationale for data point {total_num + 1}/{dataset_len} (via refine attempt {refine_count+1})")
                    print("correct_number:", correct)

                    # Store the refined rationale
                    rationales.append({
                        'prompt_id': str(total_num),
                        'prompt': prompt_only_example,
                        'messages': [
                            {"role": "user", "content": prompt_only_example},
                            {"role": "assistant", "content": refined_rationale.strip()}
                        ],
                    })
                    break
                else:
                    # If still incorrect, add errors and the new output to feed next iteration
                    print("Refinement attempt failed. Trying further, if within limit.")
                    all_error_messages.append(refined_error_message)
                    all_error_programs.append(refined_rationale)

            if not found_correct:
                # If all refine attempts failed, you can decide how to handle
                print("All refinements failed. No rationale recorded for this data point.")

        total_num += 1

    return rationales, correct, total_num


def post_process_batch_data_eval(batch_prompts, batch_items, batch_responses, mode, total_num, correct, model, max_tokens=2048, temperature=0.7, top_p=0.9, top_k=40, stop=None, is_chat_model=False, max_refine=0, prompt_mode='v1'):
    rationales = []
    for prompt, item, rationale_response in zip(batch_prompts, batch_items, batch_responses):
        label = item['label']
        predict_j = None

        # Collect all error messages and erroneous programs for this sample
        all_error_messages = []
        all_error_programs = []

        # Flag to indicate if a correct candidate was found
        found_correct = False

        for j in range(len(rationale_response)):
            rationale_response_sample_j = rationale_response[j]
            # parse_answer should return: (parsed_text, predicted_label, error_message)
            rationale_response_sample_j, predict_j, error_message = parse_answer(rationale_response_sample_j, mode, prompt_mode=prompt_mode)

            if predict_j == label:
                correct += 1
                found_correct = True
                break
            else:
                all_error_messages.append(error_message)
                all_error_programs.append(rationale_response_sample_j)

        # If all initial candidates fail, try refinement up to max_refine times
        refined_code = None
        refined_predict = None
        refined_error_message = None
        if not found_correct:
            for refine_count in range(max_refine):
                # Call self_refine to generate a corrected version
                refined_code = self_refine(
                    all_error_messages,
                    all_error_programs,
                    model,
                    max_tokens,
                    temperature,
                    top_p,
                    top_k,
                    stop,
                    is_chat_model
                )
                rationale_response_sample_j, refined_predict, refined_error_message = parse_answer(refined_code, mode, prompt_mode)
                predict_j = refined_predict
                
                if refined_predict == label:
                    correct += 1
                    found_correct = True
                    break
                else:
                    # Optionally add the newly refined attempt's error to the list
                    all_error_messages.append(refined_error_message)
                    all_error_programs.append(rationale_response_sample_j)
                if refined_error_message is None:
                    break
        # Choose which rationale to store in the final result.
        # If you want to store the last refined version when all attempts fail or
        # the successful refine version if it succeeds, you can do so:
        final_rationale = rationale_response_sample_j.strip()



        # # If all candidates fail to match the correct answer
        # if not found_correct:
        #     # Call self_refine to generate a corrected version
        #     refined_code = self_refine(all_error_messages, all_error_programs, model, max_tokens, temperature, top_p, top_k, stop, is_chat_model)
        #     rationale_response_sample_j, refined_predict, refined_error_message = parse_answer(refined_code, mode)
        #     predict_j = refined_predict
        #     if refined_predict == label:
        #         correct += 1

            # Optionally, you could check again if refine also failed, 
            # and decide whether to refine further or finalize an error.

        # Choose which rationale to store in the final result.
        # If you want to store the refined version, replace rationale_response_sample_j with refined_rationale.
        rationales.append({
            "premises": item['premises'],
            "conclusions": item['conclusion'],
            "rationale": final_rationale.strip(),
            "label": item['label'],
            "predict": predict_j,
            "user_prompt": prompt,
        })

        total_num += 1
        print(f"{correct} out of {total_num} is correct!")
        accuracy = correct / total_num if total_num > 0 else 0.0

    return rationales, correct, total_num, accuracy


def post_process_batch_data_eval_multi_candidate(batch_prompts, batch_items, batch_responses, mode, total_num, correct, model, max_tokens=2048, temperature=0.7, top_p=0.9, top_k=40, stop=None, is_chat_model=False, max_refine=0, prompt_mode='v1', mode_indicaters=[]):
    rationales = []
    for prompt, item, rationale_response in zip(batch_prompts, batch_items, batch_responses):
        label = item['label']
        predictions = []
        all_rationales = []

        num_modes = len(rationale_response)
        for i in range(num_modes):
            for j in range(len(rationale_response[i])):
                rationale_response_mode_i_sample_j = rationale_response[i][j]
                rationale_response_sample_ij, predict_ij, error_message = parse_answer(rationale_response_mode_i_sample_j, mode_indicaters[i], prompt_mode=prompt_mode)
                predictions.append(predict_ij)
                all_rationales.append(rationale_response_sample_ij.strip())
        
        # vote for correctness
        final_predict = majority_vote(predictions)

        if final_predict == label:
            correct += 1
        
        rationales.append({
            "premises": item['premises'],
            "conclusions": item['conclusion'],
            "rationales": all_rationales,
            "label": item['label'],
            "final_predict": final_predict,
            "predictions": predictions,
            "user_prompt": prompt,
        })

        total_num += 1
        print(f"{correct} out of {total_num} is correct!")
        accuracy = correct / total_num if total_num > 0 else 0.0

    return rationales, correct, total_num, accuracy

def post_process_batch_data_eval_sample_multiple_times(batch_prompts, batch_items, batch_responses, mode, total_num, correct, model, max_tokens=2048, temperature=0.7, top_p=0.9, top_k=40, stop=None, is_chat_model=False, max_refine=0, prompt_mode='v1'):
    rationales = []
    for prompt, item, rationale_response in zip(batch_prompts, batch_items, batch_responses):
        label = item['label']
        predictions = []
        all_rationales = []


        for i in range(len(rationale_response)):
            rationale_response_sample_i = rationale_response[i]
            rationale_response_sample_i, predict_i, error_message = parse_answer(rationale_response_sample_i, mode, prompt_mode=prompt_mode)
            predictions.append(predict_i)
            all_rationales.append(rationale_response_sample_i.strip())

        # vote for correctness
        print(label)
        print(predictions)
        if label in predictions:
            correct += 1

        rationales.append({
            "premises": item['premises'],
            "conclusions": item['conclusion'],
            "rationales": all_rationales,
            "label": item['label'],
            "predictions": predictions,
            "user_prompt": prompt,
        })
        print(len(all_rationales))
        total_num += 1
        print(f"{correct} out of {total_num} is correct!")
        accuracy = correct / total_num if total_num > 0 else 0.0

    return rationales, correct, total_num, accuracy


def majority_vote(predictions):
    """
    Selects the most common prediction from the list.
    If there is a tie, randomly selects one among the tied values.
    """
    if not predictions:
        return None  # Return None if the list is empty

    counter = Counter(predictions)
    max_count = max(counter.values())

    # Get all predictions that have the maximum count
    candidates = [key for key, count in counter.items() if count == max_count]
    print(candidates)
    # Randomly select one if there is a tie
    return random.choice(candidates)


def self_refine(
    error_messages, 
    error_programs, 
    model, 
    max_tokens=128, 
    temperature=0.7, 
    top_p=0.9, 
    top_k=40, 
    stop=None, 
    is_chat_model=False
):
    """
    Refines a failed or erroneous program by constructing a prompt that includes
    error messages and incorrect code, then calls 'generate_responses_batch'
    for a new, improved program/answer. We request a structured output format
    so we can parse the result easily.

    :param error_messages: list[str] – error messages for each failed candidate
    :param error_programs: list[str] – corresponding code/predictions that failed
    :param model: LLM or chat model object
    :param max_tokens: int – max tokens for generation
    :param temperature: float – sampling temperature
    :param top_p: float – nucleus sampling parameter
    :param top_k: int – top-k sampling parameter
    :param stop: list[str] or None – optional stop tokens
    :param is_chat_model: bool – whether the model is a chat model
    :return: str – the refined program/answer from the model
    """

    # 1) Construct a single prompt that summarizes all errors and attempts
    combined_errors = []
    for i, (msg, prog) in enumerate(zip(error_messages, error_programs)):
        combined_errors.append(
            f"[Candidate {i+1}]\nError Message: {msg}\nErroneous Program:\n{prog}\n"
        )

    # 2) We add instructions to provide the final solution strictly within <SOLUTION>...</SOLUTION> tags.
    refine_prompt = (
        "We have the following erroneous candidates for the same question.\n"
        "Each includes an error message and the code or reasoning that failed.\n\n"
        + "\n".join(combined_errors)
        + "\nPlease provide a corrected or refined solution, free of these errors. You mush solve the errors by modifying the erroneeous codes."
        "\n\nIMPORTANT: Your answer must follow this exact format:\n"
        "<PYTHON>\n"
        "Your refined code.\n"
        "</PYTHON>\n"
        "Only include refined code between these tags and do not include other thing between these tags.\n"
    )

    refine_prompt = [{"role": "user","content": refine_prompt}]

    # 3) Use generate_responses_batch to have your model produce a refined candidate
    refined_candidates = generate_responses_batch(
        model=model,
        user_prompts=[refine_prompt],  # only one prompt here
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop=stop,
        is_chat_model=is_chat_model,
        number_candidates=1
     )

     # refined_candidates is a list of lists because generate_responses_batch
     # returns multiple outputs for multiple prompts. We only have one prompt,
     # so we take refined_candidates[0][0].
    raw_refined_code = refined_candidates[0][0]

     # 4) (Optional) You could parse out the text inside <SOLUTION>...</SOLUTION> immediately here,
     # or do it later in your parse_answer logic. For example:
     # refined_code = extract_solution_content(raw_refined_code)

    return raw_refined_code



















