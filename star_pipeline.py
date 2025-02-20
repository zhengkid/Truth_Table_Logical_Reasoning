import random
import numpy as np
import os
import re
import json
import time
import torch
import tqdm
import argparse
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, default_data_collator,TrainingArguments,DataCollatorForLanguageModeling
from functools import partial
import deepspeed
from vllm import LLM, SamplingParams
import gc
import socket
from vllm.distributed.parallel_state import destroy_model_parallel
torch.use_deterministic_algorithms(False)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'       
##########################################################Begin: Formating Prompts##########################################################################
# Prompting Truth Table 
def get_sys_prompt_rational_truth_table():
    file_path = os.path.join('./Prompts', 'sys_prompt_truth_table_star.txt')
    with open(file_path) as f:
        sys_prompt = f.read()
    return sys_prompt

def get_few_shot_prompt_rational_truth_table():
    file_path = os.path.join('./Prompts', 'prompt_truth_table_star.txt')
    with open(file_path) as f:
        in_context_examples = f.read()
    return in_context_examples

def get_prompt_rational_truth_table():
    fewshot_example = get_few_shot_prompt_rational_truth_table()
    sys_prompt = get_sys_prompt_rational_truth_table()
    full_prompt = sys_prompt + "\n\n" + fewshot_example
    return full_prompt

# Prompting Code
def get_sys_prompt_rational_code():
    file_path = os.path.join('./Prompts', 'sys_prompt_code_star.txt')
    with open(file_path) as f:
        sys_prompt = f.read()
    return sys_prompt

def get_few_shot_prompt_rational_code():
    file_path = os.path.join('./Prompts', 'prompt_code_star.txt')
    with open(file_path) as f:
        in_context_examples = f.read()
    return in_context_examples

def get_prompt_rational_code():
    fewshot_example = get_few_shot_prompt_rational_code()
    sys_prompt = get_sys_prompt_rational_code()
    full_prompt = sys_prompt + "\n\n" + fewshot_example
    return full_prompt


# Prompting nl
def get_sys_prompt_rational_nl():
    file_path = os.path.join('./Prompts', 'sys_prompt_nl_star.txt')
    with open(file_path) as f:
        sys_prompt = f.read()
    return sys_prompt

def get_few_shot_prompt_rational_nl():
    file_path = os.path.join('./Prompts', 'prompt_nl_star.txt')
    with open(file_path) as f:
        in_context_examples = f.read()
    return in_context_examples

def get_prompt_rational_nl():
    fewshot_example = get_few_shot_prompt_rational_nl()
    sys_prompt = get_sys_prompt_rational_nl()
    full_prompt = sys_prompt + "\n\n" + fewshot_example
    return full_prompt

##########################################################Code for Sampling Data##########################################################################
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
    test_dataset = dataset['validation'] 

    # Shuffle and select a subset
    print(f"Selecting {num_samples} samples from the dataset...")
    seed_dataset = train_dataset.shuffle(seed=seed).select(range(num_samples))
    print(f"Seed dataset obtained with {len(seed_dataset)} samples.")
    return seed_dataset,test_dataset

##########################################################Load Model, Tokenizer##########################################################################

def load_model_and_tokenizer(model_name_or_path='gemma-2-9b', low_cpu_mem_usage=True, use_flash_attention_2=False, torch_dtype='fp16'):
    """
    Load a pre-trained model from Hugging Face Transformers.
    
    Args:
        model_name (str): The name or path of the model.
        device (str, optional): The device to load the model onto ('cuda', 'cpu', or 'mps').
                                If None, it selects GPU if available, otherwise CPU.
    
    Returns:
        model: The loaded model.
        device: The device on which the model is loaded.
    """
    model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="auto",
                low_cpu_mem_usage=low_cpu_mem_usage,
                use_flash_attention_2=use_flash_attention_2,
                torch_dtype="auto",
            )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # set eos token and pad token 
    tokenizer.padding_side='left'
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_model_inference(model_name_or_path='gemma-2-9b'):
    gpu_count = torch.cuda.device_count()
    model = LLM(model=model_name_or_path, tensor_parallel_size=1)
    return model

##########################################################Code for Training Data Preparation##########################################################################
def convert_to_custom_format(input_data):
    """
    Convert the data into a custom format.
    Args:
        input_data (list): A list of data points, each containing premises, conclusions, rationale, and label.
    Returns:
        list: Converted data in the required format.
    """
    converted_data = []
    
    for item in input_data:
        messages = [
            {
                "content": item['user_prompt'],
                "role": "user"
            },
            {
                "content": item["rationale"],
                "role": "assistant"
            }
        ]
        converted_data.append({"messages": messages})
    
    return converted_data

##########################################################Code for Training##########################################################################
def preprocess_function(examples, tokenizer, is_chat_model):
    """
    Process dataset into chat-style format for instruction tuning.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []

    for user_prompt, rationale in zip(examples["user_prompt"], examples["rationale"]):


        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": rationale}
        ]

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=4096)
        #print(len(model_inputs["input_ids"]))
        if model_inputs["input_ids"][-1] != tokenizer.eos_token_id:
            model_inputs["input_ids"].append(tokenizer.eos_token_id)
            model_inputs["attention_mask"].append(1)
        all_input_ids.append(model_inputs["input_ids"])
        all_attention_mask.append(model_inputs["attention_mask"])
        all_labels.append(model_inputs["input_ids"].copy())
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }

def finetune(client, dataset_path, output_dir, n_epochs=4, batch_size=16, micro_batch_size=1, learning_rate=1e-5, is_chat_model=False):
    """
    Fine-tune a Hugging Face model using full parameter fine-tuning.
    """
    model, tokenizer = client
    dataset = load_dataset("json", data_files=dataset_path)['train']
    #print(dataset)
    #print(dataset[0])
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer, is_chat_model), batched=True,remove_columns=["label"])
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        save_strategy="epoch",
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=batch_size // micro_batch_size,
        num_train_epochs=n_epochs,
        learning_rate=learning_rate,
        save_total_limit=1,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        push_to_hub=False,
        full_determinism=True,
        #deepspeed=ds_config
    )
    
    data_collator = default_data_collator
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")
    return model



##########################################################Code for Evaluation##########################################################################

def evaluation_batch(model, dataset, output_dir, raw_data_path, accuracy_path,
               max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None,
               mode='truth_table', is_chat_model=False, batch_size=16):
    rationales = []
    correct_num = 0
    total_num = 0   

    rationale_prompt = {
        'truth_table': get_prompt_rational_truth_table(),
        'code': get_prompt_rational_code(),
        'nl': get_prompt_rational_nl()
    }.get(mode, "")

    prompts = []
    for item in dataset:
        premises = item.get("premises", "")
        conclusions = item.get("conclusion", "")
        prompt = rationale_prompt.format(Premises=premises, Conclusions=conclusions)
        if is_chat_model:
            prompt = [  {
                        "role": "user",
                        "content": prompt
                        },
                        ]
        prompts.append(prompt)

    for batch_start in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        batch_items = dataset[batch_start: batch_start + batch_size]
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
            batch_premises = batch_items['premises']
            batch_conclusion = batch_items['conclusion']
            batch_label = batch_items['label']
            for prompt, premise, conclusion, label, rationale_response in zip(batch_prompts, batch_premises, batch_conclusion, batch_label, batch_responses):            
                #print(rationale_response)
                rationale_response = rationale_response.split("<Reasoning>")[-1]
                rationale_response = rationale_response.split("</Answer>")[0] + "</Answer>"
                print(rationale_response)
                if "(A)" in rationale_response:
                    predict = "True"
                elif "(B)" in rationale_response:
                    predict = "False"
                elif "(C)" in rationale_response:
                    predict = "Uncertain"
                else:
                    predict = "Unknown"
                rationales.append({
                    "premises": premise,
                    "conclusions": conclusion,
                    "rationale": rationale_response.strip(),
                    "label": label,
                    "predict": predict,
                    "user_prompt": prompt,
                })

                if predict == label:
                    correct_num += 1
                total_num += 1
                print(f"{correct_num} out of {total_num} is correct!")
                accuracy = correct_num / total_num if total_num > 0 else 0.0
        else:
            batch_premises = batch_items['premises']
            batch_conclusion = batch_items['conclusion']
            batch_label = batch_items['label']
            for prompt, premise, conclusion, label, code_response in zip(batch_prompts, batch_premises, batch_conclusion, batch_label, batch_responses):            
                code_response = code_response.split("<PYTHON>")[-1]
                code_response = code_response.split("</PYTHON")[0]
                globals_dict = globals().copy()
                #exec(code_response, globals_dict)
                exec(code_response)
                predict = locals().get("result")
                rationales.append({
                    "premises": premise,
                    "conclusions": conclusion,
                    "rationale": code_response.strip(),
                    "label": label,
                    "predict": predict,
                    "user_prompt": prompt,
                })

                if predict == label:
                    correct_num += 1
                total_num += 1
                print(f"{correct_num} out of {total_num} is correct!")
                accuracy = correct_num / total_num if total_num > 0 else 0.0


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

def generate_rationales(model, dataset, output_dir, output_file, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None, mode='truth_table', is_chat_model=False, batch_size=16):
    """
    Generate rationales for each data point in the dataset.
    """
    rationales = []
    total_num = 0   
    print(model)
    rationale_prompt = {
        'truth_table': get_prompt_rational_truth_table(),
        'code': get_prompt_rational_code(),
        'nl': get_prompt_rational_nl()
    }.get(mode, "")

    prompts = []
    for item in dataset:
        premises = item.get("premises", "")
        conclusions = item.get("conclusion", "")
        prompt = rationale_prompt.format(Premises=premises, Conclusions=conclusions)
        if is_chat_model:
            prompt = [  {
                        "role": "user",
                        "content": prompt
                        },
                        ]
        prompts.append(prompt)

    for batch_start in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        batch_items = dataset[batch_start: batch_start + batch_size]
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
            batch_premises = batch_items['premises']
            batch_conclusion = batch_items['conclusion']
            batch_label = batch_items['label']
            for prompt, premise, conclusion, label, rationale_response in zip(batch_prompts, batch_premises, batch_conclusion, batch_label, batch_responses):            
                #print(rationale_response)
                rationale_response = rationale_response.split("<Reasoning>")[-1]
                rationale_response = rationale_response.split("</Answer>")[0] + "</Answer>"
                print(rationale_response)
                if "(A)" in rationale_response:
                    predict = "True"
                elif "(B)" in rationale_response:
                    predict = "False"
                elif "(C)" in rationale_response:
                    predict = "Uncertain"
                else:
                    predict = "Unknown"

                if predict == label:
                    rationales.append({
                        "premises": premise,
                        "conclusions": conclusion,
                        "rationale": rationale_response.strip(),
                        'label': label,
                        'user_prompt': prompt,
                    })
                    print(f"Generated rationale for data point {total_num + 1}/{len(dataset)}")
                else:
                    print(f"Filter out the data point due to poor quality.")
                
                total_num += 1

        else:
            batch_premises = batch_items['premises']
            batch_conclusion = batch_items['conclusion']
            batch_label = batch_items['label']
            for prompt, premise, conclusion, label, code_response in zip(batch_prompts, batch_premises, batch_conclusion, batch_label, batch_responses):            
                code_response = code_response.split("<PYTHON>")[-1]
                code_response = code_response.split("</PYTHON")[0]
                exec(code_response)
                predict = locals().get("result")

                if predict == label:
                    rationales.append({
                        "premises": premise,
                        "conclusions": conclusion,
                        "rationale": rationale_response.strip(),
                        'label': label,
                        'user_prompt': prompt,
                    })
                    print(f"Generated rationale for data point {i + 1}/{len(dataset)}")
                else:
                    print(f"Filter out the data point due to poor quality.")
                total_num += 1
    with open(os.path.join(output_dir, output_file), 'w') as f:
        json.dump(rationales, f, indent=4)
    print(f"Rationales saved to {os.path.join(output_dir, output_file)}")

################################################# Star Pipeline #############################################################

def star_pipeline_base_reset(model_name_and_path, dataset_name, output_dir, n_samples=200, n_outer_loops=10, n_epochs=4,
                             batch_size=16, micro_batch_size=1, learning_rate=1e-5, seed=42, max_tokens=512, temperature=1.0, test_temperature=0.7, top_p=0.9, top_k=50, stop=None, mode='truth_table', is_chat_model=False):
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

    # Load Base Model and Tokenizer
    #base_model, tokenizer = load_model_and_tokenizer(model_name_or_path=model_name_and_path)

    outer_loop_responses = []
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    # Obtain seed training set and test set
    dataset, test_dataset = obtain_seed_dataset(dataset_name, n_samples, seed)
    
    # Step -1: Evaluate few-shot perfomrnace with different ideas
    # Load Model 
    init_model_name = model_name_and_path
    base_model = load_model_inference(model_name_or_path=init_model_name)
    rationale_file = f"rationales_{mode}_{0}.jsonl"
    test_rationale_file = model_name_and_path.split('/')[-1] + f"-{mode}-r{0}-Raw.jsonl"
    test_accuracy_file = model_name_and_path.split('/')[-1] + f"-{mode}-r{0}-Result.jsonl"
    if os.path.exists(os.path.join(output_dir, test_rationale_file)):
            pass
    else:
        #pass
        evaluation_batch(
                model=base_model,  # Always use the base model
                dataset=test_dataset,
                output_dir=output_dir,
                raw_data_path=test_rationale_file, 
                accuracy_path=test_accuracy_file,
                max_tokens=max_tokens,
                temperature=test_temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                mode=mode,
                is_chat_model=is_chat_model,  
        )

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    destroy_model_parallel()
    del base_model.llm_engine.model_executor
    del base_model # Isn't necessary for releasing memory, but why not
    gc.collect()
    torch.cuda.empty_cache()
    #torch.distributed.destroy_process_group()
    import ray
    ray.shutdown()

    model_name = init_model_name
    print(model_name)
    for n in range(1, n_outer_loops+1):
        print(f"--- Outer Loop {n} ---")
        # Step 1: Perform rationale generation
        print("Generating rationales...")
        model = load_model_inference(model_name_or_path=model_name)
        rationale_file = f"rationales_{mode}_{n}.jsonl"
        test_rationale_file = model_name_and_path.split('/')[-1] + f"-{mode}-r{n}-Raw.jsonl"
        test_accuracy_file = model_name_and_path.split('/')[-1] + f"-{mode}-r{n}-Result.jsonl"
        finetune_response_save_path = f"fine_tuning_{mode}_{batch_size}_{learning_rate}_round_{n}"
        if os.path.exists(os.path.join(output_dir, rationale_file)):
            pass
        else:
            generate_rationales(
                model=model,
                dataset=dataset,
                output_dir=output_dir,
                output_file=rationale_file,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                mode=mode,
                is_chat_model=is_chat_model,  
            )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        destroy_model_parallel()
        del model.llm_engine.model_executor
        del model # Isn't necessary for releasing memory, but why not
        gc.collect()
        torch.cuda.empty_cache()
        #torch.distributed.destroy_process_group()
        import ray
        ray.shutdown()



        # Step 2: Fine-tune the base model with rationalized datasets
        print("Fine-tuning base model...")
        model, tokenizer = load_model_and_tokenizer(model_name_or_path=init_model_name)
        trainin_data_path = rationale_file#.split('.')[0] + "_train." +  rationale_file.split('.')[1]
        model = finetune(
            client=[model, tokenizer],
            dataset_path=os.path.join(output_dir, trainin_data_path),
            output_dir=os.path.join(output_dir, finetune_response_save_path),
            n_epochs=n_epochs,
            batch_size=batch_size,
            micro_batch_size=micro_batch_size,
            learning_rate=learning_rate,
            is_chat_model=is_chat_model,  
        )
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(20)
        # Step 4: Fine-tune the base model with rationalized datasets
        model_name = os.path.join(output_dir, finetune_response_save_path)
        model = load_model_inference(model_name_or_path=model_name)
        evaluation_batch(
                model=model,  # Always use the base model
                dataset=test_dataset,
                output_dir=output_dir,
                raw_data_path=test_rationale_file, 
                accuracy_path=test_accuracy_file,
                max_tokens=max_tokens,
                temperature=test_temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                mode=mode,
                is_chat_model=is_chat_model,
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        destroy_model_parallel()
        del model.llm_engine.model_executor
        del model # Isn't necessary for releasing memory, but why not
        gc.collect()
        torch.cuda.empty_cache()
        #torch.distributed.destroy_process_group()
        import ray
        ray.shutdown()
    return outer_loop_responses
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
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Directory to save generated files and responses.")
    parser.add_argument("--n_samples", type=int, default=200, 
                        help="Number of samples to use from the dataset.")
    parser.add_argument("--n_outer_loops", type=int, default=10, 
                        help="Number of outer-loop iterations for the STaR pipeline.")
    parser.add_argument("--n_epochs", type=int, default=4, 
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for fine-tuning.")
    parser.add_argument("--micro_batch_size", type=int, default=16, 
                        help="Mirco Batch size for fine-tuning.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Learning rate for fine-tuning.")
    parser.add_argument("--lora", action="store_true", 
                        help="Enable LoRA fine-tuning.")
    parser.add_argument("--lora_r", type=int, default=8, 
                        help="Rank for LoRA adapter weights.")
    parser.add_argument("--lora_alpha", type=int, default=8, 
                        help="Alpha value for LoRA training.")
    parser.add_argument("--lora_dropout", type=float, default=0.0, 
                        help="Dropout probability for LoRA layers.")
    parser.add_argument("--max_tokens", type=int, default=512, 
                        help="Maximum number of tokens for generated responses.")
    parser.add_argument("--temperature", type=float, default=1, 
                        help="Sampling temperature for generation.")
    parser.add_argument("--test_temperature", type=float, default=0.7,
                        help="test temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p (nucleus) sampling parameter.")
    parser.add_argument("--top_k", type=int, default=50, 
                        help="Top-k sampling parameter.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility.")

    # Parse arguments
    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()
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
    star_pipeline_base_reset(
        model_name_and_path=args.model_name_and_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        n_outer_loops=args.n_outer_loops,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        test_temperature=args.test_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mode=args.mode,
        is_chat_model=is_chat,  
    )

if __name__ == "__main__":
    main()






