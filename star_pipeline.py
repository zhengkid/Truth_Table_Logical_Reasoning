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
def preprocess_function(examples, tokenizer):
    """
    Process dataset into chat-style format for instruction tuning.
    """
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    user_prompt = examples.get("user_prompt", "")
    assistant_response = examples.get("rationale", "")

    for user_prompt, rationale in zip(examples["user_prompt"], examples["rationale"]):
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": rationale}
        ]

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)
        print(len(model_inputs["input_ids"]))
        all_input_ids.append(model_inputs["input_ids"])
        all_attention_mask.append(model_inputs["attention_mask"])
        all_labels.append(model_inputs["input_ids"].copy())
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels,
    }

def finetune(client, dataset_path, output_dir, n_epochs=4, batch_size=16, learning_rate=1e-5):
    """
    Fine-tune a Hugging Face model using full parameter fine-tuning.
    """
    model, tokenizer = client
    dataset = load_dataset("json", data_files=dataset_path)['train']
    #print(dataset)
    #print(dataset[0])
    tokenized_dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True,remove_columns=["label"])
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        learning_rate=learning_rate,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        push_to_hub=False,
        #fp16=torch.cuda.is_available(),
        full_determinism=True
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
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
def evaluation(model, tokenizer, dataset, output_dir, raw_data_path, accuracy_path, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None, mode='truth_table', is_chat_model=False):
    """
    Evaluate model performance on a dataset.
    """
    rationales = []
    correct_num = 0
    total_num = 0
    print(mode)
    rationale_prompt = {
        'truth_table': get_prompt_rational_truth_table(),
        'code': get_prompt_rational_code(),
        'nl': get_prompt_rational_nl()
    }.get(mode, "")
    
    for i, item in tqdm.tqdm(enumerate(dataset)):
        premises = item.get("premises", "")
        conclusions = item.get("conclusion", "")
        label = item.get("label", "")  
        
        prompt = rationale_prompt.format(Premises=premises, Conclusions=conclusions)
        #print(prompt) 
        try:
            rationale_response = generate_response(
                model=model,
                tokenizer=tokenizer,
                user_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k, stop=stop, is_chat_model=True
            )
            print(rationale_response)
            rationale_response = rationale_response.split("<Answer>")[-1]
            print(rationale_response) 
            if "(A)" in rationale_response:
                predict = "True"
            elif "(B)" in rationale_response:
                predict = 'False'
            elif "(C)" in rationale_response:
                predict = 'Uncertain'
            else:
                predict = 'Unknown'

            rationales.append({
                "premises": premises,
                "conclusions": conclusions,
                "rationale": rationale_response.strip(),
                'label': label,
                'predict': predict,
                'user_prompt': prompt,
            })
            
            if predict == label:
                correct_num += 1
            total_num += 1

            print(f"{correct_num} out of {total_num} is correct!")
        except Exception as e:
            print(f"Error generating rationale for data point {i + 1}: {e}")
            continue

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

def generate_responses_batch(model, tokenizer, prompts, max_tokens, temperature, top_p, top_k, stop, is_chat_model):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id if stop is None else tokenizer.convert_tokens_to_ids(stop)
    )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses

def evaluation_batch(model, tokenizer, dataset, output_dir, raw_data_path, accuracy_path,
               max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None,
               mode='truth_table', is_chat_model=False, batch_size=256):
    rationales = []
    correct_num = 0
    total_num = 0
    model.eval()

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
        prompts.append(prompt)

    for batch_start in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch_prompts = prompts[batch_start: batch_start + batch_size]
        batch_items = dataset[batch_start: batch_start + batch_size]
        try:
            with torch.no_grad():
                batch_responses = generate_responses_batch(
                    model=model,
                    tokenizer=tokenizer,
                    prompts=batch_prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop,
                    is_chat_model=True
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
            for code_response in batch_responses:            
                #print(rationale_response)
                code_response = code_response.split("<PYTHON>")[-1]
                code_response = code_response.split("</PYTHON")[0]
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

def generate_response(model, tokenizer, user_prompt, max_tokens=50, temperature=1.0, top_p=1.0, top_k=50, stop=None, device=None, is_chat_model=False):
    """
    Generate a response using the model.
    
    Args:
        model: The pre-trained model.
        tokenizer: The tokenizer corresponding to the model.
        user_prompt (str): The input prompt.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling probability.
        top_k (int): Number of top tokens to consider.
        stop (list, optional): Stop sequences.
        device (str, optional): Device to run the model on.
    
    Returns:
        response: The generated response text.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if is_chat_model:
        inputs = tokenizer.apply_chat_template([{ "role": "user", "content": user_prompt }], return_tensors="pt").to(device)
    else:
        inputs = tokenizer(user_prompt, return_tensors="pt").to(device)
    output = model.generate(
        inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id if stop is None else tokenizer.convert_tokens_to_ids(stop)
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def generate_response_batch(model, tokenizer, user_prompts, max_tokens, temperature, top_p, top_k, stop, is_chat_model):
    inputs = tokenizer(user_prompts, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id
    )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses

def generate_rationales(model, tokenizer, dataset, output_dir, output_file, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None, mode='truth_table', eval=False, is_chat_model=False):
    """
    Generate rationales for each data point in the dataset.
    """
    rationales = []
    rationale_prompt = {
        'truth_table': get_prompt_rational_truth_table(),
        'code': get_prompt_rational_code(),
        'nl': get_prompt_rational_nl()
    }.get(mode, "")
    
    for i, item in enumerate(dataset):
        premises = item.get("premises", "")
        conclusions = item.get("conclusion", "")
        label = item.get("label", "")  
        
        prompt = rationale_prompt.format(Premises=premises, Conclusions=conclusions)
        
        try:
            rationale_response = generate_response(
                model=model,
                tokenizer=tokenizer,
                user_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k, stop=stop, is_chat_model=True
            )
            rationale_process = rationale_response.split("<Reasoning>")[-1]
            answer_response = rationale_response.split("<Answer>")[-1]
            predict = 'None'
            if "(A)" in answer_response:
                predict = "True"
            elif "(B)" in answer_response:
                predict = 'False'
            elif "(C)" in answer_response:
                predict = 'Uncertain'

            if predict == label:
                rationales.append({
                    "premises": premises,
                    "conclusions": conclusions,
                    "rationale": rationale_process.strip(),
                    'label': label,
                    'user_prompt': prompt,
                })
                print(f"Generated rationale for data point {i + 1}/{len(dataset)}")
            else:
                print(f"Filter out the data point due to poor quality.")
        except Exception as e:
            print(f"Error generating rationale for data point {i + 1}: {e}")
            continue
    
    with open(os.path.join(output_dir, output_file), 'w') as f:
        json.dump(rationales, f, indent=4)
    print(f"Rationales saved to {os.path.join(output_dir, output_file)}")

################################################# Star Pipeline #############################################################

def star_pipeline_base_reset(model_name_and_path, dataset_name, output_dir, n_samples=200, n_outer_loops=10, n_epochs=4,
                             batch_size=16, learning_rate=1e-5, lora=False, lora_params=None, seed=42, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None, mode='truth_table'):
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
    base_model, tokenizer = load_model_and_tokenizer(model_name_or_path=model_name_and_path)

    outer_loop_responses = []
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    # Obtain seed training set and test set
    dataset, test_dataset = obtain_seed_dataset(dataset_name, n_samples, seed)
    
    # Step -1: Evaluate few-shot perfomrnace with different ideas
    rationale_file = f"rationales_{mode}_{0}.jsonl"
    test_rationale_file = model_name_and_path.split('/')[-1] + f"-{mode}-r{0}-Raw.jsonl"
    test_accuracy_file = model_name_and_path.split('/')[-1] + f"-{mode}-r{0}-Result.jsonl"
    if os.path.exists(os.path.join(output_dir, test_rationale_file)):
            pass
    else:
        pass
        evaluation_batch(
                model=base_model,  # Always use the base model
                tokenizer=tokenizer,
                dataset=test_dataset,
                output_dir=output_dir,
                raw_data_path=test_rationale_file, 
                accuracy_path=test_accuracy_file,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                mode=mode  
        )

    model = base_model
    for n in range(1, n_outer_loops+1):
        print(f"--- Outer Loop {n} ---")
        
        # Step 1: Perform rationale generation
        print("Generating rationales...")
        rationale_file = f"rationales_{mode}_{n}.jsonl"
        test_rationale_file = model_name_and_path.split('/')[-1] + f"-{mode}-r{n}-Raw.jsonl"
        test_accuracy_file = model_name_and_path.split('/')[-1] + f"-{mode}-r{n}-Result.jsonl"
        finetune_response_save_path = f"fine_tuning_{mode}_{batch_size}_{learning_rate}_round_{n}.jsonl"
        if os.path.exists(os.path.join(output_dir, rationale_file)):
            pass
        else:
            generate_rationales(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                output_dir=output_dir,
                output_file=rationale_file,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                mode=mode,
            )
        
        # Step 2: Fine-tune the base model with rationalized datasets
        print("Fine-tuning base model...")
        trainin_data_path = rationale_file#.split('.')[0] + "_train." +  rationale_file.split('.')[1]
        model = finetune(
            client=[base_model, tokenizer],
            dataset_path=os.path.join(output_dir, trainin_data_path),
            output_dir=os.path.join(output_dir, finetune_response_save_path),
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )

        # Step 4: Fine-tune the base model with rationalized datasets


        evaluation_batch(
                model=model,  # Always use the base model
                tokenizer=tokenizer,
                dataset=test_dataset,
                output_dir=output_dir,
                raw_data_path=test_rationale_file, 
                accuracy_path=test_accuracy_file,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=stop,
                mode=mode,
        )
    return outer_loop_responses


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
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature for generation.")
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

    # Run the pipeline
    star_pipeline_base_reset(
        model_name_and_path=args.model_name_and_path,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        n_outer_loops=args.n_outer_loops,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora=args.lora,
        lora_params={
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        seed=args.seed,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        mode=args.mode,
    )

if __name__ == "__main__":
    main()






