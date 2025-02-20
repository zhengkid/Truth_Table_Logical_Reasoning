import os
from together import Together
from datasets import load_dataset
import json
import argparse
import time
import tqdm
import re

##########################################################Begin: Formating Prompts##########################################################################
# Prompting Truth Table 
def get_sys_prompt_rational_truth_table():
    file_path = os.path.join('./prompts', 'sys_prompt_truth_table_star.txt')
    with open(file_path) as f:
        sys_prompt = f.read()
    return sys_prompt

def get_few_shot_prompt_rational_truth_table():
    file_path = os.path.join('./prompts', 'prompt_truth_table_star.txt')
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
    file_path = os.path.join('./prompts', 'sys_prompt_code_star.txt')
    with open(file_path) as f:
        sys_prompt = f.read()
    return sys_prompt

def get_few_shot_prompt_rational_code():
    file_path = os.path.join('./prompts', 'prompt_code_star.txt')
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
    file_path = os.path.join('./prompts', 'sys_prompt_nl_star.txt')
    with open(file_path) as f:
        sys_prompt = f.read()
    return sys_prompt

def get_few_shot_prompt_rational_nl():
    file_path = os.path.join('./prompts', 'prompt_nl_star.txt')
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
def finetune(client, file_resp, output_dir, model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
             validation_file=None, suffix="custom-ft", n_epochs=4, n_evals=0, n_checkpoints=1,
             batch_size=16, learning_rate=1e-5, min_lr_ratio=0.0, warmup_ratio=0.0, 
             lora=False, lora_r=8, lora_alpha=8, lora_dropout=0.0, lora_trainable_modules="all-linear"):
    """
    Function to upload a dataset file and trigger a fine-tuning job with optional parameters.

    Args:
        client (Together): An initialized Together API client instance.
        file_path (str): Path to the dataset file (e.g., JSONL file).
        output_dir (str): Directory to save the fine-tuning response.
        model (str): Base model for fine-tuning. Default is Meta-Llama.
        validation_file (str): Path to validation dataset file (optional).
        suffix (str): Suffix for the fine-tuned model name.
        n_epochs (int): Number of training epochs. Default is 4.
        n_evals (int): Number of evaluations on validation set. Default is 0.
        n_checkpoints (int): Number of checkpoints to save during training. Default is 1.
        batch_size (int): Batch size for training. Default is 16.
        learning_rate (float): Learning rate for training. Default is 1e-5.
        min_lr_ratio (float): Ratio of final LR to peak LR. Default is 0.0.
        warmup_ratio (float): Warmup percentage of total training steps. Default is 0.0.
        lora (bool): Whether to enable LoRA training. Default is False.
        lora_r (int): Rank for LoRA adapter weights. Default is 8.
        lora_alpha (int): Alpha value for LoRA training. Default is 8.
        lora_dropout (float): Dropout probability for LoRA layers. Default is 0.0.
        lora_trainable_modules (str): LoRA trainable modules. Default is "all-linear".

    Returns:
        dict: Response from Together API.
    """
    
    # Trigger the fine-tuning job
    try:
        # response = client.fine_tuning.create(
        #     suffix = suffix,
        #     model= model,
        #     training_file=file_resp.id,
        #     n_checkpoints=1,
        #     n_epochs=1,
        #     batch_size=16,
        #     learning_rate=1e-5,
        #     # wandb_api_key=os.environ.get("WANDB_API_KEY"),
        # )

        response = client.fine_tuning.create(
            training_file=file_resp.id,
            # validation_file=validation_file_id,
            model=model,
            suffix=suffix,
            n_epochs=n_epochs,
            # n_evals=n_evals,
            # n_checkpoints=n_checkpoints,
            batch_size=batch_size,
            learning_rate=learning_rate,
            # min_lr_ratio=min_lr_ratio,
            # warmup_ratio=warmup_ratio,
            lora=lora,
            # lora_r=lora_r,
            # lora_alpha=lora_alpha,
            # lora_dropout=lora_dropout,
            # lora_trainable_modules=lora_trainable_modules,
            wandb_api_key=os.environ.get("WANDB_API_KEY"),
        )
        print(f"Fine-tuning job {response.id} created successfully!")
        print(response)
        # Block until the fine-tuning job is finished
        ft_id = response.id
        ft_status = client.fine_tuning.retrieve(ft_id)
        while not ft_status.status._value_ == "completed":
            time.sleep(10)  # Poll every 10 seconds
            ft_status = client.fine_tuning.retrieve(ft_id)
        print(f"Fine-tuning job {response.id} completed!")
        
        response_file_path = output_dir
        with open(response_file_path, 'w') as f:
            data_to_save = {
                "job_id": response.id,
                "model_name": response.output_name
            }
            json.dump(data_to_save, f, indent=4)
        
        print(f"Fine-tuning response saved to {response_file_path}")
        return response
    except Exception as e:
        print("Error creating fine-tuning job:", e)
        return {"error": str(e)}

##########################################################Code for Evaluation##########################################################################



def evaluation(client, model, dataset, output_dir, raw_data_path, accuracy_path, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None, mode='truth_table'):
    # Prepare a list to store rationales
    rationales = []
    correct_num = 0
    total_num = 0
    if mode == 'truth_table':
        rationale_prompt = get_prompt_rational_truth_table()
    elif mode == 'code':
        rationale_prompt = get_prompt_rational_code()
    elif mode == 'nl':
        rationale_prompt = get_prompt_rational_nl()

    # print(rationale_prompt)
    # Generate rationale for each data point
    for i, item in tqdm.tqdm(enumerate(dataset)):
        premises = item.get("premises", "")
        conclusions=item.get("conclusion", "")
        label = item.get("label", "")  
        
        # Construct the prompt for this data point
        # print(rationale_prompt)
        prompt = rationale_prompt.format(Premises=premises, Conclusions=conclusions)
        # print("prompt", prompt)
        try:
            # Generate rationale using the Together API
            rationale_response = generate_response(
                client=client,
                model=model,
                user_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k, stop=stop
            )
            rationale_response = rationale_response.split("<Answer>")[-1]
            
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
                # Add the generated rationale to the output list
                correct_num += 1
                total_num += 1
            else:
                total_num += 1

            print(f"{correct_num} out of {total_num} is correct!")
        except Exception as e:
            print(f"Error generating rationale for data point {i + 1}: {e}")
            continue


    accuracy = correct_num / total_num if total_num > 0 else 0.0

    # Save rationales to a file
    with open(os.path.join(output_dir, raw_data_path), 'w') as f:
        json.dump(rationales, f, indent=4)
    print(f"Rationales saved to {os.path.join(output_dir, raw_data_path)}")

    # Save accuracy to a text file
    with open(os.path.join(output_dir, accuracy_path), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Total samples: {total_num}\n")
        f.write(f"Correct predictions: {correct_num}\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total samples: {total_num}")
    print(f"Correct predictions: {correct_num}")
    print(f"Rationales saved to {raw_data_path}")
    print(f"Accuracy report saved to {accuracy_path}")


##########################################################Code for Generating Response##########################################################################
def generate_response(client, model, user_prompt, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None):
    """
    Function to generate a response from a model using Together API with advanced parameters including stop tokens.

    Args:
        client (Together): An initialized Together API client instance.
        model (str): The model to use for generating the response.
        user_prompt (str): The user prompt to provide as input.
        max_tokens (int): Maximum number of tokens in the generated response. Default is 512.
        temperature (float): Sampling temperature. Default is 0.7.
        top_p (float): Top-p (nucleus) sampling parameter. Default is 0.9.
        top_k (int): Top-k sampling parameter. Default is 50.
        stop (list): A list of stop tokens to indicate where generation should stop. Default is None.

    Returns:
        str: The generated response content from the model.
    """
    try:
        # Call Together's chat completion API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,  # Pass stop tokens to the API
        )
        # Extract the generated response content
        # print(response)
        generated_content = response.choices[0].message.content
        print("Generated response:", generated_content)
        return generated_content
    except Exception as e:
        print(f"Error generating response: {e}")
        return ""

def generate_rationales(client, base_model, dataset, output_dir, output_file, max_tokens=512, temperature=0.7, top_p=0.9, top_k=50, stop=None, mode='truth_table', eval=False):
    """
    Generate rationales for each data point in the dataset.

    Args:
        client (Together): An initialized Together API client instance.
        base_model (str): Pre-trained base model.
        dataset (str): Path to the dataset (e.g., JSONL file).
        output_file (str): Path to save generated rationales.
        max_tokens (int): Maximum tokens for each generated rationale.

    Returns:
        None
    """

    # Prepare a list to store rationales
    rationales = []

    if mode == 'truth_table':
        rationale_prompt = get_prompt_rational_truth_table()
    elif mode == 'code':
        rationale_prompt = get_prompt_rational_code()
    elif mode == 'nl':
        rationale_prompt = get_prompt_rational_nl()
    # rationale_add_hint_prompt = get_prompt_rational_add_hint()
    # Generate rationale for each data point
    for i, item in enumerate(dataset):
        premises = item.get("premises", "")
        conclusions=item.get("conclusion", "")
        label = item.get("label", "")  
        
        # Construct the prompt for this data point
        prompt = rationale_prompt.format(Premises=premises, Conclusions=conclusions)
        # print("prompt", prompt)
        try:
            # Generate rationale using the Together API
            rationale_response = generate_response(
                client=client,
                model=base_model,
                user_prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k, stop=stop
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
                # Add the generated rationale to the output list
                rationales.append({
                    "premises": premises,
                    "conclusions": conclusions,
                    "rationale": rationale_process.strip(),
                    'label': label,
                    'user_prompt': prompt,
                })
                print(f"Generated rationale for data point {i + 1}/{len(dataset)}")
            else:
                print(f"Filter out the data point as the poor quality.")
        except Exception as e:
            print(f"Error generating rationale for data point {i + 1}: {e}")
            continue

    # Save the rationales to the output file
    
    with open(os.path.join(output_dir, output_file), 'w') as f:
        json.dump(rationales, f, indent=4)
    print(f"Rationales saved to {os.path.join(output_dir, output_file)}")

    if not eval:
        # Convert the data format
        converted_data = convert_to_custom_format(rationales)
        
        # Save the converted data as a JSON file
        output_file = output_file.split('.')[0] + '_train' + "." + output_file.split('.')[1]
        with open(os.path.join(output_dir, output_file), 'w', encoding='utf-8') as f:
            for item in converted_data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')  # Add a newline after each JSON object
        print(f"Data successfully converted and saved to {os.path.join(output_dir, output_file)}")


def star_pipeline_base_reset(client, base_model, dataset_name, output_dir, n_samples=200, n_outer_loops=10, n_epochs=4,
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
    outer_loop_responses = []
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    
    dataset, test_dataset = obtain_seed_dataset(dataset_name, n_samples, seed)
    
    # Step -1: Evaluate few-shot perfomrnace with different ideas

    rationale_file = f"rationales_{mode}_{0}.jsonl"
    test_rationale_file = base_model.split('/')[-1] + f"-{mode}-r{0}-Raw.jsonl"
    test_accuracy_file = base_model.split('/')[-1] + f"-{mode}-r{0}-Result.jsonl"
    if os.path.exists(os.path.join(output_dir, test_rationale_file)):
            pass
    else:
        evaluation(
                client=client,
                model=base_model,  # Always use the base model
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
    # Step 0: Obtain Seed Dataset 
    model = base_model
    for n in range(1, n_outer_loops+1):
        print(f"--- Outer Loop {n} ---")
        
        # Step 1: Perform rationale generation
        print("Generating rationales...")
        rationale_file = f"rationales_{mode}_{n}.jsonl"
        test_rationale_file = base_model.split('/')[-1] + f"-{mode}-r{n}-Raw.jsonl"
        test_accuracy_file = base_model.split('/')[-1] + f"-{mode}-r{n}-Result.jsonl"
        finetune_response_save_path = f"fine_tuning_{mode}_{batch_size}_{learning_rate}_round_{n}.jsonl"
        if os.path.exists(os.path.join(output_dir, rationale_file)):
            pass
        else:
            generate_rationales(
                client=client,
                base_model=model,  # Always use the base model
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

        # Step 2: Prepare Data for Together AI training
        trainin_data_path = rationale_file.split('.')[0] + "_train." +  rationale_file.split('.')[1]
        print(os.path.join(output_dir, trainin_data_path))
        file_resp = client.files.upload(file=os.path.join(output_dir, trainin_data_path), check=True)
        print(file_resp.model_dump())

        # Step 3: Fine-tune the base model with rationalized datasets
        print("Fine-tuning base model...")
        if os.path.exists(os.path.join(output_dir, finetune_response_save_path)):
            with open(os.path.join(output_dir, finetune_response_save_path), 'r') as file:
                fine_tune_response = json.loads(file.read().strip())
                model = fine_tune_response['model_name']
                print(model)
        else:
            lora_params = lora_params or {}
            fine_tune_response = finetune(
                client=client,
                file_resp=file_resp,
                output_dir=os.path.join(output_dir, finetune_response_save_path),
                model="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",  # Reset to base model every time
                n_epochs=n_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                lora=lora,
                **lora_params
            )
            outer_loop_responses.append(fine_tune_response)
            model = fine_tune_response.output_name

        # Step 4: Fine-tune the base model with rationalized datasets
        # To do 
        evaluation(
            client=client,
            model=model,  # Always use the base model
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
    return outer_loop_responses


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run the STaR pipeline with fine-tuning.")

    # Add arguments
    parser.add_argument("--base_model", type=str, required=True, 
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
    args = parser.parse_args()

    # Print arguments for verification
    print("Running with the following arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Initialize Together client 
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    # Run the pipeline
    star_pipeline_base_reset(
        client=client,
        base_model=args.base_model,
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






