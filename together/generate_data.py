import argparse
import random
import openai
from datasets import load_dataset, Dataset
import torch

# Define a mapping of model names to their types ("chat" or "base")
MODEL_TYPE_MAPPING = {
    "gpt-3.5-turbo": "chat",
    "gpt-4": "chat",
    "NousResearch/Meta-Llama-3.1-8B-Instruct": 'chat',
    "NousResearch/Meta-Llama-3-8B-Instruct": 'chat',
    "NousResearch/Meta-Llama-3.1-8B": 'base',
    "NousResearch/Meta-Llama-3-8B": 'base',
    # Additional chat-based models can be added here
    # Other models default to "base"
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline script for generating truth tables using OpenAI-compatible interfaces")
    parser.add_argument("--model_source", type=str, choices=["openai", "vllm"], required=True,
                        help="Select model source: 'openai' for official OpenAI API or 'vllm' for self-hosted VLLM API")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the OpenAI model to use")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name or path of the dataset to load")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the augmented dataset")
    parser.add_argument("--sample_size", type=int, default=10, help="Number of samples to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--few_shot_prompt_file", type=str, required=True, 
                        help="File path containing the few-shot prompt")
    # Common OpenAI API parameters
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature parameter for generation")
    parser.add_argument("--top_p", type=float, default=None, help="nucleus sampling parameter top_p")
    parser.add_argument("--frequency_penalty", type=float, default=None, help="frequency penalty parameter")
    parser.add_argument("--presence_penalty", type=float, default=None, help="presence penalty parameter")
    parser.add_argument("--stop", type=str, nargs='*', default=None, help="stop sequence(s) for generation")

    # VLLM-specific parameter
    parser.add_argument("--api_base", type=str, default=None, help="Base URL for the self-hosted VLLM OpenAI-compatible API")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_few_shot_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def post_process_truth_table(generated_text):
    """
    Extracts content between <TRUTH TABLE> and </TRUTH TABLE> tags from the generated text.
    If the tags are not found, returns an empty string.
    """
    start_tag = "<TRUTH TABLE>"
    end_tag = "<\TRUTH TABLE>"

    # Find the positions of the start and end tags
    start_index = generated_text.find(start_tag)
    end_index = generated_text.find(end_tag, start_index)

    # If tags are not found, return empty string
    if start_index == -1 or end_index == -1:
        return ""

    # Extract content between tags and strip extra whitespace
    start_index += len(start_tag)
    truth_table_content = generated_text[start_index:end_index].strip()

    return truth_table_content


def generate_truth_table_openai(
    prompt,
    model_name,
    max_tokens,
    temperature,
    top_p=None,
    frequency_penalty=None,
    presence_penalty=None,
    stop=None
):
    # Determine the model type based on the provided model name
    model_type = MODEL_TYPE_MAPPING.get(model_name, "base")
    
    # Common parameters for both chat and base completions
    common_params = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop
    }
    # Add optional parameters if provided
    if top_p is not None:
        common_params["top_p"] = top_p
    if frequency_penalty is not None:
        common_params["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        common_params["presence_penalty"] = presence_penalty

    if model_type == "chat":
        # For chat-based models, construct messages and use ChatCompletion
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            **common_params
        )
        return response.choices[0].message['content'].strip()
    else:
        # For base completion models
        response = openai.Completion.create(
            engine=model_name,
            prompt=prompt,
            **common_params,
            n=1
        )
        return post_process_truth_table(response.choices[0].text.strip())

def process_and_augment_dataset(dataset, sample_size, few_shot_prompt, args):
    sampled_dataset = dataset.shuffle(seed=args.seed).select(range(sample_size))
    augmented_entries = []

    for entry in sampled_dataset:
        input_text = entry['text'] if 'text' in entry else str(entry)
        prompt = few_shot_prompt + "\n<NEW TEXT>\n" + input_text + "\n<\\NEW TEXT>\n<TRUTH TABLE>"
        truth_table = generate_truth_table_openai(
            prompt=prompt,
            model_name=args.model_name,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            frequency_penalty=args.frequency_penalty,  
            presence_penalty=args.presence_penalty,    
            stop=args.stop                    
        )
        entry = dict(entry)
        entry['truth_table'] = truth_table
        augmented_entries.append(entry)

    # Convert the augmented entries into a Hugging Face Dataset object
    columns = augmented_entries[0].keys()
    dataset_dict = {col: [entry[col] for entry in augmented_entries] for col in columns}
    return Dataset.from_dict(dataset_dict)

if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)

    # Set OpenAI API key
    openai.api_key = args.openai_api_key

    # Configure api_base based on the model source
    if args.model_source == "vllm":
        if not args.api_base:
            raise ValueError("When using 'vllm' mode, --api_base parameter specifying the VLLM API URL must be provided")
        openai.api_base = args.api_base
    # No need to set api_base for official OpenAI API

    # Load few-shot prompt text
    few_shot_prompt = load_few_shot_prompt(args.few_shot_prompt_file)

    # Load dataset
    dataset = load_dataset(args.dataset_name)

    # Process and augment dataset (assuming using the train split)
    augmented_dataset = process_and_augment_dataset(
        dataset['train'],
        args.sample_size,
        few_shot_prompt,
        args
    )

    # Save the augmented dataset
    augmented_dataset.save_to_disk(args.output_path)
    print(f"Augmented dataset saved to {args.output_path}")
