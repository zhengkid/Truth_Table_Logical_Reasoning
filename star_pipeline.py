from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import torch
import random
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and STaR process for truth table generation")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Hugging Face model to use")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_iterations", type=int, default=5, help="Number of STaR iterations")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for fine-tuning")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs for fine-tuning")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def formate_prompt(premise, conclusion, truth_table, answer):
    """
    Formats the input data into a structured prompt-response for SFT with task description 
    and well-defined answer format.
    
    Args:
        premise (str): The premise or logical conditions.
        conclusion (str): The conclusion derived from the premise.
        truth_table (str): The truth table explaining the logic.
        answer (str): The final answer derived from the truth table.

    Returns:
        dict: A dictionary with "prompt" and "response" keys, suitable for SFT.
    """
    # Task description to provide additional clarity
    task_description = (
        "Task Description:\n"
        "You are given a logical premise and a desired conclusion. "
        "Your task is to:\n"
        "1. Analyze the logical premise.\n"
        "2. Generate the corresponding truth table based on the logical relationships.\n"
        "3. Derive the final answer based on the truth table.\n"
        "The final answer must be one of the following:\n"
        "- True: The conclusion is always satisfied.\n"
        "- False: The conclusion is never satisfied.\n"
        "- Uncertain: The conclusion is satisfied under certain conditions.\n"
    )

    # Construct the prompt
    prompt = (
        f"{task_description}\n"
        f"Premise:\n{premise}\n\n"
        f"Conclusion:\n{conclusion}\n\n"
        f"Question: Generate the truth table and the final answer based on the above information."
    )

    # Construct the response to include both truth table and formatted answer
    response = (
        f"Truth Table:\n{truth_table}\n\n"
        f"Answer: {answer}"  # Ensure `answer` is one of "True", "False", or "Uncertain"
    )

    return {"prompt": prompt, "response": response}


def load_model_and_tokenizer(model_name):
    """
    Load a Hugging Face model and tokenizer for fine-tuning.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # Set Pad Token
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def parse_generated_output(output_text):
    """
    Parse the generated output to extract the truth table and the final answer.
    """
    if "<TRUTH TABLE>" in output_text and "Answer:" in output_text:
        truth_table = output_text.split("<TRUTH TABLE>")[1].split("</TRUTH TABLE>")[0].strip()
        answer = output_text.split("Answer:")[1].strip()
        return truth_table, answer
    return None, None


def evaluate_answer(generated_answer, ground_truth_answer):
    """
    Evaluate whether the generated answer matches the ground truth answer.
    """
    return generated_answer.strip().lower() == ground_truth_answer.strip().lower()


def generate_hint(question, ground_truth_answer):
    """
    Generate a hint for the model to guide its reasoning.
    """
    return f"Hint: Ensure the truth table matches the logic of the question, and the answer is '{ground_truth_answer}'."


def fine_tune_model(model, tokenizer, train_dataset, output_dir, learning_rate, batch_size, num_epochs):
    """
    Fine-tune a Hugging Face model using the Trainer API.
    """
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    return model


def star_process(model, tokenizer, dataset, num_iterations, output_dir, learning_rate, batch_size, num_epochs):
    """
    Executes the STaR process: iterative improvement and fine-tuning.
    """
    for iteration in range(num_iterations):
        print(f"STaR Iteration {iteration + 1}/{num_iterations}")

        augmented_data = []  # Store augmented data for the current iteration

        # Step 1: Generate truth table and answers for each example in the dataset
        for example in dataset:
            question = example["question"]
            ground_truth_answer = example["answer"]

            # Generate truth table and answer using the model
            input_ids = tokenizer.encode(question, return_tensors="pt")
            outputs = model.generate(input_ids, max_length=512, temperature=0.7)  # Adjust generation params if needed
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse the generated output
            truth_table, generated_answer = parse_generated_output(generated_text)

            # Step 2: Evaluate if the generated answer is correct
            is_correct = evaluate_answer(generated_answer, ground_truth_answer)

            if is_correct:
                # If correct, add to the augmented dataset
                augmented_data.append({"question": question, "truth_table": truth_table, "answer": ground_truth_answer})
            else:
                # Step 3: Rationalize with hints and regenerate
                hint = generate_hint(question, ground_truth_answer)
                hint_input = tokenizer.encode(hint, return_tensors="pt")
                hint_outputs = model.generate(hint_input, max_length=512, temperature=0.7)
                rationalized_text = tokenizer.decode(hint_outputs[0], skip_special_tokens=True)

                # Parse the rationalized output
                rationalized_truth_table, rationalized_answer = parse_generated_output(rationalized_text)

                # Check if the rationalized answer is correct
                is_rationalized_correct = evaluate_answer(rationalized_answer, ground_truth_answer)

                if is_rationalized_correct:
                    augmented_data.append({
                        "question": question,
                        "truth_table": rationalized_truth_table,
                        "answer": ground_truth_answer
                    })

        # Step 4: Convert augmented data into a Hugging Face Dataset
        train_dataset = Dataset.from_dict({
            "input": [d["question"] for d in augmented_data],
            "output": [d["truth_table"] for d in augmented_data]
        })

        # Step 5: Fine-tune the model on the augmented dataset
        model = fine_tune_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            output_dir=f"{output_dir}/iteration_{iteration + 1}",
            learning_rate=learning_rate,
            batch_size=batch_size,
            num_epochs=num_epochs
        )

    return model


def main():
    args = parse_arguments()
    set_seed(args.seed)

    # Load dataset
    dataset = load_dataset(args.dataset_name)["train"]

    # Load Hugging Face model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model_name)

    # Step 1 Warmup - Finetune the model with generated truth table and answers

    model = fine_tune_model()

    # Step 2 Perform STaR process
    model = star_process(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        num_iterations=args.num_iterations,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
    )

    # Save the fine-tuned model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Fine-tuned model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
