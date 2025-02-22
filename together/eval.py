import os
from together import Together
from datasets import load_dataset
import json
import tqdm

def get_prompt_rational():
    file_path = os.path.join('../prompts', 'prompt_truth_table_star.txt')
    with open(file_path) as f:
        in_context_examples = f.read()
    return in_context_examples

def get_prompt_rational_add_hint():
    file_path = os.path.join('../prompts', 'prompt_truth_table_star_add_hint.txt')
    with open(file_path) as f:
        in_context_examples = f.read()
    return in_context_examples

if __name__ == '__main__':

    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    round = 3
    few_shot = False
    if round == 1:
        model_name = 'simenghan/Meta-Llama-3.1-8B-Instruct-Reference-Star-r1-38710805'
        if few_shot:
            accuracy_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r1-Result.txt'
            raw_data_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r1-Raw.txt'
        else:
            accuracy_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r1-0-shot-Result.txt'
            raw_data_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r1-0-shot-Raw.txt'
    elif round == 2:
        model_name = 'simenghan/Meta-Llama-3.1-8B-Instruct-Reference-Star-r2-c249f72b'
        if few_shot:
            accuracy_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r2-Result.txt'
            raw_data_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r2-Raw.txt'
        else:
            accuracy_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r2-0-shot-Result.txt'
            raw_data_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r2-0-shot-Raw.txt'
    elif round == 3:
        model_name = 'simenghan/Meta-Llama-3.1-8B-Instruct-Reference-Star-r3-c414e58c'
        if few_shot:
            accuracy_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r3-Result.txt'
            raw_data_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r3-Raw.txt'
        else:
            accuracy_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r3-0-shot-Result.txt'
            raw_data_path = 'Meta-Llama-3.1-8B-Instruct-Reference-Star-r3-0-shot-Raw.txt'
    dataset_name = 'yale-nlp/FOLIO'
    output_dir = './result'


    rationales = []
    dataset = load_dataset(dataset_name)['validation']
    rationale_prompt = get_prompt_rational()
    total = 0
    correct = 0
    for i, item in tqdm.tqdm(enumerate(dataset)):
        premises = item.get("premises", "")
        conclusions=item.get("conclusion", "")
        label = item.get("label", "")  
        
        # Construct the prompt for this data point
        if few_shot:
            prompt = rationale_prompt.format(Premises=premises, Conclusions=conclusions)
        else:
            prompt = "<Premises>\n{Premises}\n</Premises>\n<Question>\nIs the following statement true, false, or uncertain? {Conclusions}\n</Question>\n<Options>(A) True\n(B) False\n(C) Uncertain\n</Options>\n<Answer>".format(Premises=premises, Conclusions=conclusions)
        print("prompt", prompt)

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=2048,
            temperature=0.7,
        )

        rationale_response= response.choices[0].message.content

        print(rationale_response)
        rationale_response = rationale_response.split("<Answer>")[-1]
        if "(A)" in rationale_response:
            predict = "True"
        elif "(B)" in rationale_response:
            predict = 'False'
        elif "(C)" in rationale_response:
            predict = 'Uncertain'
        else:
            predict = 'Unknown'

        # Add the generated rationale to the output list
        rationales.append({
            "premises": premises,
            "conclusions": conclusions,
            "rationale": rationale_response.strip(),
            'predict': predict,
            'label': label
        })
        if predict == label:
            total += 1
            correct += 1
            print(correct)
        else:
            total += 1
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0

    # Save rationales to a file
    with open(raw_data_path, 'w') as f:
        json.dump(rationales, f, indent=4)

    # Save accuracy to a text file
    with open(accuracy_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Correct predictions: {correct}\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Rationales saved to {raw_data_path}")
    print(f"Accuracy report saved to {accuracy_path}")