


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import json
import argparse

def compute_avg_logprob_chat(user_prompt: str, completion: str, model, tokenizer) -> float:

    chat_prompt = tokenizer.apply_chat_template(
        user_prompt,    
        tokenize=False,
        add_generation_prompt=False
    )

    full_text = chat_prompt + completion

    input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)


    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits


    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:].to(model.device)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    selected_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    avg_logprob = selected_log_probs.mean().item()

    return avg_logprob

def remove_tags_from_list(data):
    tags = ["<truth_table>\n", "<code>\n", "<nl>\n"]
    for entry in data:
        if "content" in entry:
            for tag in tags:
                entry["content"] = entry["content"].replace(tag, "")
    return data

def main():

    parser = argparse.ArgumentParser(description='response select')
    parser.add_argument('--json1', type=str, help='file path 1')
    parser.add_argument('--json2', type=str, help='file path 2')
    parser.add_argument('--json3', type=str, help='file path 3')
    parser.add_argument('--model_name_or_path', type=str, help='model_name_or_path')
    args = parser.parse_args()
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
    model.eval()

    with open(args.json1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    with open(args.json2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    with open(args.json3, 'r', encoding='utf-8') as f:
        data3 = json.load(f)

    correct = 0
    for item1, item2, item3 in zip(data1, data2, data3):




        candidates = {
        "1": "<truth_table>", #item1['rationale'],
        "2": "<code>",
        "3": "<nl>"
        }
        candidates_prompt = {
        "1": item1['user_prompt'],
        "2": item2['user_prompt'],
        "3": item3['user_prompt']
        }
        candidates_item = {
        "1": item1,
        "2": item2,
        "3": item3
        }

        candidates_item_predict = {
        "1": item1['predict'],
        "2": item2['predict'],
        "3": item3['predict']
        }

        results = {}
        for tag, content in candidates.items():
            score = compute_avg_logprob_chat(candidates_prompt[tag], content, model, tokenizer)
            results[tag] = score

        # 按得分排序返回
        results_sorted = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        print(results_sorted)
        max_key = max(results, key=results.get)
        print(max_key)
        if candidates_item[max_key]['predict'] == candidates_item[max_key]['label']:
            correct += 1
        print(item1['label'])
        print(candidates_item_predict)
        print(correct)

    print(correct/len(data1))




main()
