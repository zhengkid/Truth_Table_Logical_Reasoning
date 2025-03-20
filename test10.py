from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
print(tokenizer.tokenize("<end_of_nl_cot>"))
