import os
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
resp = client.files.upload(file="./results/rationales_1_train.jsonl")

print(resp.model_dump())

resp = client.fine_tuning.create(
        suffix = f"INC-v1",
        model= "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
        training_file=resp.id,
        n_checkpoints=5,
        n_epochs=5,
        batch_size=16,
        learning_rate=1e-5,
    )
print(resp)
