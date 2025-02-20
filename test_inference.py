import os
from together import Together


client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

user_prompt = "What is the capital of France?"

response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=[
        {
            "role": "user",
            "content": user_prompt,
        }
    ],
    max_tokens=512,
    temperature=0.7,
)

print(response.choices[0].message.content)
