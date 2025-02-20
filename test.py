import os
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))








response = client.fine_tuning.retrieve('ft-d709d69f-31ed-4165-9605-6200ec35927a')
print(response)
print(response.status._value_) # STATUS_UPLOADING





