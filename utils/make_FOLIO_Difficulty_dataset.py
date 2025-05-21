from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import login
import os




# 加载 validation 数据
ds = load_dataset("yale-nlp/FOLIO", split="validation")

# 根据 story_id 分割
easy_data = ds.filter(lambda x: int(x["story_id"]) <= 303)
difficult_data = ds.filter(lambda x: int(x["story_id"]) >= 304)

# 保存为 DatasetDict 并上传到 hub
easy = DatasetDict({"validation": easy_data})
difficult = DatasetDict({"validation": difficult_data})

easy.push_to_hub("TongZheng1999/FOLIO-Easy")
difficult.push_to_hub("TongZheng1999/FOLIO-Difficult")

