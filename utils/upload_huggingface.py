















import os
import shutil
from huggingface_hub import Repository

# 请将下面的 YOUR_HF_TOKEN、YOUR_USERNAME 和 YOUR_DATASET_NAME 替换为你的实际信息
HF_TOKEN = "YOUR_HF_TOKEN"
USERNAME = "TongZheng1999"
DATASET_NAME = "ProofWriter"  # 仓库名称

# 构造仓库 URL（数据集仓库）
repo_url = f"https://huggingface.co/datasets/{USERNAME}/{DATASET_NAME}"
# 本地克隆目录
local_repo_path = f"./{DATASET_NAME}"

# 如果本地目录不存在，则克隆仓库；如果不存在则创建
if not os.path.exists(local_repo_path):
    repo = Repository(local_repo_path, clone_from=repo_url, token=HF_TOKEN)
else:
    repo = Repository(local_repo_path, token=HF_TOKEN)

# 将要上传的文件路径
file_to_upload = "test_modified.json"
destination_path = os.path.join(local_repo_path, "dev_modified.json")

# 将文件复制到仓库目录下
shutil.copy(file_to_upload, destination_path)
print(f"文件已复制到 {destination_path}")

# 添加文件到 git 并提交
repo.git_add(auto_lfs_track=True)
repo.git_commit("Add modified JSON file with premises, conclusion and label")

# 推送到私有仓库
repo.git_push()
print("文件已成功推送到 Hugging Face 私有数据集仓库:", repo_url)







