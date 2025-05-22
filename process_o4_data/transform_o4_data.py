from datasets import load_dataset, DatasetDict
import json
import re

def merge_system_user_and_set_response(ds):
    """
    对每个 example，将原 messages 中的 system 内容拼接在前，
    user 内容拼接在后，作为单条 user 消息；将 response 作为 assistant 消息。
    同时彻底去除示例说明行和所有示例区块，清理多余空行。
    """
    def fn(example):
        msgs = example.get("messages", [])
        # 收集 system 和 user 内容
        system_parts = [m["content"].strip() for m in msgs if m["role"] == "system"]
        user_parts   = [m["content"].strip() for m in msgs if m["role"] == "user"]

        # 合并文本
        combined = "\n".join(system_parts + user_parts)

        # 去除任何示例说明行
        # 包括 “Below are three examples...” 及其变体
        combined = re.sub(r"(?mi)^\s*Below are three examples.*$", "", combined)
        # 去除所有 <EXAMPLE ...> ... </EXAMPLE ...> 区块
        combined = re.sub(r"<EXAMPLE[\s\S]*?</EXAMPLE.*?>", "", combined)
        # 清理多余连续空行
        combined = re.sub(r"\n{2,}", "\n", combined)
        combined = combined.strip()

        # 构造新的 messages
        example["messages"] = [
            {"role": "user",      "content": combined},
            {"role": "assistant", "content": example.get("response", "").strip()}
        ]
        return example

    return ds.map(fn, batched=False)


def main():
    # 从 Hugging Face 加载已经过滤好的数据集
    ds = load_dataset("TongZheng1999/o4-FL-data-filtered", split="train")

    # 转换 messages 结构并清理示例说明与区块
    ds2 = merge_system_user_and_set_response(ds)

    # 推送到 Hugging Face
    DatasetDict({"train": ds2}).push_to_hub("TongZheng1999/o4-FL-data-transformed", private=True)
    print(f"Transformed {len(ds2)} examples → pushed to TongZheng1999/o4-FL-data-transformed")

if __name__ == "__main__":
    main()

