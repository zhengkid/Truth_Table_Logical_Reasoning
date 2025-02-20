import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 模型名称：NousResearch/Meta-Llama-3.1-8B-Instruct
    model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    
    # 加载 tokenizer（建议 use_fast=False，以防某些分词问题）
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # 加载模型，使用半精度降低显存占用，并启用低 CPU 内存使用模式
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # 使用 DeepSpeed 推理扩展包装模型
    # 如果你的硬件支持并行，可尝试调整 tensor_parallel 的 tp_size 参数（例如 tp_size=2）
    model = deepspeed.init_inference(
        model=model,
        #tensor_parallel={"tp_size": 1},  # 若模型关键维度满足条件，可设置为 {"tp_size": 2} 等
        #dtype=torch.bfloat16,
        #checkpoint=None,
        replace_with_kernel_inject=True
    )
    
    # 构造推理输入（示例中使用中文提示）
    prompt = "请写一首关于春天的诗："
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 推理生成文本
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    
    # 解码生成文本并输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("生成结果：", generated_text)

if __name__ == "__main__":
    main()

