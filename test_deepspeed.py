import torch
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 替换为实际模型路径或 Hugging Face 上的标识符
    model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    
    # 加载 tokenizer，建议关闭 fast 模式以避免某些特殊情况问题
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # 加载模型，使用 fp16 降低显存占用，low_cpu_mem_usage 可减少 CPU 内存压力
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    
    # 配置 DeepSpeed 推理参数







    
    # 使用 deepspeed.init_inference 包装模型（仅用于推理，不涉及梯度计算）
    model = deepspeed.init_inference(model,
                                 tensor_parallel={"tp_size": 1},
                                 dtype=torch.half,
                                 checkpoint=None, #if args.pre_load_checkpoint else args.checkpoint_json,
                                 replace_with_kernel_inject=True)
    #model = deepspeed.init_inference(
    #    model=model,
    #    mp_size=1,                      # 模型并行度，单卡时为1
    #    dtype=torch.float16,            # 指定数据类型
    #    replace_method="auto",          # 自动替换模型中部分模块以提升推理性能
    #    replace_with_kernel_inject=True,# 开启 kernel 注入优化

    #)
    
    # 准备输入文本
    prompt = "请写一首关于春天的诗。"
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # 进行推理生成
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    
    # 解码并输出生成结果
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("生成结果：", generated_text)

if __name__ == "__main__":
    main()

