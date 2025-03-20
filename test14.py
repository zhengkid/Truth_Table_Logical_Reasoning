from vllm import LLM, SamplingParams

# 初始化模型（可替换为你自己的模型路径）
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 设置生成参数
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=1024
)

# 带system角色的prompt
messages_with_system = [
    {"role": "system", "content": "You are a rigorous and logically precise AI assistant. Your task is to answer a logical reasoning problem strictly following one of three modes, as explicitly specified in the input. Only one mode will be present in the input. Follow that mode exclusively.\n\n- Code Mode (<code> ... </code> <answer> ... </answer>)  \n  - If the input contains <code>, translate the problem into Python code.  \n  - Execute the logic and derive the answer.  \n\n- Natural Language Chain-of-Thought Mode (<nl_cot> ... </nl_cot> <answer> ... </answer>)  \n  - If the input contains <nl_cot>, solve the problem step by step in natural language.  \n\n- Truth Table Mode (<truth_table> ... </truth_table> <answer> ... </answer>)  \n  - If the input contains <truth_table>, construct a truth table and derive the answer from it.  \n\n### Rules  \n- Only use the mode specified in the input. Do not switch modes.  \n- Generate output strictly in the specified mode and format, with no additional text.  \n- Enclose all reasoning strictly within the corresponding mode tags.  \n- The final answer must be strictly enclosed in <answer> ... </answer>.  \n- If no valid mode is detected, return: <error> Mode not specified. </error> Do not attempt to infer one.  \n- Do not provide any reasoning or explanations outside of the designated mode tags."},

    {"role": "user", "content": "<EXAMPLE 1>\n<premises>\nPeter Parker is either a superhero or a civilian.\nThe Hulk is a destroyer.\nThe Hulk wakes up when he is angry.\nIf the Hulk wakes up, then he will break a bridge.\nThor is a god.\nThor will break a bridge when he is happy.\nA god is not a destroyer.\nPeter Parker wears a uniform when he is a superhero.\nPeter Parker is not a civilian if a destroyer is breaking a bridge.\nIf Thor is happy, the Hulk is angry.\n</premises>\n<conclusion>\nIf Thor is happy, then Peter Parker wears a uniform.\n</conclusion>\n<question>\nIs the following statement true, false, or uncertain? If Thor is happy, then Peter Parker wears a uniform.\n</question> \n<options>\n(A) True\n(B) False\n(C) Uncertain\n</options>\n<nl_cot>\n... (省略示例推理步骤)\n</nl_cot>\n<answer>\nThe final answer is (A).\n</answer>\n</EXAMPLE 1>\n\n<premises>\nEvents are either happy or sad.\nAt least one event is happy.\n</premises>\n<conclusion>\nAll events are sad.\n</conclusion>\n<question>\nIs the following statement true, false, or uncertain? All events are sad.\n</question>\n<options>\n(A) True\n(B) False\n(C) Uncertain\n</options>\n<nl_cot>"}
]

# 不带system角色的prompt
messages_without_system = [
    {"role": "user", "content": "You are a rigorous and logically precise AI assistant. Your task is to answer a logical reasoning problem strictly following one of three modes... (完整提示与示例省略重复内容)\n\n<premises>\nEvents are either happy or sad.\nAt least one event is happy. \n</premises>\n<conclusion>\nAll events are sad.\n</conclusion>\n<question>\nIs the following statement true, false, or uncertain? All events are sad.\n</question>\n<options>\n(A) True\n(B) False\n(C) Uncertain\n</options>\n<nl_cot>"}
]

# 推理并输出结果
output_with_system = llm.chat(messages_with_system, sampling_params=sampling_params)
output_without_system = llm.chat(messages_without_system, sampling_params=sampling_params)

print("✅ 带system角色的输出:\n", output_with_system)
print("\n⚠️ 不带system角色的输出:\n", output_without_system)
