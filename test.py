import re

def parse_answer(rationale_response, mode, prompt_mode):
    predict = "Unknown"  # 默认值

    if mode != 'code' or (mode == 'code' and 'v3' in prompt_mode):
        # 1. 截取 <Reasoning> 之后的内容
        rationale_response = rationale_response.split("<Reasoning>")[-1]
        rationale_response = rationale_response.split("</Answer>")[0] + "</Answer>"
        
        # 2. 提取 <Answer> 标签内容
        answer_match = re.search(r'<Answer>(.*?)</Answer>', rationale_response, re.DOTALL)
        answer_response = answer_match.group(1).strip() if answer_match else ""

        # 3. 解析答案，支持不同格式
        match = re.search(r'\(?([A-D])\)?', answer_response)
        if match:
            extracted_answer = match.group(1)
            predict_mapping = {
                "A": "True",
                "B": "False",
                "C": "Uncertain",
                "D": "Unknown"
            }
            predict = predict_mapping.get(extracted_answer, "Unknown")

        return rationale_response, predict, None


rationale_response_1 = "<Answer>A</Answer>"
rationale_response_2 = "<Answer>(A) True</Answer>"
rationale_response_3 = "<Answer>The answer is (F)</Answer>"

print(parse_answer(rationale_response_1, 'text', 'v3'))  # ('<Answer>A</Answer>', 'True', None)
print(parse_answer(rationale_response_2, 'text', 'v3'))  # ('<Answer>(A) True</Answer>', 'True', None)
print(parse_answer(rationale_response_3, 'text', 'v3'))  # ('<Answer>The final answer is (A)</Answer>', 'True', None)

