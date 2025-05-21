import ast
import traceback

def is_syntactically_valid(code_str):
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False

def is_executable(code_str):
    try:
        exec(code_str, {})
        return True
    except Exception:
        return False

def try_run(code_str):
    try:
        loc = {}
        exec(code_str, {}, loc)
        return loc
    except Exception as e:
        return str(e)

# 假设你已经有了模型预测正确的 code-mode 输出
sample_outputs = [
    "def solve():\n    return False",  # sample 1
    "if True and False:\n    return True",  # sample 2
    "return True",  # sample 3
]

for idx, code in enumerate(sample_outputs):
    print(f"\nSample #{idx+1}")
    print("Code:", code)
    valid = is_syntactically_valid(code)
    exec_ok = is_executable(code)
    result = try_run(code)
    print(f"  Syntax Valid: {valid}")
    print(f"  Executable: {exec_ok}")
    print(f"  Run Output: {result}")

