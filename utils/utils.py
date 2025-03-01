#!/usr/bin/env python3
import sys
import re

if len(sys.argv) != 8:
    print("Usage: {} <config_file> <output_dir> <hub_model_id> <model_name_or_path> <tokenizer_name_or_path> <dataset_pattern> <dataset_replacement>".format(sys.argv[0]))
    sys.exit(1)

config_file = sys.argv[1]
output_dir = sys.argv[2]
hub_model_id = sys.argv[3]
model_name_or_path = sys.argv[4]
tokenizer_name_or_path = sys.argv[5]
dataset_pattern = sys.argv[6]
dataset_replacement = sys.argv[7]

substitutions = {
    r"^output_dir:.*": f"output_dir: {output_dir}",
    r"^hub_model_id:.*": f"hub_model_id: {hub_model_id}",
    r"^model_name_or_path:.*": f"model_name_or_path: {model_name_or_path}",
    r"^tokenizer_name_or_path:.*": f"tokenizer_name_or_path: {tokenizer_name_or_path}",
    dataset_pattern: dataset_replacement
}

with open(config_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    new_line = line
    for pattern, replacement in substitutions.items():
        new_line = re.sub(pattern, replacement, new_line)
    new_lines.append(new_line)

with open(config_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

