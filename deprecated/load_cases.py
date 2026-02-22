import numpy as np
import csv
import seaborn as sns
import json

# 读取 new_cases.txt，按顺序保存坐标
cases_list = []
with open("new_cases.txt", "r") as f:
    next(f)  # 跳过表头
    for line in f:
        parts = line.strip().split()
        if len(parts) == 5:
            _, sx, sy, bx, by = parts
            cases_list.append(
                {
                    "small_pos": [str(float(sx)), str(float(sy))],
                    "big_pos": [str(float(bx)), str(float(by))],
                }
            )

# 更新 new_test_cases.jsonl
updated = []
with open("new_test_cases.jsonl", "r") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        if idx < len(cases_list):
            data["small_pos"] = cases_list[idx]["small_pos"]
            data["big_pos"] = cases_list[idx]["big_pos"]
        updated.append(data)

# 保存为新文件
with open("new_test_cases_fixed.jsonl", "w") as f:
    for d in updated:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print("已生成 new_test_cases_fixed.jsonl")
