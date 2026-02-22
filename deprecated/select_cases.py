import json

# 输入输出文件
input_file = "new_test_cases_fixed.jsonl"
output_file = "8_10_test_case.jsonl"

# 要筛选的 success_rate
target_rates = { "8/10","9/10","10/10"}

selected = []

with open(input_file, "r") as f:
    for line in f:
        data = json.loads(line)
        if data.get("success_rate") in target_rates:
            selected.append(data)

# 保存筛选结果
with open(output_file, "w") as f:
    for d in selected:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"筛选完成，共得到 {len(selected)} 条数据，保存为 {output_file}")
