import os, json, re, asyncio, functools, pathlib, base64, random
from datetime import datetime

difficulty = pathlib.Path("results/difficulties.json")
result = pathlib.Path("results/queries.jsonl")

with result.open("r") as fr:
    res = []
    for f in fr:
        res.append(json.loads(f))

with difficulty.open("r", encoding="utf-8") as fd:
    data = json.load(fd)

succ = 0
fail = 0
for i in range(len(data)):
    if data[i] == "easy":
        if res[i]["first_success"] == "0":
            fail += 1
    elif data[i] == "medium":
        if res[i]["first_success"] == "0":
            fail += 1
        else:
            succ += 1
    elif data[i] == "hard":
        if res[i]["first_success"] == "1":
            succ += 1
print(succ, fail)
