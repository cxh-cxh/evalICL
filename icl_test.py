import openai
import requests
import base64
from PIL import Image
from io import BytesIO
import json
from time import sleep

import os
# os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
# mine
os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
# zmy
import pathlib

from openai import OpenAI

client = OpenAI(
    # # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="OPENAI_API_KEY",
    base_url="https://api.chatanywhere.tech/v1"
    # # base_url="https://api.chatanywhere.org/v1"
)

def b64_of(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_jsonl(path="./test_cases.jsonl", limit=10):
    data = []
    p = pathlib.Path(path)
    if not p.exists():
        return data
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                data.append(ex)
                if len(data) >= limit:
                    break
            except Exception:
                continue
    return data

prompt = f"""
The robot will be performing the task of putting the pink cube on the blue cube according to a policy trained on provided dataset.

The robot will be using two cameras to observe the environment and make decisions based on the visual input. The whole environment will follow a certain protocol to ensure the overall consistency of the settings.

Under this protocol, the basic elements of the environment will be the same. However, the lighting, the pose of the cameras may vary on a small scale due to unavoidable errors.

Now I want you to consider the images of the initial setting of any rollout and analyze its difficulty compared to the training dataset.

Although I will not be abled to provide you with the complete training dataset, I can offer you some relative difference between the test case and the training data in the <rel> and </rel> tags. Below is an example:

<rel>

- L2 distance of the test case with the nearest training data(blue cube and pink cube distance added): 2.0cm

- L2 distance of the blue cube in the test case with the nearest blue cube in the training data: 1.0cm

- L2 distance of the pink cube in the test case with the nearest pink cube in the training data: 0.5cm

</rel>

You will analyze the difficulty of the task based on the image of the initial setting provided. The difficulty should be assessed based on how well the robot can perform the task given the visual input from the camera.

Below are some examples of test cases and their difficulty value where a difficulty value of x means the model achieved a success rate of (100-10x)% on this test case. Please learn from these examples to evaluate the difficulty of the test case you will be given.
"""

examples = load_jsonl("./test_cases.jsonl", limit=20)
examples_content = []
desc_front_img = "This image below shows the front camera view in this test case."
desc_side_img = "This image below shows the side camera view in this test case."
for ex in examples:
    # f"tx={ex.get('rel_tx_cm','')}cm, ty={ex.get('rel_ty_cm','')}cm, "
    # f"yaw={ex.get('rel_yaw_rad','')}rad\n"
    ex_desc = f"""[EXAMPLE]
        
<rel>

- L2 distance of the test case with the nearest training data: {ex.get('l2_cm', '')}cm

- L2 distance of the blue cube in the test case with the nearest blue cube in the training data: {ex.get('l2_big_cm', '')}cm

- L2 distance of the pink cube in the test case with the nearest pink cube in the training data: {ex.get('l2_small_cm', '')}cm

</rel>

- Test case difficulty: {10 - int(ex.get('success_rate','').split('/')[0])}
"""
    examples_content.extend([
        {"type": "text", "text": ex_desc},
        {"type": "text", "text": desc_front_img},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(ex['front_img'])}"}},
        {"type": "text", "text": desc_side_img},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(ex['side_img'])}"}},
    ])

# 组装完整 content：先放全局任务说明，再放 few-shot 示例，再放当前待评估样本
full_content = []
full_content.append({"type": "text", "text": prompt})
full_content.extend(examples_content)

queries = load_jsonl("./queries.jsonl", limit=10)
difficulties = []

for i, q in enumerate(queries):
    copy_full_content = full_content.copy()
    query_content = []
    query_desc = f"""[QUERY]
    
Now I want you to evaluate the difficulty of the following test case according to the examples.

<rel>

- L2 distance of the test case with the nearest training data: {q.get('l2_cm', '')}cm

- L2 distance of the blue cube in the test case with the nearest blue cube in the training data: {q.get('l2_big_cm', '')}cm

- L2 distance of the pink cube in the test case with the nearest pink cube in the training data: {q.get('l2_small_cm', '')}cm

</rel>

First analyze the task in the <think> and </think> tags below:

<think>

[Write your detailed analysis here]
</think>

Then list out all the factors that may influence the outcome of the process and classify them into certain levels of difficulty.

Finally provide a difficulty value from 0 to 10 between <difficulty> and </difficulty> tags.

<difficulty>

[Provide a single integer of 0 to 10]

</difficulty>
"""
    
# - 10 means the test case is very hard and the model is expected to achieve a success rate of 0% on it
# - 8-9 means the test case is rather hard and the model is expected to achieve a success rate of 10%-20% on it
# - 5-7 means the test case is moderate in difficulty and the model is expected to achieve a success rate of 30%-50% on it
# - 2-4 means the test case is rather easy and the model is expected to achieve a success rate of 60%-80% on it
# - 1 means the test case is very easy and the model is expected to achieve a success rate of 90%-100% on it

    query_content.extend([
        {"type": "text", "text": query_desc},
        {"type": "text", "text": desc_front_img},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(q['front_img'])}"}},
        {"type": "text", "text": desc_side_img},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(q['side_img'])}"}},
    ])
    copy_full_content.extend(query_content)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": copy_full_content}],
        )
    except Exception as e:
        print(f"Error processing query {i}: {e}")
        print("Dumping all previous queries and exit")
        break

    # print(response.choices[0].message.content)

    save_path = f"./results/response_{i}.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(response.to_dict(), f, indent=4, ensure_ascii=False)

    import re
    difficulty_match = re.search(r"<difficulty>(.*?)</difficulty>", response.choices[0].message.content, re.DOTALL)
    if difficulty_match:
        difficulty = difficulty_match.group(1).strip()
    else:
        difficulty = "No difficulty provided"
    print(f"Difficulty for query {i}: {difficulty}")
    difficulties = difficulties + [difficulty]

    # sleep(70)  # Sleep for a second to avoid hitting rate limits

with open(f"./results/difficulties_new.json", "w", encoding="utf-8") as f:
    json.dump(difficulties, f, indent=4, ensure_ascii=False)


