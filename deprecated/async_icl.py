# python
import os, json, re, asyncio, functools, pathlib, base64
from openai import AsyncOpenAI
from datetime import datetime

API_KEY = os.environ["OPENAI_API_KEY"]
client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.chatanywhere.tech/v1")

@functools.lru_cache(maxsize=None)
def b64_of(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_jsonl(path="./test_cases.jsonl", limit=10):
    data = []
    p = pathlib.Path(path)
    if not p.exists():
        return data
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                data.append(ex)
                if len(data) >= limit:
                    break
            except:
                pass
    return data

examples = load_jsonl("./test_cases.jsonl", limit=50)
desc_front_img = "This image below shows the front camera view in this test case."
desc_side_img = "This image below shows the side camera view in this test case."

def build_examples_content():
    content = []
    for ex in examples:
        sr = ex.get("success_rate","")
        try:
            succ = int(sr.split("/")[0])
            difficulty = 10 - succ
        except:
            difficulty = "NA"
        ex_desc_1 =  f"""[EXAMPLE START]

The test case is described by the following relative data and images:

<rel>

- L2 distance of the test case with the nearest training data: {ex.get('l2_cm', '')}cm
- L2 distance of the blue cube in the test case with the nearest blue cube in the training data: {ex.get('l2_big_cm', '')}cm
- L2 distance of the pink cube in the test case with the nearest pink cube in the training data: {ex.get('l2_small_cm', '')}cm

</rel>
"""
        
        ex_desc_2 = f"""<difficulty>
{difficulty}
</difficulty>

[EXAMPLE END]
"""
        content.extend([
            {"type": "text", "text": ex_desc_1},
            {"type": "text", "text": desc_front_img},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(ex['front_img'])}"}},
            {"type": "text", "text": desc_side_img},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(ex['side_img'])}"}},
            {"type": "text", "text": ex_desc_2},
            
        ])
    return content

# - Relative alignment between cubes: lateral offset, forward/back offset, height differences impacting reach and stacking stability.
# - Camera pose variance: unusual perspective, tilt, extreme foreshortening.
# - Lighting: glare, strong shadows, uneven illumination reducing color/edge contrast.
# - Background or table clutter / visual distractions (if present).

# - Treat the numeric distances as primary quantitative anchors; images contextualize qualitative difficulty modifiers.
# - If qualitative factors are neutral (normal lighting, clear view, standard perspective), do not inflate difficulty—let distances dominate.
# - If distances are low but qualitative issues are severe, you may elevate difficulty proportionally.

BASE_PROMPT = f"""You will evaluate the difficulty of a robot manipulation test case: stacking the pink cube onto the blue cube using a policy trained on a prior dataset.

You will later be given:
1. Structured numeric similarity metrics inside <rel> ... </rel>.
2. Two images: a front camera view and a side camera view of the initial scene.

Goal: Predict an integer difficulty D in [0,10].
Interpretation: approximate success rate ≈ (100 - 10 * D)%. 0 = almost certainly succeeds, 10 = almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Spatial displacement from training data (the provided L2 distances: total, blue-cube-only, pink-cube-only).
- Robot dynamics: will the robot possibly be obstructed on its trajectory? (e.g., reaching for the pink block while the blue block is in the way or if they are too close).
- Occlusion: are any cube faces not completely visible in any of the images? (e.g., if the pink cube is completely/partially hidden behind the blue cube or completely visible in the front image).

These factors are listed according to their priorities from dominant to less dominant.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a single integer inside <difficulty> ... </difficulty> as later instructed.

Use the forthcoming examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""

EXAMPLES_CONTENT = build_examples_content()

def build_message_for_query(q):
    query_desc_1 = f"""[QUERY START]
    
Now I want you to evaluate the difficulty of the following test case according to the examples.

The test case is described by the following relative data and images:

<rel>
- L2 distance of the test case with the nearest training data(blue cube and pink cube distance added): {q.get('l2_cm', '')}cm
- L2 distance of the blue cube in the test case with the nearest blue cube in the training data: {q.get('l2_big_cm', '')}cm
- L2 distance of the pink cube in the test case with the nearest pink cube in the training data: {q.get('l2_small_cm', '')}cm
</rel>
"""
    
    query_desc_2 = f"""First analyze the task in the <think> and </think> tags below:

<think>
[Write your detailed analysis here]
</think>

Then list out all the factors that may influence the outcome of the process and classify them into certain levels of difficulty.

Finally provide a difficulty value from 0 to 10 between <difficulty> and </difficulty> tags.

<difficulty>
[Provide a single integer of 0 to 10]
</difficulty>

[QUERY END]
"""
    content = []
    content.append({"type": "text", "text": BASE_PROMPT})
    content.extend(EXAMPLES_CONTENT)
    content.extend([
        {"type": "text", "text": query_desc_1},
        {"type": "text", "text": desc_front_img},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(q['front_img'])}"}},
        {"type": "text", "text": desc_side_img},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(q['side_img'])}"}},
        {"type": "text", "text": query_desc_2},
    ])
    return content

model_type = "gpt-4o-mini"

async def handle_query(sema, i, q, results, output_dir):
    async with sema:
        print(f"Processing query {i}...")
        try:
            resp = await client.chat.completions.create(
                model=model_type,
                messages=[{"role": "user", "content": build_message_for_query(q)}],
            )
        except Exception as e:
            print(f"Query {i} error: {e}")
            return
        text = resp.choices[0].message.content
        m = re.search(r"<difficulty>(.*?)</difficulty>", text, re.DOTALL)
        diff = m.group(1).strip() if m else "NA"
        results[i] = diff
        with open(os.path.join(output_dir, f"response_{i}.json"), "w", encoding="utf-8") as f:
            json.dump(resp.to_dict(), f, ensure_ascii=False, indent=2)
        resp_path = os.path.join(output_dir, "response_text.txt")
        with open(resp_path, "a", encoding="utf-8") as ft:
            ft.write(f"----- RESPONSE {i} -----\n{text}\n\n")
        print(f"[{i}] difficulty={diff}")

async def main():
    queries = load_jsonl("./queries.jsonl", limit=20)
    results = {}
    sema = asyncio.Semaphore(20)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "./results"
    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, timestamp + "_" + model_type)
    os.makedirs(output_dir, exist_ok=True)
    tasks = [asyncio.create_task(handle_query(sema, i, q, results, output_dir)) for i, q in enumerate(queries)]
    await asyncio.gather(*tasks)
    ordered = [results.get(i, "NA") for i in range(len(queries))]
    with open(os.path.join(output_dir, "difficulties.json"), "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())