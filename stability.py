# python
import os, json, re, asyncio, functools, pathlib, base64, random
from openai import AsyncOpenAI
from datetime import datetime

os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
API_KEY = os.environ["OPENAI_API_KEY"]
client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.chatanywhere.tech/v1")
# client = AsyncOpenAI(api_key=API_KEY)

@functools.lru_cache(maxsize=None)
def b64_of(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_jsonl(path="./new_test_cases.jsonl", q_size=30, e_size=40):
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
            except:
                pass
    random.shuffle(data)
    queries = data[:q_size]
    examples = data[q_size:q_size+e_size]
    return queries, examples

queries, examples = load_jsonl("./new_test_cases.jsonl")
desc_front_img = "This image below shows the front camera view in this test case."
desc_side_img = "This image below shows the side camera view in this test case."

def map_to_category(succ):
    if succ <=1:
        return "stable"
    else:
        return "unstable"
    
def build_examples_content():
    content = []
    for ex in examples:
        sr = ex.get("success_rate","")
        try:
            succ = int(sr.split("/")[0])
            stability = map_to_category(succ)
        except:
            stability = "NA"
        ex_desc_1 =  f"""[EXAMPLE START]

The test case is described by the following data:

<rel>

- L2 distance of the test case with the nearest training data: {ex.get('l2_cm', '')}cm
- L2 distance of the blue cube in the test case with the nearest blue cube in the training data: {ex.get('l2_big_cm', '')}cm
- L2 distance of the pink cube in the test case with the nearest pink cube in the training data: {ex.get('l2_small_cm', '')}cm

</rel>

<pos>
- Position of the blue cube in the work space: ({ex.get('big_pos', '')[0]}, {ex.get('big_pos', '')[1]})
- Position of the pink cube in the work space: ({ex.get('small_pos', '')[0]}, {ex.get('small_pos', '')[1]})
</pos>
"""
        
        ex_desc_2 = f"""<stability>
{stability}
</stability>

[EXAMPLE END]
"""
        content.extend([
            {"type": "text", "text": ex_desc_1},
            # {"type": "text", "text": desc_front_img},
            # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(ex['front_img'])}"}},
            # {"type": "text", "text": desc_side_img},
            # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(ex['side_img'])}"}},
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
# - Robot dynamics: will the robot possibly be obstructed on its trajectory? (e.g., reaching for the pink block while the blue block is in the way or if they are too close).
# - Occlusion: are any cube faces not completely visible in any of the images? (e.g., if the pink cube is completely/partially hidden behind the blue cube or completely visible in the front image).

BASE_PROMPT = f"""You will evaluate the stability of a robot manipulation test case: stacking the pink cube onto the blue cube using a policy trained on a prior dataset.

The pink cube is 4cm on each side, the blue cube is 8cm on each side.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1. Structured numeric similarity metrics inside <rel> ... </rel>.
2. The position of the two cubes in the reference system of the work space inside <pos> ... </pos>(their bottom left corners' coordinates will be given).

Goal: Predict an stability in "stable" or "unstable".
Interpretation: "stable" means the robot is always successful or failed, the outcome is predictable; "unstable" means the robot sometimes succeeds and sometimes fails, the outcome is unpredictable.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Spatial displacement from training data (the provided L2 distances: total, blue-cube-only, pink-cube-only).
- Position of the two cubes in the reference system of the work space, the distance of the two cubes, the distance of each cube with the robot.
- The dynamic constraints of the robot.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- Do not output the stablity until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <stability> ... </stablity> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the stability of the test case itself. The stabilitty is not a measure of how stable it is for you, but rather how stable it is for the robot to complete the task in the test case.


Use the forthcoming examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""

EXAMPLES_CONTENT = build_examples_content()

def build_message_for_query(q):
    query_desc_1 = f"""[QUERY START]
    
Now I want you to evaluate the stability of the following test case according to the examples.

The test case is described by the following data:

<rel>
- L2 distance of the test case with the nearest training data(blue cube and pink cube distance added): {q.get('l2_cm', '')}cm
- L2 distance of the blue cube in the test case with the nearest blue cube in the training data: {q.get('l2_big_cm', '')}cm
- L2 distance of the pink cube in the test case with the nearest pink cube in the training data: {q.get('l2_small_cm', '')}cm
</rel>

<pos>
- Position of the blue cube in the work space: ({q.get('big_pos', '')[0]}, {q.get('big_pos', '')[1]})
- Position of the pink cube in the work space: ({q.get('small_pos', '')[0]}, {q.get('small_pos', '')[1]})
</pos>


"""
    
    query_desc_2 = f"""First analyze the task in the <think> and </think> tags below:

<think>
[Write your detailed analysis here]
</think>

Then consider a stability for each of the factors you analyzed on a scale of "stable" or "unstable" in the <rating> and </rating> tags below.


<rating>
[Write your stability rating here, e.g., "factor1: stable, factor2: unstable, ..."]
</rating>


Finally provide a stablity in "stable" or "unstable" of all factors between <stability> and </stability> tags.

<stability>
[Provide a evaluation of stability in "stable" or "unstable"]
</stability>

[QUERY END]
"""
    content = []
    content.append({"type": "text", "text": BASE_PROMPT})
    content.extend(EXAMPLES_CONTENT)
    content.extend([
        {"type": "text", "text": query_desc_1},
        # {"type": "text", "text": desc_front_img},
        # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(q['front_img'])}"}},
        # {"type": "text", "text": desc_side_img},
        # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(q['side_img'])}"}},
        {"type": "text", "text": query_desc_2},
    ])
    return content

model_type = "gpt-4o"

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
        m = re.search(r"<stability>(.*?)</stability>", text, re.DOTALL)
        diff = m.group(1).strip() if m else "NA"
        results[i] = diff
        with open(os.path.join(output_dir, f"response_{i}.json"), "w", encoding="utf-8") as f:
            json.dump(resp.to_dict(), f, ensure_ascii=False, indent=2)
        resp_path = os.path.join(output_dir, "response_text.txt")
        with open(resp_path, "a", encoding="utf-8") as ft:
            ft.write(f"----- RESPONSE {i} -----\n{text}\n\n")
        print(f"[{i}] stability={diff}")

async def main():
    results = {}
    sema = asyncio.Semaphore(40)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "./results"
    os.makedirs(base_dir, exist_ok=True)
    output_dir = os.path.join(base_dir, timestamp + "_" + model_type + "_stability_910")
    os.makedirs(output_dir, exist_ok=True)
    tasks = [asyncio.create_task(handle_query(sema, i, q, results, output_dir)) for i, q in enumerate(queries)]
    await asyncio.gather(*tasks)
    ordered = [results.get(i, "NA") for i in range(len(queries))]
    with open(os.path.join(output_dir, "stabilities.json"), "w", encoding="utf-8") as f:
        json.dump(ordered, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "queries.jsonl"), "w", encoding="utf-8") as fq:
        for q in queries:
            fq.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "examples.jsonl"), "w", encoding="utf-8") as fe:
        for e in examples:
            fe.write(json.dumps(e, ensure_ascii=False) + "\n")
if __name__ == "__main__":
    asyncio.run(main())