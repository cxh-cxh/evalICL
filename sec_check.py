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

def load_queries(path="./new_test_cases_selected.jsonl", q_size=20):
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
    return queries

def load_examples(path_19="./1_9_test_case.jsonl", path_010 = "./0_10_test_case" ,e_size = 40):
    data1 = []
    data2 = []
    p1 = pathlib.Path(path_19)
    p2 = pathlib.Path(path_010)
    if not p1.exists() or not p2.exists():
        return []
    with p1.open("r", encoding="utf_8") as f1:
        for line1 in f1:
            line1 = line1.strip()
            if not line1:
                continue
            try:
                ex =json.loads(line1)
                data1.append(ex)
            except:
                pass
    with p2.open("r", encoding="utf_8") as f2:
        for line2 in f2:
            line2 = line2.strip()
            if not line2:
                continue
            try:
                ex = json.loads(line2)
                data2.append(ex)
            except:
                pass
    random.shuffle(data2)
    examples = data1+data2[:(e_size-len(data1))]
    return examples


queries=load_queries("./new_test_cases_selected.jsonl")
examples = load_examples(path_19="./1_9_test_case.jsonl", path_010 = "./0_10_test_case.jsonl" ,e_size = 40)
desc_front_img = "This image below shows the front camera view in this test case."
desc_side_img = "This image below shows the side camera view in this test case."

def map_to_category(difficulty):
    if 0 <= difficulty <= 5:
        return "hard"
    elif 6 <= difficulty <= 10:
        return "easy"
    
def build_examples_content():
    content = []
    for ex in examples:
        sr = ex.get("success_rate","")
        try:
            succ = int(sr.split("/")[0])
            difficulty = map_to_category(succ)
        except:
            difficulty = "NA"
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
        
        ex_desc_2 = f"""<difficulty>
{difficulty}
</difficulty>

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

BASE_PROMPT = f"""You will evaluate the difficulty of a robot manipulation test case: stacking the pink cube onto the blue cube using a policy trained on a prior dataset.

The pink cube is 4cm on each side, the blue cube is 8cm on each side.

The work space is a 15cm x 30cm rectangle on a flat horizontal surface with the x axis along the longer side and the y axis along the shorter side. The origin (0,0) is at the bottom left corner of the work space, with positive x going right and positive y going front.

The robot base is at (15, 0) in the work space. The robot can reach the entire work space(reaching for the two far corners of work space presents high difficulty). 

You will later be given:
1. Structured numeric similarity metrics inside <rel> ... </rel>.
2. The position of the two cubes in the reference system of the work space inside <pos> ... </pos>(their bottom left corners' coordinates will be given).

Goal: Predict an difficulty in "easy" or "hard".
Interpretation: "easy" means almost certainly succeeds and "hard" is almost certainly fails.

Key factors to consider, these factors must be explicitly rated in your thinking process point by point:
- Spatial displacement from training data (the provided L2 distances: total, blue-cube-only, pink-cube-only).
- Position of the two cubes in the reference system of the work space, the distance of the two cubes, the distance of each cube with the robot.
- The dynamic constraints of the robot.

Instructions:
- Only consider the key factors listed above and use information is only visible or provided. Do not assume unseen factors.
- Keep reasoning explicit: reference concrete visual or numeric evidence.
- Do not output the difficulty until a [QUERY START] block is given. You will place detailed reasoning inside <think> ... </think> and a output inside <difficulty> ... </difficulty> as later instructed.
- This evaluation is about predicting other model's outcome on the test case, not about predicting the difficulty of the test case itself. The difficulty is not a measure of how hard it is for you, but rather how hard it is for the robot to complete the task in the test case.

Important calibration note:
Do not be overly conservative. If the distances and positions are within normal ranges seen in training data, default to "easy".
Only choose "hard" when there is clear, strong evidence that success is unlikely.



Use the forthcoming examples to calibrate your internal scale. Do not restate these instructions in the answer.
"""

EXAMPLES_CONTENT = build_examples_content()

def build_message_for_query(q, second_check=False):
    query_desc_1 = f"""[QUERY START]
    
Now I want you to evaluate the difficulty of the following test case according to the examples.

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

<failure>
Failure mode observed in similar test cases: {q.get("Reasons for failure", "")}
</failure>

"""
    
    query_desc_2 = f"""First analyze the task in the <think> and </think> tags below:

<think>
[Write your detailed analysis here]
</think>

Then consider a difficulty for each of the factors you analyzed on a scale of "easy" or "hard" in the <rating> and </rating> tags below.


<rating>
[Write your difficulty rating here, e.g., "factor1: easy, factor2: hard, ..."]
</rating>


Finally provide a difficulty in "easy" or "hard" of all factors between <difficulty> and </difficulty> tags.

<difficulty>
[Provide a evaluation of difficulty in "easy" or "hard"]
</difficulty>

[QUERY END]
"""
    
    query_desc_3 = f"""[SECOND QUERY START]

Previously, the test case was classified as "hard".  
Now I want you to refine the classification and determine whether it belongs to:
- "hard-easy": cases that are challenging but manageable, or only slightly hard
- "hard-hard": cases that are very challenging, complex, or significantly harder than typical

The test case is described by the following data:

<rel>
- L2 distance of the test case with the nearest training data (blue cube and pink cube distance added): {q.get('l2_cm', '')}cm
- L2 distance of the blue cube in the test case with the nearest blue cube in the training data: {q.get('l2_big_cm', '')}cm
- L2 distance of the pink cube in the test case with the nearest pink cube in the training data: {q.get('l2_small_cm', '')}cm
</rel>

<pos>
- Position of the blue cube in the work space: ({q.get('big_pos', '')[0]}, {q.get('big_pos', '')[1]})
- Position of the pink cube in the work space: ({q.get('small_pos', '')[0]}, {q.get('small_pos', '')[1]})
</pos>

<failure>
Failure mode observed in similar test cases: {q.get("Reasons for failure", "")}
</failure>

"""
    

    query_desc_4 = f"""First analyze the task in the <think> and </think> tags below:

<think>
[Write your detailed analysis here]
</think>

Then consider a difficulty for each of the factors you analyzed on a scale of "hard-easy" or "hard-hard" in the <rating> and </rating> tags below.


<rating>
[Write your difficulty rating here, e.g., "factor1: hard-easy, factor2: hard-hard, ..."]
</rating>


Finally provide a difficulty in "hard-easy" or "hard-hard" of all factors between "difficulty" tags.

<difficulty>
[Provide a evaluation of difficulty in "hard-easy" or "hard-hard"]
</difficulty>

[SECOND QUERY END]
"""


    content = []
    content.append({"type": "text", "text": BASE_PROMPT})
    content.extend(EXAMPLES_CONTENT)
    content.extend([
        {"type": "text", "text": query_desc_1} if not second_check else {"type": "text", "text": query_desc_3},
        # {"type": "text", "text": desc_front_img},
        # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(q['front_img'])}"}},
        # {"type": "text", "text": desc_side_img},
        # {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_of(q['side_img'])}"}},
        {"type": "text", "text": query_desc_2} if not second_check else {"type": "text", "text": query_desc_4}
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
        m = re.search(r"<difficulty>(.*?)</difficulty>", text, re.DOTALL)
        diff = m.group(1).strip() if m else "NA"
        with open(os.path.join(output_dir, f"response_{i}.json"), "w", encoding="utf-8") as f:
            json.dump(resp.to_dict(), f, ensure_ascii=False, indent=2)
        resp_path = os.path.join(output_dir, "response_text.txt")
        with open(resp_path, "a", encoding="utf-8") as ft:
            ft.write(f"----- RESPONSE {i} -----\n{text}\n\n")
        if diff == "hard":
            print(f"[{i}] initial difficulty = hard, run second check")
            try:
                resp = await client.chat.completions.create(
                    model=model_type,
                    messages=[{"role": "user", "content": build_message_for_query(q,second_check=True)}],
                )
            except Exception as e:
                print(f"Query {i} error: {e}")
                return
            text2 = resp.choices[0].message.content
            m2 = re.search(r"<difficulty>(.*?)</difficulty>", text2, re.DOTALL)
            diff = m2.group(1).strip() if m2 else "NA"
            with open(os.path.join(output_dir, f"response_{i}_2.json"), "w", encoding="utf-8") as f:
                json.dump(resp.to_dict(), f, ensure_ascii=False, indent=2)
            resp_path = os.path.join(output_dir, "response_text.txt")
            with open(resp_path, "a", encoding="utf-8") as ft:
                ft.write(f"----- RESPONSE {i}_2 -----\n{text2}\n\n")
        print(f"[{i}] difficulty={diff}")
        if diff == "hard-hard":
            diff = "hard"
        if diff == "hard-easy":
            diff = "easy"
        results[i] = diff

async def main():
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
    with open(os.path.join(output_dir, "queries.jsonl"), "w", encoding="utf-8") as fq:
        for q in queries:
            fq.write(json.dumps(q, ensure_ascii=False) + "\n")
    with open(os.path.join(output_dir, "examples.jsonl"), "w", encoding="utf-8") as fe:
        for e in examples:
            fe.write(json.dumps(e, ensure_ascii=False) + "\n")

    all_case_re_path = os.path.join(base_dir, "all_case_new.json")
    with open(all_case_re_path, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    queries_result= {}
    for i ,q in enumerate(queries):  
        front_path = q.get("front_img", "")
        m = re.search(r"front_(\d+)\.png", front_path)
        if m:
            data_id = m.group(1)
        else:
            data_id = "NA"
        diff = results[i]
        all_results[data_id]["prediction"].append(diff)
        queries_result[data_id] = diff
    all_results = {k: all_results[k] for k in sorted(all_results.keys(),key=lambda x: int(x))}
    with open(all_case_re_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "queries_result.json"), "w", encoding="utf-8") as f:
        json.dump(queries_result, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    asyncio.run(main())