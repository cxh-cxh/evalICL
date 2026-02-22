# python
import os, json, re, asyncio, functools, pathlib, base64, random
from openai import AsyncOpenAI, OpenAI
from datetime import datetime
from retrieval import Retriever
from config import configs
from templete_new import *
from video_sampler import sample_video
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--name", type=str, default=None, help="name of the batch")
parser.add_argument("-c", "--count", type=int, default=1, help="count of runs")
parser.add_argument("--policy", type=str, help="name of the policy")
parser.add_argument(
    "--task", type=str, default="t10003", help="name of the task (for base prompt)"
)
parser.add_argument("--vlm", type=str, default="qwen3-vl-plus", help="name of the vlm")
parser.add_argument("--k", type=int, default=3, help="count of neighbours")
parser.add_argument(
    "--starting_num",
    type=int,
    default=0,
    help="count of initial records in the retriever",
)
parser.add_argument(
    "--queries_num",
    type=int,
    default=100,
    help="count of queries",
)
parser.add_argument("--envs", nargs="+", type=str, help="test environments")
parser.add_argument(
    "--train_envs", nargs="*", type=str, default=[], help="train environments"
)
parser.add_argument(
    "--video_root",
    type=str,
    default="/mnt/sdb/benchmarking/videos/pi0_2026_1_drawer_aug_drawer",
)
args = parser.parse_args()


def load_vlm(vlm_type):
    config = configs[vlm_type]
    os.environ["OPENAI_API_KEY"] = config.api_key
    API_KEY = os.environ["OPENAI_API_KEY"]
    client = AsyncOpenAI(api_key=API_KEY, base_url=config.base_url)
    return client


# @functools.lru_cache(maxsize=None)
def b64_of(img, type="jpeg") -> str:
    success, encoded_image = cv2.imencode(".jpg", img)
    if not success:
        raise ValueError("图像编码失败")
    return base64.b64encode(encoded_image.tobytes()).decode("utf-8")


def load_jsonl(path="./test_cases.jsonl"):
    data = []
    p = pathlib.Path(path)
    if not p.exists():
        return data
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                data.append(ex)
            except:
                pass
    return data


def map_to_category(difficulty):
    if 0 <= difficulty <= 2:
        return "hard"
    elif 8 <= difficulty <= 10:
        return "easy"
    elif 3 <= difficulty <= 7:
        return "medium"
    else:
        return "NA"


def build_examples_content(test_records, test_names):
    content = []
    for i in range(len(test_records)):
        ex = test_records[i]

        front_imgs = sample_video(
            os.path.join(args.video_root, ex["video"]["front"]),
            # sample_rate=30,
            num=10,
        )

        side_imgs = sample_video(
            os.path.join(args.video_root, ex["video"]["side"]),
            # sample_rate=30,
            num=10,
        )
        imgs = []
        for j in range(len(front_imgs)):
            imgs.append({"front": front_imgs[j], "side": side_imgs[j]})

        ex_desc_1 = example_templete_1.render(ex, count=len(imgs), is_video=True)
        fail_desc = ""
        if ex["is_train"] == 0:
            prog = ex["progress"]
            if prog == ex["max_progress"]:
                success_rate = "1/1"
            else:
                success_rate = "0/1"
            if prog in progress_desc[args.task]:
                fail_desc += fail_desc_templete.render(
                    desc=progress_desc[args.task][prog],
                    count=1,
                    total=1,
                )
        ex_desc_2 = example_templete_2.render(
            ex,
            success_rate=success_rate,
            fail_desc=fail_desc,
        )
        content.extend(
            [
                {"type": "text", "text": ex_desc_1},
            ]
        )
        id = 0
        for img in imgs:
            content.extend(
                [
                    {"type": "text", "text": video_frame_templete.render(index=id)},
                    {"type": "text", "text": desc_front_img_templete.render(ex)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_of(img['front'])}"
                        },
                    },
                    {"type": "text", "text": desc_side_img_templete.render(ex)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_of(img['side'])}"
                        },
                    },
                ]
            )
            id += 1
        content.extend(
            [
                {"type": "text", "text": ex_desc_2},
            ]
        )
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

base_prompt_templetes = {"drawer": base_prompt_templete_drawer}

BASE_PROMPT = base_prompt_templetes[args.task].render(k=args.k)


def build_message_for_query(args, test_record, test_name, retriever):
    front_imgs = sample_video(
        os.path.join(args.video_root, test_record["video"]["front"]),
        # sample_rate=30,
        num=10,
    )

    side_imgs = sample_video(
        os.path.join(args.video_root, test_record["video"]["side"]),
        # sample_rate=30,
        num=10,
    )
    imgs = []
    for i in range(len(front_imgs)):
        imgs.append({"front": front_imgs[i], "side": side_imgs[i]})

    query_desc_1 = query_templete_1.render(count=len(imgs), is_video=True)
    query_desc_2 = query_templete_2.render()

    content = []
    content.append({"type": "text", "text": BASE_PROMPT})

    retrieve = retriever.retrieve(test_record, test_name, args.k, True)
    content.extend(build_examples_content(retrieve[1], retrieve[2]))

    content.extend(
        [
            {"type": "text", "text": query_desc_1},
        ]
    )
    id = 0
    for img in imgs:
        content.extend(
            [
                {"type": "text", "text": video_frame_templete.render(index=id)},
                {"type": "text", "text": desc_front_img_templete.render()},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_of(img['front'])}"
                    },
                },
                {"type": "text", "text": desc_side_img_templete.render()},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_of(img['side'])}"
                    },
                },
            ]
        )
        id += 1
    content.append({"type": "text", "text": query_desc_2})

    return content


async def handle_query(sema, i, client, vlm_type, query_msg, results, output_dir):
    async with sema:
        print(f"Processing query {i}...")
        try:
            resp = await client.chat.completions.create(
                model=vlm_type,
                messages=[{"role": "user", "content": query_msg}],
                stream=False,
                extra_body=configs[vlm_type].extra_body,
            )
        except Exception as e:
            print(f"Query {i} error: {e}")
            return
        text = resp.choices[0].message.content
        m = re.search(r"<difficulty>(.*?)</difficulty>", text, re.DOTALL)
        diff = m.group(1).strip() if m else "NA"
        with open(
            os.path.join(output_dir, f"response_{i}.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(resp.to_dict(), f, ensure_ascii=False, indent=2)
        resp_path = os.path.join(output_dir, "response_text.txt")
        with open(resp_path, "a", encoding="utf-8") as ft:
            ft.write(f"----- RESPONSE {i} -----\n{text}\n\n")
        print(f"[{i}] difficulty={diff}")
        results[i] = diff


async def main():

    results = {}
    sema = asyncio.Semaphore(30)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "results"
    # base_dir = "/media/rhos/6fa5c3bb-4171-4883-bd72-51d36b407285/lerobot/icl_result"
    batch_dir = (
        f"{args.name + '_' if args.name is not None else ''}{timestamp}_{args.vlm}"
    )
    os.makedirs(os.path.join(base_dir, batch_dir), exist_ok=True)

    client = load_vlm(args.vlm)

    with open(os.path.join(base_dir, batch_dir, "info.json"), "w") as f:
        json.dump({"name": args.name, "vlm_model": args.vlm, "count": args.count}, f)

    for run in range(args.count):

        all_records = []
        all_test_names = []
        for test_name in args.envs:
            data = load_jsonl("data/" + args.policy + "/" + test_name + ".jsonl")
            all_records += data
            all_test_names += [test_name] * len(data)

        train_records = []
        train_names = []
        for train_name in args.train_envs:
            data = load_jsonl("data/" + args.policy + "/" + train_name + ".jsonl")
            train_records += data
            train_names += [train_name] * len(data)

        print("Total records: ", len(all_records))

        indices = list(range(len(all_records)))
        random.shuffle(indices)
        all_records = [all_records[index] for index in indices]
        all_test_names = [all_test_names[index] for index in indices]

        retriever = Retriever(
            starting_test_records=train_records + all_records[: args.starting_num],
            starting_test_names=train_names + all_test_names[: args.starting_num],
            model_name=args.policy,
            img_emb_path="data/video_emb.hdf5",
            is_video=True,
        )
        all_records = all_records[args.starting_num :]
        all_test_names = all_test_names[args.starting_num :]

        queries = []
        for i in range(args.queries_num):

            queries.append(
                build_message_for_query(
                    args, all_records[i], all_test_names[i], retriever
                )
            )

        output_dir = os.path.join(base_dir, batch_dir, f"run_{run}")
        os.makedirs(output_dir, exist_ok=True)
        with open(
            os.path.join(output_dir, "queries.jsonl"), "w", encoding="utf-8"
        ) as fq:
            for q in queries:
                fq.write(
                    json.dumps(
                        [q_ for q_ in q if q_["type"] == "text"], ensure_ascii=False
                    )
                    + "\n"
                )
        # for q in queries:
        #     print([q_ for q_ in q if q_["type"] == 'text'])
        tasks = [
            asyncio.create_task(
                handle_query(sema, i, client, args.vlm, q, results, output_dir)
            )
            for i, q in enumerate(queries)
        ]
        await asyncio.gather(*tasks)
        ordered = [results.get(i, "NA") for i in range(len(queries))]
        final_result = [
            {"difficulty": ordered[i], "record": all_records[i]}
            for i in range(args.queries_num)
        ]
        with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
