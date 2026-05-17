import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
import seaborn as sns

sns.set_theme("notebook", style="darkgrid", palette="pastel")

alpha = 0.9
use_progress = True
batch_paths = [
    # "results/pi0_t10003_full_20260129_162720_qwen3-vl-plus",
    # "results/pi0_t7_full_20251110_220814_qwen3-vl-plus",
    "results/pi0_t10_random_context_20260508_234640_qwen3-vl-plus",
    # "results/pi0_t10_full_20260130_185110_qwen3-vl-plus",
    # "results/pi0_drawer_full_20260207_195342_qwen3-vl-plus,"
    # "results/pi05_box_no_train_data_20260508_162015_qwen3-vl-plus,"
    # "results/pi0_drawer_no_icl_20260508_172635_qwen3-vl-plus",
    # "results/pi0_drawer_random_context_20260508_184340_qwen3-vl-plus",
    # "results/pi05_box_no_icl_20260508_165031_qwen3-vl-plus",
    # "results/pi05_box_random_context_20260508_193523_qwen3-vl-plus",
    # "results/pi0_t10_no_icl_20260507_180435_qwen3-vl-plus",
]


def main(batch_path, collate):
    all_ = {"easy": [], "medium": [], "hard": []}

    result_paths = glob.glob(os.path.join(batch_path, "run_*", "result.json"))
    results = []
    for result_path in result_paths:
        with open(result_path, "r") as f:
            results.append(json.load(f))

    with open(os.path.join(batch_path, "info.json"), "r") as f:
        meta = json.load(f)
    ratios = {"easy": [], "medium": [], "hard": []}

    scores = []
    old_scores = []

    for result in results:
        ratio = {"easy": [], "medium": [], "hard": []}
        succ = {"easy": 0, "medium": 0, "hard": 0}
        total = {"easy": 1e-3, "medium": 1e-3, "hard": 1e-3}
        for item in result:
            res = item["record"].get("progress", [])
            max_progress = item["record"].get("max_progress", 1)
            # suc = res / max_progress
            if isinstance(res, list):
                sum = len(res)
                suc = res.count(max_progress) / sum
            else:
                sum = 1
                suc = (res == max_progress) / sum
            if use_progress:
                for prog in range(1, max_progress):
                    if isinstance(res, list):
                        suc += (prog / max_progress) * (res.count(prog) / sum)
                    else:
                        suc += (prog / max_progress) * ((res == prog) / sum)
            fai = 1 - suc
            if item["difficulty"] == "easy":
                total["easy"] += 1
                succ["easy"] += suc
                all_["easy"].append(suc)
            elif item["difficulty"] == "medium":
                total["medium"] += 1
                succ["medium"] += suc
                all_["medium"].append(suc)
            elif item["difficulty"] == "hard":
                total["hard"] += 1
                succ["hard"] += suc
                all_["hard"].append(suc)
            else:
                total["medium"] += 1
                succ["medium"] += suc
                all_["medium"].append(suc)

            # total_succ += suc * alpha
            # total_weight += alpha
            # score.append(total_succ/total_weight)
            ratio["easy"].append(succ["easy"] / total["easy"])
            ratio["medium"].append(succ["medium"] / total["medium"])
            ratio["hard"].append(succ["hard"] / total["hard"])

        ratios["easy"].append(ratio["easy"])
        ratios["medium"].append(ratio["medium"])
        ratios["hard"].append(ratio["hard"])

    ratios["easy"] = np.array(ratios["easy"])
    ratios["medium"] = np.array(ratios["medium"])
    ratios["hard"] = np.array(ratios["hard"])
    scores = np.array(scores)
    old_scores = np.array(old_scores)
    # for ratio in ratios:
    # axs[0].plot(ratio)
    # for old_ratio in old_ratios:
    #     axs[1].plot(old_ratio, ls="--")
    collate[meta["name"]] = {
        "easy": all_["easy"],
        "medium": all_["medium"],
        "hard": all_["hard"],
    }
    # collate[meta["name"]] = {
    #     "easy": ratios["easy"][1][-1],
    #     "medium": ratios["medium"][1][-1],
    #     "hard": ratios["hard"][1][-1],
    # }
    # print(ratios["easy"][0][-1], ratios["medium"][0][-1], ratios["hard"][0][-1])


if __name__ == "__main__":
    collate_path = "./collate/difficulty.json"

    collate = {}
    if os.path.exists(collate_path):
        with open(collate_path, "r") as f:
            collate.update(json.load(f))
    for batch_path in batch_paths:
        print(batch_path)
        main(batch_path, collate)
    with open(collate_path, "w") as f:
        json.dump(collate, f, indent=4)
