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
    "results/pi0_t10003_full_20260129_162720_qwen3-vl-plus",
    "results/pi0_t7_full_20251110_220814_qwen3-vl-plus",
    "results/pi05_task40_full_20260130_024756_qwen3-vl-plus",
    "results/pi0_t10003_sim_full_20260131_004414_qwen3-vl-plus",
]


def bilinear(x, mid):
    if x < mid:
        return 0.5 / mid * x
    else:
        return 0.5 / (1 - mid) * (x - mid) + 0.5


def bilinear2(x, mid):
    if x < mid:
        return 1 / mid * (mid - x)
    else:
        return 1 / (1 - mid) * (x - mid)


def main(batch_path, collate):
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
            sr = item["record"].get("success_rate", "")
            progress_1 = item["record"].get("progress_1_rate", "")
            progress_2 = item["record"].get("progress_2_rate", "")
            sum = int(sr.split("/")[1])
            suc = int(sr.split("/")[0]) / sum
            if use_progress:
                if progress_2 != "":  # max progress 3
                    progress_1 = int(progress_1.split("/")[0]) / sum
                    progress_2 = int(progress_2.split("/")[0]) / sum
                    suc += progress_1 * 1 / 3 + progress_2 * 2 / 3
                elif progress_1 != "":  # max progress 2
                    progress_1 = int(progress_1.split("/")[0]) / sum
                    suc += progress_1 * 1 / 2
            fai = 1 - suc
            # alpha = old_succ / (old_succ + old_fail) * 0.25 + 0.75
            if item["difficulty"] == "easy":
                alpha = 1 - bilinear(suc, succ["easy"] / total["easy"])
                # alpha = bilinear2(suc, succ["easy"] / total["easy"])
                # alpha = 1 - suc
                total["easy"] += 1
                succ["easy"] += suc
            elif item["difficulty"] == "medium":
                # alpha = bilinear2(suc, succ["medium"] / total["medium"])
                alpha = 0.5
                total["medium"] += 1
                succ["medium"] += suc
            elif item["difficulty"] == "hard":
                # alpha = bilinear2(suc, succ["hard"] / total["hard"])
                alpha = bilinear(suc, succ["hard"] / total["hard"])
                # alpha = suc
                total["hard"] += 1
                succ["hard"] += suc
            else:
                # alpha = bilinear2(suc, succ["medium"] / total["medium"])
                alpha = 0.5
                total["medium"] += 1
                succ["medium"] += suc

            # total_succ += suc * alpha
            # total_weight += alpha
            # score.append(total_succ/total_weight)
            ratio["easy"].append(succ["easy"] / total["easy"])
            ratio["medium"].append(succ["medium"] / total["medium"])
            ratio["hard"].append(succ["hard"] / total["hard"])

        ratios["easy"].append(ratio["easy"])
        ratios["medium"].append(ratio["medium"])
        ratios["hard"].append(ratio["hard"])

    fig, axs = plt.subplots(2, figsize=(10, 8))
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
        "easy": np.mean(ratios["easy"], axis=0)[-1],
        "medium": np.mean(ratios["medium"], axis=0)[-1],
        "hard": np.mean(ratios["hard"], axis=0)[-1],
    }


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
