import json
import os, glob
import numpy as np

from plot_factory import *


alpha = 0.9
use_progress = True
batch_paths = [
    # "results/pi0_t10003_sim_full_20260131_004414_qwen3-vl-plus",
    "results/pi0_t10003_000_20260225_153314_qwen3-vl-plus",
    "results/pi0_t10003_001_20260225_161406_qwen3-vl-plus",
    "results/pi0_t10003_002_20260225_164636_qwen3-vl-plus",
    "results/pi0_t10003_006_20260225_184929_qwen3-vl-plus",
    "results/pi0_t10003_007_20260225_181622_qwen3-vl-plus",
    "results/pi0_t7_003_20260225_200353_qwen3-vl-plus",
    # "results/pi0_t10003_20251113_175854_qwen3-vl-plus",
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


def main(batch_path):
    result_paths = glob.glob(os.path.join(batch_path, "run_*", "result.json"))

    results = []
    for result_path in result_paths:
        with open(result_path, "r") as f:
            results.append(json.load(f))

    ratios = {"easy": [], "medium": [], "hard": []}

    scores = []
    old_scores = []

    for result in results:
        ratio = {"easy": [], "medium": [], "hard": []}
        score = []
        total_weight = 1e-3
        old_score = []
        succ = {"easy": 0, "medium": 0, "hard": 0}
        total = {"easy": 1e-3, "medium": 1e-3, "hard": 1e-3}
        total_succ = 0
        total_fail = 1e-3
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

            total_succ += suc * alpha
            total_weight += alpha
            score.append(total_succ / total_weight)
            # total_succ += suc * alpha
            # total_weight += alpha
            # score.append(total_succ/total_weight)
            old_score.append(
                (succ["easy"] + succ["medium"] + succ["hard"])
                / (
                    total["easy"]
                    + total["medium"]
                    + total["hard"]
                    - succ["easy"]
                    - succ["medium"]
                    - succ["hard"]
                )
            )
            ratio["easy"].append(succ["easy"] / total["easy"])
            ratio["medium"].append(succ["medium"] / total["medium"])
            ratio["hard"].append(succ["hard"] / total["hard"])

        ratios["easy"].append(ratio["easy"])
        ratios["medium"].append(ratio["medium"])
        ratios["hard"].append(ratio["hard"])
        scores.append(score)
        old_scores.append(old_score)

    ratios["easy"] = np.array(ratios["easy"])
    ratios["medium"] = np.array(ratios["medium"])
    ratios["hard"] = np.array(ratios["hard"])
    scores = np.array(scores)
    old_scores = np.array(old_scores)
    # for ratio in ratios:
    # axs[0].plot(ratio)
    # for old_ratio in old_ratios:
    #     axs[1].plot(old_ratio, ls="--")

    # scores = np.cumsum(scores,axis = -1) / np.arange(1,scores.shape[1]+1)
    # scores = scores/(1+scores)
    old_scores = old_scores / (1 + old_scores)

    data = {
        "length": ratios["easy"].shape[1],
        "easy": ratios["easy"],
        "medium": ratios["medium"],
        "hard": ratios["hard"],
        "scores": scores,
        "old_scores": old_scores,
    }

    axs = get_subplots(2)
    difficulty_plot(axs[0], data)

    print(scores[:, -1])
    savefig("dynamic_weight/" + batch_path.split("/")[-1] + ".png")


if __name__ == "__main__":
    for batch_path in batch_paths:
        print(batch_path)
        main(batch_path)
