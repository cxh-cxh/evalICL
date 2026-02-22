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
    "results/pi0_t10_full_20260130_185110_qwen3-vl-plus",
    "results/pi0_t10_5mm_20260131_032123_qwen3-vl-plus",
    "results/pi0_t10_2mm_20260131_043921_qwen3-vl-plus",
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

    ratios = {"easy": [], "medium": [], "hard": []}

    with open(os.path.join(batch_path, "info.json"), "r") as f:
        meta = json.load(f)

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
            res = item["record"].get("progress", [])
            max_progress = item["record"].get("max_progress", 1)
            sum = len(res)
            suc = res.count(max_progress) / sum
            if use_progress:
                for prog in range(1, max_progress):
                    suc += (prog / max_progress) * (res.count(prog) / sum)
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
                    # - succ["easy"]
                    # - succ["medium"]
                    # - succ["hard"]
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

    fig, axs = plt.subplots(2, figsize=(10, 8))
    ratios["easy"] = np.array(ratios["easy"])
    ratios["medium"] = np.array(ratios["medium"])
    ratios["hard"] = np.array(ratios["hard"])
    scores = np.array(scores)
    old_scores = np.array(old_scores)

    # old_scores = old_scores / (1 + old_scores)

    collate[meta["name"]] = {
        "new": np.mean(scores, axis=0).tolist(),
        "old": np.mean(old_scores, axis=0).tolist(),
    }


if __name__ == "__main__":
    collate_path = "./collate/result.json"

    collate = {}
    if os.path.exists(collate_path):
        with open(collate_path, "r") as f:
            collate.update(json.load(f))
    for batch_path in batch_paths:
        print(batch_path)
        main(batch_path, collate)
    with open(collate_path, "w") as f:
        json.dump(collate, f, indent=4)
