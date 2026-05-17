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
    # "results/pi0_t10_full_20260130_185110_qwen3-vl-plus",
    # "results/pi0_t10_5mm_20260131_032123_qwen3-vl-plus",
    # "results/pi0_t10_2mm_20260131_043921_qwen3-vl-plus",
    # "results/pi0_drawer_full_20260207_195342_qwen3-vl-plus",
    # "results/pi05_box_no_train_data_20260508_162015_qwen3-vl-plus",
    "results/pi0_t10_random_context_20260508_234640_qwen3-vl-plus",
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


weight_func = {
    "linear": {
        "easy": (lambda a, b: 1 - bilinear(a, b)),
        "medium": (lambda a, b: 0.5),
        "hard": (lambda a, b: bilinear(a, b)),
    },
    "mult": {
        "easy": (lambda a, b: bilinear2(a, b) * (1 - a)),
        "medium": (lambda a, b: bilinear2(a, b) * bilinear2(a, 0.5)),
        "hard": (lambda a, b: bilinear2(a, b) * a),
    },
    "mult_sqrt": {
        "easy": (lambda a, b: np.sqrt(bilinear2(a, b) * (1 - a))),
        "medium": (lambda a, b: np.sqrt(bilinear2(a, b) * bilinear2(a, 0.5))),
        "hard": (lambda a, b: np.sqrt(bilinear2(a, b) * a)),
    },
    "l2": {
        "easy": (lambda a, b: np.sqrt(((a - b) ** 2 + (1 - a) ** 2) / 2)),
        "medium": (lambda a, b: np.sqrt(((a - b) ** 2 + (0.5 - a) ** 2) / 2)),
        "hard": (lambda a, b: np.sqrt(((a - b) ** 2 + a**2) / 2)),
    },
    "gaussian": {
        "easy": (
            lambda a, b: 1 / (np.exp(-((a - b) ** 2)) + np.exp(-((1 - a) ** 2))) - 0.5
        ),
        "medium": (
            lambda a, b: 1 / (np.exp(-((a - b) ** 2)) + np.exp(-((0.5 - a) ** 2))) - 0.5
        ),
        "hard": (lambda a, b: 1 / (np.exp(-((a - b) ** 2)) + np.exp(-(a**2))) - 0.5),
    },
}

func_name = "l2"


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
            # alpha = old_succ / (old_succ + old_fail) * 0.25 + 0.75
            if item["difficulty"] == "easy":
                alpha = weight_func[func_name]["easy"](
                    suc, succ["easy"] / total["easy"]
                )
                total["easy"] += 1
                succ["easy"] += suc
            elif item["difficulty"] == "medium":
                alpha = weight_func[func_name]["medium"](
                    suc, succ["medium"] / total["medium"]
                )
                total["medium"] += 1
                succ["medium"] += suc
            elif item["difficulty"] == "hard":
                alpha = weight_func[func_name]["hard"](
                    suc, succ["hard"] / total["hard"]
                )
                total["hard"] += 1
                succ["hard"] += suc
            else:
                alpha = weight_func[func_name]["medium"](
                    suc, succ["medium"] / total["medium"]
                )
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
    collate_path = f"./collate/result_{func_name}.json"

    collate = {}
    if os.path.exists(collate_path):
        with open(collate_path, "r") as f:
            collate.update(json.load(f))
    for batch_path in batch_paths:
        print(batch_path)
        main(batch_path, collate)
    with open(collate_path, "w") as f:
        json.dump(collate, f, indent=4)
