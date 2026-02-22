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
    "results/pi0_drawer_full_20260207_193820_qwen3-vl-plus",
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


def rolling_variance(x, window_size, ddof=1):
    """
    计算序列在滑动窗口内的方差

    Parameters:
    -----------
    x : array_like
        输入序列
    window_size : int
        窗口长度
    ddof : int, default=1
        Delta Degrees of Freedom（样本方差用1，总体方差用0）

    Returns:
    --------
    var : ndarray
        方差序列，长度为 len(x) - window_size + 1
    """
    x = np.asarray(x)
    n = len(x)

    if window_size > n:
        raise ValueError(f"窗口大小 {window_size} 不能大于序列长度 {n}")

    # 方法1: 使用滑动窗口视图（NumPy 1.20+，直观易懂）
    if hasattr(np.lib.stride_tricks, "sliding_window_view"):
        windows = np.lib.stride_tricks.sliding_window_view(x, window_size)
        return np.var(windows, axis=1, ddof=ddof)

    # 方法2: 使用卷积（更高效，内存友好，适用于大数组）
    else:
        # 计算窗口内元素的和与平方和
        kernel = np.ones(window_size)
        sum_x = np.convolve(x, kernel, mode="valid")
        sum_x2 = np.convolve(x**2, kernel, mode="valid")

        # Var(X) = E[X^2] - (E[X])^2
        mean = sum_x / window_size
        variance = (sum_x2 / window_size) - mean**2

        # 样本方差修正（无偏估计）
        if ddof == 1:
            variance = variance * window_size / (window_size - 1)

        return variance


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
            progress = item["record"].get("progress", 0)
            max_progress = item["record"].get("max_progress", 0)
            suc = int(progress == max_progress)
            if use_progress:
                suc = progress / max_progress
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

    fig, axs = plt.subplots(3, figsize=(10, 12))
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

    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0, ratios["easy"].shape[1])
    axs[0].set_title("Difficulty Prediction")
    axs[0].plot(np.mean(ratios["easy"], axis=0), color="green", label="Easy")
    axs[0].fill_between(
        range(ratios["easy"].shape[1]),
        np.min(ratios["easy"], axis=0),
        np.max(ratios["easy"], axis=0),
        alpha=0.2,
        color="green",
    )
    axs[0].plot(np.mean(ratios["medium"], axis=0), color="yellow", label="Medium")
    axs[0].fill_between(
        range(ratios["medium"].shape[1]),
        np.min(ratios["medium"], axis=0),
        np.max(ratios["medium"], axis=0),
        alpha=0.2,
        color="yellow",
    )
    axs[0].plot(np.mean(ratios["hard"], axis=0), color="red", label="Hard")
    axs[0].fill_between(
        range(ratios["hard"].shape[1]),
        np.min(ratios["hard"], axis=0),
        np.max(ratios["hard"], axis=0),
        alpha=0.2,
        color="red",
    )
    axs[0].legend()

    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(0, ratios["easy"].shape[1])
    axs[1].set_xlabel("test cases")
    axs[1].set_title("Final Score (Progress Rate)")
    axs[1].plot(np.mean(scores, axis=0), label="DARE pipeline")
    axs[1].plot(np.mean(old_scores, axis=0), label="Common pipeline")
    axs[1].fill_between(
        range(scores.shape[1]),
        np.min(scores, axis=0),
        np.max(scores, axis=0),
        alpha=0.2,
        label="",
    )
    axs[1].fill_between(
        range(old_scores.shape[1]),
        np.min(old_scores, axis=0),
        np.max(old_scores, axis=0),
        alpha=0.2,
        label="",
    )
    axs[1].legend(loc="upper right")

    axs[2].set_xlim(0, ratios["easy"].shape[1])
    axs[2].set_xlabel("test cases")
    axs[2].set_title("Rolling Variance of Final Score")
    axs[2].plot(rolling_variance(np.mean(scores, axis=0), 10), label="DARE pipeline")
    axs[2].plot(
        rolling_variance(np.mean(old_scores, axis=0), 10), label="Common pipeline"
    )
    axs[2].set_yscale("log")
    axs[2].legend(loc="upper right")
    plt.tight_layout()
    print(scores[:, -1])
    plt.savefig("dynamic_weight/" + batch_path.split("/")[-1] + ".png")
    print(ratios["easy"][0][-1], ratios["medium"][0][-1], ratios["hard"][0][-1])


if __name__ == "__main__":
    for batch_path in batch_paths:
        print(batch_path)
        main(batch_path)
