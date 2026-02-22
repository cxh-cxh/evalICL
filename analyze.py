import json
from matplotlib import pyplot as plt
import os,glob
import numpy as np

alpha = 0.9

# result_paths = [
#     "results/20251105_200236_qwen3-vl-plus_3/result.json",
#     "results/20251105_200831_qwen3-vl-plus_3/result.json",
#     "results/20251105_201300_qwen3-vl-plus_3/result.json",
#     # "results/20251105_202244_poison_qwen3-vl-plus_3/result.json",
#     # "results/20251105_202622_poison_qwen3-vl-plus_3/result.json",
#     # "results/20251105_203017_poison_qwen3-vl-plus_3/result.json",
# ]

batch_path = "results/pi0_t10003_20251112_172955_qwen3-vl-plus"

result_paths = glob.glob(os.path.join(batch_path, "run_*","result.json"))

results = []
for result_path in result_paths:
    with open(result_path, "r") as f:
        results.append(json.load(f))

ratios = []
old_ratios = []

hit = 0
miss = 0


def map_to_category(suc):
    if 0 <= suc <= 0.1:
        return "hard"
    elif 0.9 <= suc <= 1:
        return "easy"
    elif 0.1 < suc < 0.9:
        return "medium"
    else:
        return None


for result in results:
    ratio = []
    old_ratio = []
    succ = 0
    fail = 1e-3
    old_succ = 0
    old_fail = 1e-3
    for item in result:
        sr = item["record"].get("success_rate", "")
        total = int(sr.split("/")[1])
        suc = int(sr.split("/")[0]) / total
        fai = 1 - suc
        old_succ += suc * total
        old_fail += fai * total
        # alpha = old_succ / (old_succ + old_fail) * 0.25 + 0.75
        if item["difficulty"] == "easy":
            fail = fail + fai * alpha
            succ = succ + suc * (1 - alpha)
        elif item["difficulty"] == "medium":
            fail = fail + fai * 0.5
            succ = succ + suc * 0.5
        elif item["difficulty"] == "hard":
            succ = succ + suc * alpha
            fail = fail + fai * (1 - alpha)
        else:
            fail = fail + fai * 0.5
            succ = succ + suc * 0.5
        ratio.append(succ / fail)
        old_ratio.append(old_succ / (old_fail))

        if item["difficulty"] == map_to_category(suc):
            hit += 1
        else:
            miss += 1

    ratios.append(ratio)
    old_ratios.append(old_ratio)

fig, axs = plt.subplots(2)
ratios = np.array(ratios)
old_ratios = np.array(old_ratios)
# for ratio in ratios:
    # axs[0].plot(ratio)
# for old_ratio in old_ratios:
#     axs[1].plot(old_ratio, ls="--")
axs[0].set_ylim(0, 2)
axs[0].set_xlim(0, ratios.shape[1])
axs[0].set_title("Mean")
axs[0].plot(np.mean(ratios,axis=0))
axs[0].fill_between(range(ratios.shape[1]), 
                 np.min(ratios,axis=0), 
                 np.max(ratios,axis=0),
                 alpha=0.2, color='skyblue', label='')
axs[0].plot(np.mean(old_ratios,axis=0), color='orange')
axs[0].fill_between(range(old_ratios.shape[1]), 
                 np.min(old_ratios,axis=0), 
                 np.max(old_ratios,axis=0),
                 alpha=0.2, color='orange', label='')

# axs[1].set_ylim(0, 1)
# axs[1].set_xlim(0, 100)
# axs[1].set_title("Current method")
axs[1].set_ylim(0, 0.1)
axs[1].set_xlim(0, ratios.shape[1])
axs[1].set_title("Var")
axs[1].plot(np.var(ratios,axis=0))
axs[1].plot(np.var(old_ratios,axis=0),ls = '--')

plt.savefig("test.png")

print(hit / (hit + miss))
print(np.mean(old_ratios,axis=0)[-1],np.mean(ratios,axis=0)[-1])