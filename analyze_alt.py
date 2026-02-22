import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os,glob
import numpy as np
import seaborn as sns
sns.set_theme("notebook", style="darkgrid", palette="pastel")

alpha = 0.9
use_progress = True
batch_path = "results/pi0_t7_full_20251110_220814_qwen3-vl-plus"

result_paths = glob.glob(os.path.join(batch_path, "run_*","result.json"))

results = []
for result_path in result_paths:
    with open(result_path, "r") as f:
        results.append(json.load(f))

ratios = {"easy":[],"medium":[],"hard":[]}

hit = 0
miss = 0

scores = []
old_scores = []

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
    ratio = {"easy":[],"medium":[],"hard":[]}
    score = []
    total_weight = 1e-3
    old_score = []
    succ = {"easy":0,"medium":0,"hard":0}
    total =  {"easy":1e-3,"medium":1e-3,"hard":1e-3}
    total_succ = 0
    total_fail = 1e-3
    for item in result:
        sr = item["record"].get("success_rate", "")
        progress_1 = item["record"].get("progress_1_rate", "")
        progress_2 = item["record"].get("progress_2_rate", "")
        sum = int(sr.split("/")[1])
        suc = int(sr.split("/")[0]) / sum
        if use_progress:
            if progress_2 != "": # max progress 3
                progress_1 = int(progress_1.split("/")[0]) / sum
                progress_2 = int(progress_2.split("/")[0]) / sum
                suc += progress_1 * 1 / 3 + progress_2 * 2 / 3
            elif progress_1 != "": # max progress 2
                progress_1 = int(progress_1.split("/")[0]) / sum
                suc += progress_1 * 1 / 2
        fai = 1 - suc
        # alpha = old_succ / (old_succ + old_fail) * 0.25 + 0.75
        if item["difficulty"] == "easy":
            # score.append(min(0, suc - succ["easy"] / total["easy"]))
            # alpha = (suc - succ["easy"] / total["easy"]) / 4 + 0.5 
            # alpha = (suc - 1) / 2 + 0.5 
            alpha = 0
            total["easy"] += 1 
            succ["easy"] += suc 
        elif item["difficulty"] == "medium":
            # score.append(suc - succ["medium"] / total["medium"])
            # alpha = (suc - succ["medium"] / total["medium"]) / 4 + 0.5 
            # alpha = (suc - 0.5) / 2 + 0.5 
            alpha = 0.5
            total["medium"] += 1 
            succ["medium"] += suc 
        elif item["difficulty"] == "hard":
            # score.append(max(0, suc - succ["hard"] / total["hard"]))
            # alpha = (suc - succ["hard"] / total["hard"]) / 4 + 0.75 
            # alpha = suc / 2 + 0.5 
            alpha = 1
            total["hard"] += 1 
            succ["hard"] += suc 
        else:
            # score.append(suc - succ["medium"] / total["medium"])
            # alpha = (suc - succ["medium"] / total["medium"]) / 4 + 0.5
            # alpha = (suc - 0.5) / 2 + 0.5 
            alpha = 0.5
            total["medium"] += 1 
            succ["medium"] += suc 

        total_fail += fai * (1-alpha)
        total_succ += suc * alpha
        score.append(total_succ/total_fail)
        # total_succ += suc * alpha
        # total_weight += alpha
        # score.append(total_succ/total_weight)
        old_score.append((succ["easy"] + succ["medium"] + succ["hard"])/(total["easy"] + total["medium"] + total["hard"] - succ["easy"] - succ["medium"] - succ["hard"]))
        ratio["easy"].append(succ["easy"] / total["easy"])
        ratio["medium"].append(succ["medium"] / total["medium"])
        ratio["hard"].append(succ["hard"] / total["hard"])


    ratios["easy"].append(ratio["easy"])
    ratios["medium"].append(ratio["medium"])
    ratios["hard"].append(ratio["hard"])
    scores.append(score)
    old_scores.append(old_score)

fig, axs = plt.subplots(2, figsize=(10, 7))
ratios["easy"] = np.array(ratios["easy"])
ratios["medium"] = np.array(ratios["medium"])
ratios["hard"] = np.array(ratios["hard"])
scores = np.array(scores)
old_scores = np.array(old_scores)
# for ratio in ratios:
    # axs[0].plot(ratio)
# for old_ratio in old_ratios:
#     axs[1].plot(old_ratio, ls="--")
axs[0].set_ylim(0, 1)
axs[0].set_xlim(0, ratios["easy"].shape[1])
axs[0].set_title("Average Success Rate by Predicted Difficulty")
axs[0].plot(np.mean(ratios["easy"],axis=0), color='green', label = "Easy")
axs[0].fill_between(range(ratios["easy"].shape[1]), 
                 np.min(ratios["easy"],axis=0), 
                 np.max(ratios["easy"],axis=0),
                 alpha=0.2, color='green')
axs[0].plot(np.mean(ratios["medium"],axis=0), color='yellow', label = "Medium")
axs[0].fill_between(range(ratios["medium"].shape[1]), 
                 np.min(ratios["medium"],axis=0), 
                 np.max(ratios["medium"],axis=0),
                 alpha=0.2, color='yellow')
axs[0].plot(np.mean(ratios["hard"],axis=0), color='red', label = "Hard")
axs[0].fill_between(range(ratios["hard"].shape[1]), 
                 np.min(ratios["hard"],axis=0), 
                 np.max(ratios["hard"],axis=0),
                 alpha=0.2, color='red')
axs[0].legend()

# scores = np.cumsum(scores,axis = -1) / np.arange(1,scores.shape[1]+1)
scores = scores/(1+scores)
old_scores = old_scores/(1+old_scores)
axs[1].set_ylim(0,1)
axs[1].set_xlim(0, ratios["easy"].shape[1])
axs[1].set_title("Final Score (Success Rate)")
axs[1].plot(np.mean(scores,axis=0),label = "DARE pipeline")
axs[1].plot(np.mean(old_scores,axis=0), label = "Prevalent pipeline")
axs[1].fill_between(range(scores.shape[1]), 
                 np.min(scores,axis=0), 
                 np.max(scores,axis=0),
                 alpha=0.2, label='')
axs[1].fill_between(range(old_scores.shape[1]), 
                 np.min(old_scores,axis=0), 
                 np.max(old_scores,axis=0),
                 alpha=0.2, label='')
axs[1].legend(loc = "upper right")
# line1 = lines.Line2D([50, 50], [0, 1], color="grey", ls="--")
# line2 = lines.Line2D([100, 100], [0, 1], color="grey", ls="--")
# axs[1].add_line(line1)
# axs[1].add_line(line2)
# axs[1].text(x=0, y=0, s="Add Env1", fontsize=12, color="grey", 
#          ha="left", va="bottom")
# axs[1].text(x=51, y=0, s="Add Env2", fontsize=12, color="grey", 
#          ha="left", va="bottom")
# axs[1].text(x=101, y=0, s="Add Env3", fontsize=12, color="grey", 
#          ha="left", va="bottom")

plt.savefig(batch_path.split("/")[-1] + ".png")

# print(np.mean(scores,axis=0)[49],np.mean(old_scores,axis=0)[49])
print(np.mean(scores,axis=0)[-1],np.mean(old_scores,axis=0)[-1])
# print(np.mean(scores,axis=0)[149],np.mean(old_scores,axis=0)[149])
print(np.mean(ratios["easy"],axis=0)[-1],np.mean(ratios["medium"],axis=0)[-1],np.mean(ratios["hard"],axis=0)[-1])

