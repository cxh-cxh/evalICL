import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os,glob
import numpy as np
import seaborn as sns
sns.set_theme("notebook", style="darkgrid", palette="pastel")

alpha = 0.9
use_progress = True
batch_paths = [
                "results/pi0_t10003_mixed_20251113_010221_qwen3-vl-plus",
               "results/pi0_t10003_staged_1_2_3_20251112_154135_qwen3-vl-plus",
               "results/pi0_t10003_staged_2_3_1_20251113_125254_qwen3-vl-plus",
               "results/pi0_t10003_staged_3_1_2_20251113_160912_qwen3-vl-plus",
               "results/pi0_t7_staged_013245_20251113_221654_qwen3-vl-plus",
               "results/poison_easy_20251108_010907_qwen3-vl-plus",
               "results/poison_hard_20251108_115704_qwen3-vl-plus",
               ]

def bilinear(x, mid):
    if x<mid:
        return 0.5/mid * x
    else:
        return 0.5/(1-mid) * (x - mid) + 0.5

def bilinear2(x, mid):
    if x<mid:
        return 1/mid * (mid - x)
    else:
        return 1/(1 - mid) * (x - mid)

def main(batch_path):
    result_paths = glob.glob(os.path.join(batch_path, "run_*","result.json"))   

    results = []
    for result_path in result_paths:
        with open(result_path, "r") as f:
            results.append(json.load(f))    

    ratios = {"easy":[],"medium":[],"hard":[]}  

    scores = []
    old_scores = [] 

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
                if len(ratios["easy"])>0:
                    alpha = (1 - np.exp(-(suc - succ["easy"] / total["easy"])**2 / np.var(np.array(ratios["easy"])) / 2)) * (1-suc)
                else:
                    alpha = 1
                total["easy"] += 1 
                succ["easy"] += suc 
            elif item["difficulty"] == "medium":
                if len(ratios["medium"])>0:
                    alpha = (1 - np.exp(-(suc - succ["medium"] / total["medium"])**2 / np.var(np.array(ratios["medium"])) / 2) ) * bilinear2(suc,0.5)
                else:
                    alpha = 1
                total["medium"] += 1 
                succ["medium"] += suc 
            elif item["difficulty"] == "hard":
                if len(ratios["hard"])>0:
                    alpha = (1 - np.exp(-(suc - succ["hard"] / total["hard"])**2 / np.var(np.array(ratios["hard"])) / 2)) * suc
                else:
                    alpha = 1
                total["hard"] += 1 
                succ["hard"] += suc 
            else:
                if len(ratios["medium"])>0:
                    alpha = (1 - np.exp(-(suc - succ["medium"] / total["medium"])**2 / np.var(np.array(ratios["medium"])) / 2)) * bilinear2(suc,0.5)
                else:
                    alpha = 1
                total["medium"] += 1 
                succ["medium"] += suc   

            alpha = np.sqrt(alpha)
            # alpha = alpha/2
            print(item["difficulty"],alpha)
            total_succ += suc * alpha
            total_weight += alpha
            score.append(total_succ/total_weight)
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
    # scores = scores/(1+scores)
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

    plt.savefig('2factor/' + batch_path.split("/")[-1] + ".png") 

    # print(np.mean(scores,axis=0)[49],np.mean(old_scores,axis=0)[49])
    print(np.mean(scores,axis=0)[-1],np.mean(old_scores,axis=0)[-1])
    # print(np.mean(scores,axis=0)[149],np.mean(old_scores,axis=0)[149])
    # print(np.mean(ratios["easy"],axis=0)[-1],np.mean(ratios["medium"],axis=0)[-1],np.mean(ratios["hard"],axis=0)[-1])   

if __name__ == "__main__":
    for batch_path in batch_paths:
        main(batch_path)