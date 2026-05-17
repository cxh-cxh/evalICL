import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
import seaborn as sns
from scipy import stats

sns.set_theme("notebook", style="ticks", palette="pastel")
plt.rcParams["axes.titlecolor"] = "black"
plt.rcParams["axes.labelcolor"] = "#414246"
plt.rcParams["xtick.color"] = "#414246"
plt.rcParams["ytick.color"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["legend.labelcolor"] = "black"

# plt.rcParams['figure.facecolor'] = 'white'

name_transform = {
    "pi0_t10003_no_train_data": "Pick and Place",
    "pi0_t7_no_train_data": "Cube in Cup",
    "pi05_task40_no_train_data": "ALOHA Transfer",
    "pi0_t10_no_train_data": "Puzzle",
    "pi0_t10003_sim_no_train_data": "Pick and Place Sim",
    "pi0_drawer_no_train_data": "Drawer",
}

tasks = [
    "pi0_t10003_no_train_data",
    "pi0_t10003_sim_no_train_data",
    "pi0_t7_no_train_data",
    "pi0_t10_no_train_data",
    "pi05_task40_no_train_data",
]

with open("./collate/difficulty.json") as f:
    data_ = json.load(f)


data = [
    [np.mean(data_[task]["easy"]) for task in tasks],
    [np.mean(data_[task]["medium"]) for task in tasks],
    [np.mean(data_[task]["hard"]) for task in tasks],
]


ticks = [name_transform[x] for x in tasks]


width = 1
count = len(data[0])

fig, ax = plt.subplots(figsize=(10, 3))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.bar(
    [4 * width * x for x in range(count)],
    data[0],
    width=width,
    color="#50D050",
    label="easy",
)
ax.bar(
    [4 * width * x + 1 for x in range(count)],
    data[1],
    width=width,
    color="#FFEC00",
    label="medium",
)
ax.bar(
    [4 * width * x + 2 for x in range(count)],
    data[2],
    width=width,
    color="#E61D5C",
    label="hard",
)
ax.set_xticks([4 * width * x + 1 for x in range(count)], ticks)


ax.set_title(
    "Average Posterior Success / Progress Rate by Predicted Difficulty",
    fontweight="bold",
)
ax.legend()
plt.tight_layout()
plt.savefig("difficulty.png", dpi=200)

for i in range(len(tasks)):
    f_stat, p_value = stats.f_oneway(
        data_[tasks[i]]["easy"], data_[tasks[i]]["medium"], data_[tasks[i]]["hard"]
    )
    print(
        name_transform[tasks[i]],
        f"\nScipy F-value: {f_stat:.2f}, p-value: {p_value:.6f}",
    )
    k = 3
    N = (
        len(data_[tasks[i]]["easy"])
        + len(data_[tasks[i]]["medium"])
        + len(data_[tasks[i]]["hard"])
    )
    df_between = k - 1
    df_within = N - k

    eta_squared_from_f = (f_stat * df_between) / (f_stat * df_between + df_within)
    print(f"从 F 值反推 η² = {eta_squared_from_f:.4f}")
