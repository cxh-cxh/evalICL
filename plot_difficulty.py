import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
import seaborn as sns

sns.set_theme("notebook", style="ticks", palette="pastel")
plt.rcParams["axes.titlecolor"] = "#8B0000"
plt.rcParams["axes.labelcolor"] = "#414246"
plt.rcParams["xtick.color"] = "#414246"
plt.rcParams["ytick.color"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["legend.labelcolor"] = "#8B0000"

# plt.rcParams['figure.facecolor'] = 'white'

name_transform = {
    "pi0_t10003_full": "Pick and Place",
    "pi0_t7_full": "Cube in Cup",
    "pi05_task40_full": "ALOHA Transfer",
    "pi0_t10_full": "Puzzle",
    "pi0_t10003_sim_full": "Pick and Place Sim",
}


with open("./collate/difficulty.json") as f:
    data_ = json.load(f)

data = [
    [x["easy"] for key, x in data_.items()],
    [x["medium"] for key, x in data_.items()],
    [x["hard"] for key, x in data_.items()],
]

ticks = [name_transform[x] for x in data_.keys()]


width = 1
count = len(data[0])

fig, ax = plt.subplots(figsize=(10, 3))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.bar(
    [4 * width * x for x in range(count)],
    data[0],
    width=width,
    color="green",
    label="easy",
)
ax.bar(
    [4 * width * x + 1 for x in range(count)],
    data[1],
    width=width,
    color="orange",
    label="medium",
)
ax.bar(
    [4 * width * x + 2 for x in range(count)],
    data[2],
    width=width,
    color="red",
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
