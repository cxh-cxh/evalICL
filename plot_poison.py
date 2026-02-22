import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
import seaborn as sns

sns.set_theme("notebook", style="ticks", palette="pastel")
RED = "#8B0000"
GREY = "#414246"
plt.rcParams["axes.titlecolor"] = "#8B0000"
plt.rcParams["axes.labelcolor"] = "#414246"
plt.rcParams["xtick.color"] = "#414246"
plt.rcParams["ytick.color"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["legend.labelcolor"] = "#8B0000"

# plt.rcParams['figure.facecolor'] = 'white'


top_label = [
    "Inject over-easy test cases",
    "Inject over-hard test cases",
]

fig_name = "t10003_poison.png"

with open("./collate/result.json") as f:
    data_ = json.load(f)

print(data_.keys())


ref_key = "pi05_task40_full"

data = np.array(
    [
        [data_["poison_easy"]["old"][-21], data_["poison_easy"]["old"][-1]],
        [data_["poison_easy"]["new"][-21], data_["poison_easy"]["new"][-1]],
        [data_["poison_hard"]["old"][-21], data_["poison_hard"]["old"][-1]],
        [data_["poison_hard"]["new"][-21], data_["poison_hard"]["new"][-1]],
    ]
)

ticks = ["common", "DARE", "common", "DARE"]


width = 1
count = len(data[0])

fig, ax = plt.subplots()
ax.set_title(
    "Pick-and-Place Task Scores with Poisoned Data",
    fontweight="bold",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.bar(
    [x * width for x in range(2)],
    data[0],
    width=width,
    color=GREY,
)
ax.bar(
    [1 * width],
    data[0][1] - data[0][0],
    bottom=data[0][0],
    width=width,
    color="white",
    fill=False,
    hatch="///",
    linewidth=1.2,
)

ax.bar(
    [(x + 2) * width + 1 for x in range(2)],
    data[1],
    width=width,
    color=RED,
)

ax.bar(
    [3 * width + 1],
    data[1][1] - data[1][0],
    bottom=data[1][0],
    width=width,
    color="white",
    fill=False,
    hatch="///",
    linewidth=1.2,
)

ax.bar(
    [(x + 4) * width + 2 for x in range(2)],
    data[2],
    width=width,
    color=GREY,
)

ax.bar(
    [5 * width + 2],
    data[2][0] - data[2][1],
    bottom=data[2][1] - 0.002,
    width=width - 0.07,
    color=GREY,
    hatch="///",
    fill=False,
    edgecolor=GREY,
    linewidth=1.2,
)


ax.bar(
    [(x + 6) * width + 3 for x in range(2)],
    data[3],
    width=width,
    color=RED,
)
ax.bar(
    [7 * width + 3],
    data[3][0] - data[3][1],
    bottom=data[3][1] - 0.002,
    width=width - 0.07,
    color=RED,
    hatch="///",
    fill=False,
    edgecolor=RED,
    linewidth=1.2,
)

ax.set_ylim(0, 0.5)
ax.text(
    2,
    0.47,
    top_label[0],
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=12,
    color=RED,
)
ax.text(
    8,
    0.47,
    top_label[1],
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=12,
    color=RED,
)

line = lines.Line2D([5, 5], [0, 0.5], color=GREY, ls="--")
ax.add_line(line)
ax.set_xticks([0.5, 3.5, 6.5, 9.5], ticks)


# ax.set_title(
# "Posterior Success / Progress Rate by Predicted Difficulty", fontweight="bold"
# )
# ax.legend()

plt.tight_layout()

plt.savefig(fig_name, dpi=200)
print(
    data[0][1] - data[0][0],
    data[1][1] - data[1][0],
    data[2][0] - data[2][1],
    data[3][0] - data[3][1],
)
