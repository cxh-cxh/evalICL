import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
import seaborn as sns

sns.set_theme("notebook", style="ticks", palette="pastel")
RED = "#8B0000"
BLUE = "#0BB4FF"
GREY = "#8E8E93"
plt.rcParams["axes.titlecolor"] = "black"
plt.rcParams["axes.labelcolor"] = "#414246"
plt.rcParams["xtick.color"] = "#414246"
plt.rcParams["ytick.color"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["legend.labelcolor"] = "black"

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

y_lim_high = [0.25, 0.5]
y_lim_low = [0, 0.1]


fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(12, 8),
    gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
)

ax1.set_title(
    "Pick-and-Place Task Scores with Poisoned Data", fontweight="bold", size="x-large"
)

ax1.set_ylim(0.25, 0.5)
ax2.set_ylim(0, 0.1)

ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.spines["bottom"].set_visible(False)
ax1.tick_params(labeltop=False)
ax2.tick_params(labeltop=False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


ax2.set_yticks(y_lim_low)

d = 0.015  # 斜线大小
kwargs = dict(transform=ax1.transAxes, color="black", clip_on=False, linewidth=1.5)
ax1.plot((-d, +d), (-d, +d), **kwargs)  # 左上斜线
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右上斜线

kwargs.update(transform=ax2.transAxes)  # 切换到ax2的坐标系
ax2.plot((-d, +d), (1 - d * 4, 1 + d * 4), **kwargs)  # 左下斜线
ax2.plot((1 - d, 1 + d), (1 - d * 4, 1 + d * 4), **kwargs)  # 右下斜线

ax1.bar(
    [x * width for x in range(2)],
    data[0],
    width=width,
    color=GREY,
)
ax2.bar(
    [x * width for x in range(2)],
    data[0],
    width=width,
    color=GREY,
)
ax1.bar(
    [1 * width],
    data[0][1] - data[0][0],
    bottom=data[0][0],
    width=width,
    color="white",
    fill=False,
    hatch="///",
    linewidth=1.2,
)

ax1.bar(
    [(x + 2) * width + 1 for x in range(2)],
    data[1],
    width=width,
    color=BLUE,
)
ax2.bar(
    [(x + 2) * width + 1 for x in range(2)],
    data[1],
    width=width,
    color=BLUE,
)
ax1.bar(
    [3 * width + 1],
    data[1][1] - data[1][0],
    bottom=data[1][0],
    width=width,
    color="white",
    fill=False,
    hatch="///",
    linewidth=1.2,
)

ax1.bar(
    [(x + 4) * width + 2 for x in range(2)],
    data[2],
    width=width,
    color=GREY,
)
ax2.bar(
    [(x + 4) * width + 2 for x in range(2)],
    data[2],
    width=width,
    color=GREY,
)
ax1.bar(
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


ax1.bar(
    [(x + 6) * width + 3 for x in range(2)],
    data[3],
    width=width,
    color=BLUE,
)
ax2.bar(
    [(x + 6) * width + 3 for x in range(2)],
    data[3],
    width=width,
    color=BLUE,
)
ax1.bar(
    [7 * width + 3],
    data[3][0] - data[3][1],
    bottom=data[3][1] - 0.002,
    width=width - 0.07,
    color=BLUE,
    hatch="///",
    fill=False,
    edgecolor=BLUE,
    linewidth=1.2,
)

ax1.text(
    2,
    0.47,
    top_label[0],
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=12,
    color="black",
)
ax1.text(
    8,
    0.47,
    top_label[1],
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=12,
    color="black",
)

line = lines.Line2D([5, 5], [0, 0.5], color=GREY, ls="--")
ax1.add_line(line)
ax1.set_xticks([0.5, 3.5, 6.5, 9.5], ticks)

line = lines.Line2D([5, 5], [0, 0.5], color=GREY, ls="--")
ax2.add_line(line)

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
