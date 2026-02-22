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

name_transform = {
    "pi0_t10003_mixed": "Pick and Place",
    "pi0_t7_full": "Cube in Cup",
}


top_label = [
    "Env 1+2",
    "Env 1+3",
    "Env 2+3",
]

fig_name = "task40.png"

with open("./collate/result.json") as f:
    data_ = json.load(f)

print(data_.keys())

keys = ["pi05_task40_12", "pi05_task40_13", "pi05_task40_23"]

ref_key = "pi05_task40_full"

data = np.array(
    [
        [data_[key]["old"][-1] for key in keys],
        [data_[key]["new"][-1] for key in keys],
    ]
)
ref = np.array([data_[ref_key]["old"][-1], data_[ref_key]["new"][-1]])

ticks = ["common", "DARE"]


width = 1
count = len(data[0])

fig, ax = plt.subplots()
ax.set_title(
    "ALOHA-Transfer Task Scores",
    fontweight="bold",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.bar(
    [width * x for x in range(count)],
    data[0],
    width=width,
    color=GREY,
)

diff0 = data[0] - ref[0]

for i in range(count):
    ax.text(
        width * i,
        np.maximum(data[0][i], ref[0]),
        top_label[i],
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=9,
        color=GREY,
    )

    if diff0[i] < 0:
        ax.bar(
            [width * i],
            height=-diff0[i],
            bottom=data[0][i],
            width=width - 0.05,
            color=GREY,
            hatch="///",
            fill=False,
            edgecolor=GREY,
            linewidth=1.2,
        )
    else:
        ax.bar(
            [width * i],
            height=diff0[i],
            bottom=ref[0],
            width=width,
            color="white",
            fill=False,
            hatch="///",
            linewidth=1.2,
        )

line0 = lines.Line2D([-1, count * width], [ref[0], ref[0]], color=GREY, ls="--")

ax.bar(
    [width * x + 1 + count * width for x in range(count)],
    data[1],
    width=width,
    color=RED,
)

diff1 = data[1] - ref[1]

for i in range(count):
    ax.text(
        width * i + 1 + count * width,
        np.maximum(data[1][i], ref[1]),
        top_label[i],
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=9,
        color=RED,
    )
    if diff1[i] < 0:
        ax.bar(
            [width * i + 1 + count * width],
            height=-diff1[i],
            bottom=data[1][i],
            width=width - 0.05,
            color=RED,
            hatch="///",
            fill=False,
            edgecolor=RED,
            linewidth=1.2,
        )
    else:
        ax.bar(
            [width * i + 1 + count * width],
            height=diff1[i],
            bottom=ref[1],
            width=width,
            color="white",
            fill=False,
            hatch="///",
            linewidth=1.2,
        )


line1 = lines.Line2D(
    [count * width, 2 * count * width + 1], [ref[1], ref[1]], color=RED, ls="--"
)

ax.add_line(line0)
ax.add_line(line1)
ax.set_xticks([4 * width * x + 1 for x in range(2)], ticks)


# ax.set_title(
# "Posterior Success / Progress Rate by Predicted Difficulty", fontweight="bold"
# )
# ax.legend()

plt.tight_layout()

plt.savefig(fig_name, dpi=200)
print(np.mean(np.abs(diff0)), np.mean(np.abs(diff1)))
