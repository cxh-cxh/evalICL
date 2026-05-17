import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
import seaborn as sns

sns.set_theme("notebook", style="whitegrid", palette="pastel")
RED = "#8B0000"
BLUE = ["#0BB4FF", "#0077B6", "#023E8A"]
GREY = "#8E8E93"
plt.rcParams["axes.titlecolor"] = "black"
plt.rcParams["axes.labelcolor"] = "#414246"
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["xtick.labelsize"] = 13
plt.rcParams["ytick.labelsize"] = 13
plt.rcParams["xtick.color"] = "#414246"
plt.rcParams["ytick.color"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["legend.labelcolor"] = "black"
plt.rcParams["font.family"].insert(0, "WenQuanYi Micro Hei")
plt.rcParams["axes.unicode_minus"] = False


top_label = [
    "加入过于简单的评测样例",
    "加入过于困难的评测样例",
]

fig_name = "t10003_poison.png"

func_names = ["linear", "mult_sqrt", "l2"]

func_name_dict = {"linear": "分段线性", "mult_sqrt": "几何平均", "l2": "L2距离"}

data_ = {}

for func_name in func_names:
    with open(f"./collate/result_{func_name}.json") as f:
        data_[func_name] = json.load(f)

print(data_[func_names[0]].keys())

keys = ["poison_easy", "poison_hard"]

data = np.array(
    [
        [
            data_[func_names[0]][keys[0]]["old"][-21],
            data_[func_names[0]][keys[0]]["old"][-1],
        ]
    ]
    + [
        [data_[func_name][keys[0]]["new"][-21], data_[func_name][keys[0]]["new"][-1]]
        for func_name in func_names
    ]
    + [
        [
            data_[func_names[0]][keys[1]]["old"][-21],
            data_[func_names[0]][keys[1]]["old"][-1],
        ]
    ]
    + [
        [data_[func_name][keys[1]]["new"][-21], data_[func_name][keys[1]]["new"][-1]]
        for func_name in func_names
    ],
)

ticks = (["传统流程"] + [func_name_dict[func_name] for func_name in func_names]) * 2

width = 1
count = len(data[0])

y_lim_high = [0.25, 0.5]
y_lim_low = [0, 0.1]


fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(18, 8),
    gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
)

ax1.set_title("堆叠方块任务加入异常数据实验结果", fontweight="bold", size=20)

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
    edgecolor="black",
)
ax2.bar(
    [x * width for x in range(2)],
    data[0],
    width=width,
    color=GREY,
    edgecolor="black",
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
    [1 * width],
    data[0][1],
    width=width,
    color="none",
    edgecolor="black",
)


for i in range(len(func_names)):
    ax1.bar(
        [(x + 2 * (i + 1)) * width + i + 1 for x in range(2)],
        data[i + 1],
        width=width,
        color=BLUE[i],
        edgecolor="black",
    )
    ax2.bar(
        [(x + 2 * (i + 1)) * width + i + 1 for x in range(2)],
        data[i + 1],
        width=width,
        color=BLUE[i],
        edgecolor="black",
    )
    ax1.bar(
        [(1 + 2 * (i + 1)) * width + i + 1],
        data[i + 1][1] - data[i + 1][0],
        bottom=data[i + 1][0],
        width=width,
        color="white",
        fill=False,
        hatch="///",
        linewidth=1.2,
    )
    ax1.bar(
        [(1 + 2 * (i + 1)) * width + i + 1],
        data[i + 1][1],
        width=width,
        color="none",
        edgecolor="black",
    )

ax1.bar(
    [(x + 2 * len(func_names) + 2) * width + len(func_names) + 1 for x in range(2)],
    data[1 + len(func_names)],
    width=width,
    color=GREY,
    edgecolor="black",
)
ax2.bar(
    [(x + 2 * len(func_names) + 2) * width + len(func_names) + 1 for x in range(2)],
    data[1 + len(func_names)],
    width=width,
    color=GREY,
    edgecolor="black",
)
ax1.bar(
    [(1 + 2 * len(func_names) + 2) * width + len(func_names) + 1],
    data[1 + len(func_names)][0] - data[1 + len(func_names)][1],
    bottom=data[1 + len(func_names)][1],
    width=width,
    color=GREY,
    hatch="///",
    fill=False,
    edgecolor=GREY,
    linewidth=1.2,
)
ax1.bar(
    [(1 + 2 * len(func_names) + 2) * width + len(func_names) + 1],
    data[1 + len(func_names)][0],
    width=width,
    color="none",
    edgecolor="black",
)

for i in range(len(func_names)):
    ax1.bar(
        [
            (x + 2 * (len(func_names) + i + 2)) * width + i + len(func_names) + 2
            for x in range(2)
        ],
        data[2 + len(func_names) + i],
        width=width,
        color=BLUE[i],
        edgecolor="black",
    )
    ax2.bar(
        [
            (x + 2 * (len(func_names) + i + 2)) * width + i + len(func_names) + 2
            for x in range(2)
        ],
        data[2 + len(func_names) + i],
        width=width,
        color=BLUE[i],
        edgecolor="black",
    )
    ax1.bar(
        [(1 + 2 * (len(func_names) + i + 2)) * width + i + len(func_names) + 2],
        data[2 + len(func_names) + i][0] - data[2 + len(func_names) + i][1],
        bottom=data[2 + len(func_names) + i][1],
        width=width,
        color=BLUE[i],
        hatch="///",
        fill=False,
        edgecolor=BLUE[i],
        linewidth=1.2,
    )
    ax1.bar(
        [(1 + 2 * (len(func_names) + i + 2)) * width + i + len(func_names) + 2],
        data[2 + len(func_names) + i][0],
        width=width,
        color="none",
        edgecolor="black",
    )
ax1.text(
    6,
    0.47,
    top_label[0],
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=15,
    color="black",
)
ax1.text(
    16,
    0.47,
    top_label[1],
    horizontalalignment="center",
    verticalalignment="bottom",
    fontsize=15,
    color="black",
)

line = lines.Line2D([11, 11], [0, 0.5], color=GREY, ls="--")
ax1.add_line(line)
ax1.set_xticks([0.5 + 3 * i for i in range(8)], ticks)

line = lines.Line2D([11, 11], [0, 0.5], color=GREY, ls="--")
ax2.add_line(line)

plt.tight_layout()
plt.savefig(fig_name, bbox_inches="tight", dpi=200)

for i in range(len(func_names)):
    print(func_names[i])
    print(
        "old:",
        f"{data[0][1] - data[0][0]:.3f}",
        "new:",
        f"{data[i+1][1] - data[i+1][0]:.3f}",
        "\n",
        "old:",
        f"{data[len(func_names)+1][0] - data[len(func_names)+1][1]:.3f}",
        "new:",
        f"{data[i+len(func_names)+2][0] - data[i+len(func_names)+2][1]:.3f}",
    )

    print(
        "old:",
        f"{(data[0][1] - data[0][0]) / data[0][0]:.2%}",
        "new:",
        f"{(data[i+1][1] - data[i+1][0]) / data[i+1][0]:.2%}",
        "\n",
        "old:",
        f"{(data[len(func_names)+1][0] - data[len(func_names)+1][1]) / data[len(func_names)+1][0]:.2%}",
        "new:",
        f"{(data[i+len(func_names)+2][0] - data[i+len(func_names)+2][1]) / data[i+len(func_names)+2][0]:.2%}",
    )
