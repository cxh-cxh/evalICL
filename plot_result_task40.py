import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
from plot_factory import broken_bar, savefig

RED = "#8B0000"
BLUE = ["#0BB4FF", "#0077B6", "#023E8A"]
GREY = "#8E8E93"

# plt.rcParams['figure.facecolor'] = 'white'

name_transform = {
    "pi0_t10003_mixed": "Pick and Place",
    "pi0_t7_full": "Cube in Cup",
}


top_label = [
    "环境\n1+2",
    "环境\n1+3",
    "环境\n2+3",
]
fig_name = "task40.png"

func_names = ["linear", "mult_sqrt", "l2"]

func_name_dict = {"linear": "分段线性", "mult_sqrt": "几何平均", "l2": "L2距离"}

data_ = {}

for func_name in func_names:
    with open(f"./collate/result_{func_name}.json") as f:
        data_[func_name] = json.load(f)

print(data_[func_names[0]].keys())

keys = ["pi05_task40_12", "pi05_task40_13", "pi05_task40_23"]

ref_key = "pi05_task40_full"

data = np.array(
    [[data_[func_names[0]][key]["old"][-1] for key in keys]]
    + [[data_[func_name][key]["new"][-1] for key in keys] for func_name in func_names]
)
ref = np.array(
    [data_[func_names[0]][ref_key]["old"][-1]]
    + [data_[func_name][ref_key]["new"][-1] for func_name in func_names]
)

ticks = ["传统流程"] + [func_name_dict[func_name] for func_name in func_names]


width = 1
count = len(data[0])
y_lim_high = [0.43, 0.54]
y_lim_low = [0, 0.1]

ax1, ax2 = broken_bar(y_lim_high, y_lim_low)
ax1.set_title(
    "移动物体任务跨环境实验结果",
    fontweight="bold",
    size="x-large",
)

ax1.bar(
    [width * x for x in range(count)],
    data[0],
    width=width,
    color=GREY,
    edgecolor="black",
)
ax2.bar(
    [width * x for x in range(count)],
    data[0],
    width=width,
    color=GREY,
    edgecolor="black",
)
diff0 = data[0] - ref[0]

for i in range(count):
    ax1.text(
        width * i,
        np.maximum(data[0][i], ref[0]),
        top_label[i],
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=15,
        color=GREY,
    )

    if diff0[i] < 0:
        ax1.bar(
            [width * i],
            height=-diff0[i],
            bottom=data[0][i],
            width=width,
            color=GREY,
            hatch="///",
            fill=False,
            edgecolor=GREY,
            linewidth=1.2,
        )
    else:
        ax1.bar(
            [width * i],
            height=diff0[i],
            bottom=ref[0],
            width=width,
            color="white",
            fill=False,
            hatch="///",
            linewidth=1.2,
        )
    ax1.bar(
        [width * i],
        height=np.maximum(data[0][i], ref[0]),
        width=width,
        color="none",
        edgecolor="black",
    )

line0 = lines.Line2D([-1, count * width], [ref[0], ref[0]], color=GREY, ls="--")

diff = {}
for i in range(len(func_names)):

    ax1.bar(
        [width * x + i + 1 + count * width * (i + 1) for x in range(count)],
        data[i + 1],
        width=width,
        color=BLUE[i],
        edgecolor="black",
    )
    ax2.bar(
        [width * x + i + 1 + count * width * (i + 1) for x in range(count)],
        data[i + 1],
        width=width,
        color=BLUE[i],
        edgecolor="black",
    )

    diff[func_names[i]] = data[i + 1] - ref[i + 1]

    for j in range(count):
        ax1.text(
            width * j + i + 1 + count * width * (i + 1),
            np.maximum(data[i + 1][j], ref[i + 1]),
            top_label[j],
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=15,
            color=BLUE[i],
        )
        if diff[func_names[i]][j] < 0:
            ax1.bar(
                [width * j + i + 1 + count * width * (i + 1)],
                height=-diff[func_names[i]][j],
                bottom=data[i + 1][j],
                width=width,
                color=BLUE[i],
                hatch="///",
                fill=False,
                edgecolor=BLUE[i],
                linewidth=1.2,
            )
        else:
            ax1.bar(
                [width * j + i + 1 + count * width * (i + 1)],
                height=diff[func_names[i]][j],
                bottom=ref[i + 1],
                width=width,
                color="white",
                fill=False,
                hatch="///",
                linewidth=1.2,
            )
        ax1.bar(
            [width * j + i + 1 + count * width * (i + 1)],
            height=np.maximum(data[i + 1][j], ref[i + 1]),
            width=width,
            color="none",
            edgecolor="black",
        )
    line = lines.Line2D(
        [count * width * (i + 1) + i, count * width * (i + 2) + i + 1],
        [ref[i + 1], ref[i + 1]],
        color=BLUE[i],
        ls="--",
    )
    ax1.add_line(line)

ax1.add_line(line0)
ax1.set_xticks([4 * width * x + 1 for x in range(4)], ticks)

savefig(fig_name)
for i in range(len(func_names)):
    print(func_names[i])
    print(
        "old:",
        f"{np.mean(np.abs(diff0)):.3f}",
        "new:",
        f"{np.mean(np.abs(diff[func_names[i]])):.3f}",
    )
    print(
        "old:",
        f"{np.mean(np.abs(diff0)) / ref[0]:.2%}",
        "new:",
        f"{np.mean(np.abs(diff[func_names[i]])) / ref[i + 1]:.2%}",
    )
