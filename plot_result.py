import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
from plot_factory import broken_bar

RED = "#8B0000"
BLUE = "#0BB4FF"
GREY = "#8E8E93"


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

fig_name = "t10003.png"

with open("./collate/result.json") as f:
    data_ = json.load(f)

print(data_.keys())

keys = ["pi0_t10003_env_12", "pi0_t10003_env_13", "pi0_t10003_env_23"]

ref_key = "pi0_t10003_full"

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
y_lim_high = [0.3, 0.5]
y_lim_low = [0, 0.1]

ax1, ax2 = broken_bar(y_lim_high, y_lim_low)
ax1.set_title(
    "Pick and Place Task Scores",
    fontweight="bold",
    size="x-large",
)

ax1.bar(
    [width * x for x in range(count)],
    data[0],
    width=width,
    color=GREY,
)
ax2.bar(
    [width * x for x in range(count)],
    data[0],
    width=width,
    color=GREY,
)
diff0 = data[0] - ref[0]

for i in range(count):
    ax1.text(
        width * i,
        np.maximum(data[0][i], ref[0]),
        top_label[i],
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=9,
        color=GREY,
    )

    if diff0[i] < 0:
        ax1.bar(
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

line0 = lines.Line2D([-1, count * width], [ref[0], ref[0]], color=GREY, ls="--")

ax1.bar(
    [width * x + 1 + count * width for x in range(count)],
    data[1],
    width=width,
    color=BLUE,
)
ax2.bar(
    [width * x + 1 + count * width for x in range(count)],
    data[1],
    width=width,
    color=BLUE,
)

diff1 = data[1] - ref[1]

for i in range(count):
    ax1.text(
        width * i + 1 + count * width,
        np.maximum(data[1][i], ref[1]),
        top_label[i],
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=9,
        color=BLUE,
    )
    if diff1[i] < 0:
        ax1.bar(
            [width * i + 1 + count * width],
            height=-diff1[i],
            bottom=data[1][i],
            width=width - 0.05,
            color=BLUE,
            hatch="///",
            fill=False,
            edgecolor=BLUE,
            linewidth=1.2,
        )
    else:
        ax1.bar(
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
    [count * width, 2 * count * width + 1], [ref[1], ref[1]], color=BLUE, ls="--"
)

ax1.add_line(line0)
ax1.add_line(line1)
ax1.set_xticks([4 * width * x + 1 for x in range(2)], ticks)


# ax.set_title(
# "Posterior Success / Progress Rate by Predicted Difficulty", fontweight="bold"
# )
# ax.legend()

plt.tight_layout()

plt.savefig(fig_name, dpi=200)
print(np.mean(np.abs(diff0)), np.mean(np.abs(diff1)))
