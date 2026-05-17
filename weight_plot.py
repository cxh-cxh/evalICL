import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
import seaborn as sns

sns.set_theme("notebook", style="whitegrid", palette="pastel")
plt.rcParams["axes.titlecolor"] = "#8B0000"
plt.rcParams["axes.labelcolor"] = "#414246"
plt.rcParams["xtick.color"] = "#414246"
plt.rcParams["ytick.color"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["axes.edgecolor"] = "#414246"
plt.rcParams["legend.labelcolor"] = "#8B0000"
plt.rcParams["font.family"].insert(0, "WenQuanYi Micro Hei")
plt.rcParams["axes.unicode_minus"] = False


def bilinear(x, mid):
    return np.where(x < mid, 0.5 / mid * x, 0.5 / (1 - mid) * (x - mid) + 0.5)


def bilinear2(x, mid):
    return np.where(x < mid, 1 / mid * (mid - x), 1 / (1 - mid) * (x - mid))


weight_func = {
    "linear": {
        "easy": (lambda a, b: 1 - bilinear(a, b)),
        "medium": (lambda a, b: np.array([0.5] * len(a))),
        "hard": (lambda a, b: bilinear(a, b)),
    },
    "mult": {
        "easy": (lambda a, b: bilinear2(a, b) * (1 - a)),
        "medium": (lambda a, b: bilinear2(a, b) * bilinear2(a, 0.5)),
        "hard": (lambda a, b: bilinear2(a, b) * a),
    },
    "mult_sqrt": {
        "easy": (lambda a, b: np.sqrt(bilinear2(a, b) * (1 - a))),
        "medium": (lambda a, b: np.sqrt(bilinear2(a, b) * bilinear2(a, 0.5))),
        "hard": (lambda a, b: np.sqrt(bilinear2(a, b) * a)),
    },
    "l2": {
        "easy": (lambda a, b: np.sqrt(((a - b) ** 2 + (1 - a) ** 2) / 2)),
        "medium": (lambda a, b: np.sqrt(((a - b) ** 2 + (0.5 - a) ** 2) / 2)),
        "hard": (lambda a, b: np.sqrt(((a - b) ** 2 + a**2) / 2)),
    },
    "gaussian": {
        "easy": (
            lambda a, b: 1 / (np.exp(-((a - b) ** 2)) + np.exp(-((1 - a) ** 2))) - 0.5
        ),
        "medium": (
            lambda a, b: 1 / (np.exp(-((a - b) ** 2)) + np.exp(-((0.5 - a) ** 2))) - 0.5
        ),
        "hard": (lambda a, b: 1 / (np.exp(-((a - b) ** 2)) + np.exp(-(a**2))) - 0.5),
    },
}

func_name = "l2"

easy = 0.7
medium = 0.4
hard = 0.2

x = np.linspace(0, 1, 1000)

fig, axs = plt.subplots(1, 3, figsize=(15, 6))
axs[0].plot(x, weight_func[func_name]["easy"](x, easy), color="green", lw=2)
# linex = lines.Line2D([easy, easy], [0, 0.5], color="black", ls="--")
# liney = lines.Line2D([0, easy], [0.5, 0.5], color="black", ls="--")
# axs[0].add_line(linex)
# axs[0].add_line(liney)
axs[0].set_title("简单评测结果权重")
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1)
axs[0].set_xlabel("$\hat{sr}$")
axs[0].set_ylabel("$\\alpha(M_i)$")
axs[0].set_xticks(list(axs[0].get_xticks()) + [easy])
labels = axs[0].get_xticklabels()
labels[-1] = "$\hat{sr}^e_i$"
axs[0].set_xticklabels(labels)
# axs[0].set_yticks(list(axs[0].get_yticks()) + [0.5])

# axs[1].plot(x,bilinear2(x, medium) / 2 + 0.25 ,color = "orange")
axs[1].plot(x, weight_func[func_name]["medium"](x, medium), color="orange", lw=2)
axs[1].set_title("中等评测结果权重")
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 1)
axs[1].set_xlabel("$\hat{sr}$")
axs[1].set_ylabel("$\\alpha(M_i)$")
axs[1].set_xticks(list(axs[1].get_xticks()) + [medium])
labels = axs[1].get_xticklabels()
labels[-1] = "$\hat{sr}^m_i$"
axs[1].set_xticklabels(labels)
# axs[1].set_yticks(list(axs[0].get_yticks()) + [0.5])

axs[2].plot(x, weight_func[func_name]["hard"](x, hard), color="red", lw=2)
# linex = lines.Line2D([hard, hard], [0, 0.5], color="black", ls="--")
# liney = lines.Line2D([0, hard], [0.5, 0.5], color="black", ls="--")
# axs[2].add_line(linex)
# axs[2].add_line(liney)
axs[2].set_title("困难评测结果权重")
axs[2].set_xlim(0, 1)
axs[2].set_ylim(0, 1)
axs[2].set_xlabel("$\hat{sr}$")
axs[2].set_ylabel("$\\alpha(M_i)$")
axs[2].set_xticks(list(axs[2].get_xticks()) + [hard])
labels = axs[2].get_xticklabels()
labels[-1] = "$\hat{sr}^h_i$"
axs[2].set_xticklabels(labels)
# axs[2].set_yticks(list(axs[0].get_yticks()) + [0.5])

plt.tight_layout()
plt.savefig("weight.png")
