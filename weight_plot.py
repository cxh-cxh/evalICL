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


def bilinear(x, mid):
    return np.where(x < mid, 0.5 / mid * x, 0.5 / (1 - mid) * (x - mid) + 0.5)


def bilinear2(x, mid):
    return np.where(x < mid, 1 / mid * (mid - x), 1 / (1 - mid) * (x - mid))


easy = 0.7
medium = 0.4
hard = 0.2

x = np.linspace(0, 1, 100)

fig, axs = plt.subplots(1, 3, figsize=(15, 6))
axs[0].plot(x, 1 - bilinear(x, easy), color="green", lw=2)
linex = lines.Line2D([easy, easy], [0, 0.5], color="black", ls="--")
liney = lines.Line2D([0, easy], [0.5, 0.5], color="black", ls="--")
axs[0].add_line(linex)
axs[0].add_line(liney)
axs[0].set_title("Weight for Easy Cases")
axs[0].set_xlim(0, 1)
axs[0].set_ylim(0, 1)
axs[0].set_xlabel("$s_k$")
axs[0].set_ylabel("$\\alpha(i_k)$")
axs[0].set_xticks(list(axs[0].get_xticks()) + [easy])
labels = axs[0].get_xticklabels()
labels[-1] = "$s^e_k$"
axs[0].set_xticklabels(labels)
axs[0].set_yticks(list(axs[0].get_yticks()) + [0.5])

# axs[1].plot(x,bilinear2(x, medium) / 2 + 0.25 ,color = "orange")
axs[1].plot(x, [0.5] * len(x), color="orange", lw=2)
axs[1].set_title("Weight for Medium Cases")
axs[1].set_xlim(0, 1)
axs[1].set_ylim(0, 1)
axs[1].set_xlabel("$s_k$")
axs[1].set_ylabel("$\\alpha(i_k)$")
axs[1].set_xticks(list(axs[1].get_xticks()) + [medium])
labels = axs[1].get_xticklabels()
labels[-1] = "$s^m_k$"
axs[1].set_xticklabels(labels)
axs[1].set_yticks(list(axs[0].get_yticks()) + [0.5])

axs[2].plot(x, bilinear(x, hard), color="red", lw=2)
linex = lines.Line2D([hard, hard], [0, 0.5], color="black", ls="--")
liney = lines.Line2D([0, hard], [0.5, 0.5], color="black", ls="--")
axs[2].add_line(linex)
axs[2].add_line(liney)
axs[2].set_title("Weight for Hard Cases")
axs[2].set_xlim(0, 1)
axs[2].set_ylim(0, 1)
axs[2].set_xlabel("$s_k$")
axs[2].set_ylabel("$\\alpha(i_k)$")
axs[2].set_xticks(list(axs[2].get_xticks()) + [hard])
labels = axs[2].get_xticklabels()
labels[-1] = "$s^h_k$"
axs[2].set_xticklabels(labels)
axs[2].set_yticks(list(axs[0].get_yticks()) + [0.5])

plt.tight_layout()
plt.savefig("weight.png")
