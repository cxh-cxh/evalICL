import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
import seaborn as sns

sns.set_theme("notebook", style="darkgrid", palette="pastel")


def get_subplots(num):
    fig, axs = plt.subplots(num, figsize=(10, 4 * num))
    return axs


def difficulty_plot(ax, data):
    easy = data["easy"]
    medium = data["medium"]
    hard = data["hard"]
    ax.set_ylim(0, 1)
    ax.set_xlim(0, ratios["easy"].shape[1])
    ax.set_title("Difficulty Prediction")
    ax.plot(np.mean(ratios["easy"], axis=0), color="green", label="Easy")
    ax.fill_between(
        range(ratios["easy"].shape[1]),
        np.mean(ratios["easy"], axis=0) - np.std(ratios["easy"], axis=0),
        np.mean(ratios["easy"], axis=0) + np.std(ratios["easy"], axis=0),
        alpha=0.2,
        color="green",
    )
    ax.plot(np.mean(ratios["medium"], axis=0), color="yellow", label="Medium")
    ax.fill_between(
        range(ratios["medium"].shape[1]),
        np.mean(ratios["medium"], axis=0) - np.std(ratios["medium"], axis=0),
        np.mean(ratios["medium"], axis=0) + np.std(ratios["medium"], axis=0),
        alpha=0.2,
        color="yellow",
    )
    ax.plot(np.mean(ratios["hard"], axis=0), color="red", label="Hard")
    ax.fill_between(
        range(ratios["hard"].shape[1]),
        np.mean(ratios["hard"], axis=0) - np.std(ratios["hard"], axis=0),
        np.mean(ratios["hard"], axis=0) + np.std(ratios["hard"], axis=0),
        alpha=0.2,
        color="red",
    )
    ax.legend()
