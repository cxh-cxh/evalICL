import json
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import os, glob
import numpy as np
import seaborn as sns

sns.set_theme("notebook", style="darkgrid", palette="pastel")


def rolling_variance(x, window_size, ddof=1):
    """
    计算序列在滑动窗口内的方差

    Parameters:
    -----------
    x : array_like
        输入序列
    window_size : int
        窗口长度
    ddof : int, default=1
        Delta Degrees of Freedom（样本方差用1，总体方差用0）

    Returns:
    --------
    var : ndarray
        方差序列，长度为 len(x) - window_size + 1
    """
    x = np.asarray(x)
    n = len(x)

    if window_size > n:
        raise ValueError(f"窗口大小 {window_size} 不能大于序列长度 {n}")

    # 方法1: 使用滑动窗口视图（NumPy 1.20+，直观易懂）
    if hasattr(np.lib.stride_tricks, "sliding_window_view"):
        windows = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=-1)
        return np.var(windows, axis=-1, ddof=ddof)

    # 方法2: 使用卷积（更高效，内存友好，适用于大数组）
    else:
        # 计算窗口内元素的和与平方和
        kernel = np.ones(window_size)
        sum_x = np.convolve(x, kernel, mode="valid")
        sum_x2 = np.convolve(x**2, kernel, mode="valid")

        # Var(X) = E[X^2] - (E[X])^2
        mean = sum_x / window_size
        variance = (sum_x2 / window_size) - mean**2

        # 样本方差修正（无偏估计）
        if ddof == 1:
            variance = variance * window_size / (window_size - 1)

        return variance


def get_subplots(num):
    fig, axs = plt.subplots(num, figsize=(10, 4 * num))
    return axs


def difficulty_plot(ax: plt.Axes, data):
    easy = data["easy"]
    medium = data["medium"]
    hard = data["hard"]
    length = data["length"]
    ax.set_ylim(0, 1)
    ax.set_xlim(0, length)
    ax.set_title("Difficulty Prediction")
    ax.plot(np.mean(easy, axis=0), color="green", label="Easy")
    ax.fill_between(
        range(length),
        np.mean(easy, axis=0) - np.std(easy, axis=0),
        np.mean(easy, axis=0) + np.std(easy, axis=0),
        alpha=0.2,
        color="green",
    )
    ax.plot(np.mean(medium, axis=0), color="yellow", label="Medium")
    ax.fill_between(
        range(length),
        np.mean(medium, axis=0) - np.std(medium, axis=0),
        np.mean(medium, axis=0) + np.std(medium, axis=0),
        alpha=0.2,
        color="yellow",
    )
    ax.plot(np.mean(hard, axis=0), color="red", label="Hard")
    ax.fill_between(
        range(length),
        np.mean(hard, axis=0) - np.std(hard, axis=0),
        np.mean(hard, axis=0) + np.std(hard, axis=0),
        alpha=0.2,
        color="red",
    )
    ax.legend()


def score_plot(ax: plt.Axes, data):
    length = data["length"]
    scores = data["scores"]
    old_scores = data["old_scores"]
    ax.set_ylim(0, 1)
    ax.set_xlim(0, length)
    ax.set_xlabel("test cases")
    ax.set_title("Final Score (Success Rate)")
    ax.plot(np.mean(scores, axis=0), label="DARE pipeline")
    ax.plot(np.mean(old_scores, axis=0), label="Common pipeline")
    ax.fill_between(
        range(length),
        np.mean(scores, axis=0) - np.std(scores, axis=0),
        np.mean(scores, axis=0) + np.std(scores, axis=0),
        alpha=0.2,
        label="",
    )
    ax.fill_between(
        range(length),
        np.mean(old_scores, axis=0) - np.std(old_scores, axis=0),
        np.mean(old_scores, axis=0) + np.std(old_scores, axis=0),
        alpha=0.2,
        label="",
    )
    ax.legend(loc="upper right")


def variance_plot(
    ax: plt.Axes,
    data,
):
    length = data["length"]
    scores = data["scores"]
    old_scores = data["old_scores"]
    ax.set_xlim(0, length)
    ax.set_xlabel("test cases")
    ax.set_title("Variance of Final Score")
    ax.plot(np.mean(rolling_variance(scores, 10), axis=0), label="DARE pipeline")
    # ax.plot((np.var(scores, axis=0)), label="DARE pipeline")
    ax.plot(np.mean(rolling_variance(old_scores, 10), axis=0), label="Common pipeline")
    # ax.plot((np.var(old_scores, axis=0)), label="Common pipeline")
    ax.set_yscale("log")
    ax.legend(loc="upper right")


def savefig(path):
    plt.tight_layout()
    plt.savefig(path)


def broken_bar(high_lim, low_lim):
    sns.set_theme("notebook", style="ticks", palette="pastel")
    RED = "#8B0000"
    GREEN = "#50D050"
    GREY = "#C6C7CC"
    plt.rcParams["axes.titlecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "#414246"
    plt.rcParams["xtick.color"] = "#414246"
    plt.rcParams["ytick.color"] = "#414246"
    plt.rcParams["axes.edgecolor"] = "#414246"
    plt.rcParams["axes.edgecolor"] = "#414246"
    plt.rcParams["legend.labelcolor"] = "black"
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.tick_params(labeltop=False)
    ax2.tick_params(labeltop=False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_yticks(low_lim)
    ax1.set_ylim(*high_lim)
    ax2.set_ylim(*low_lim)

    d = 0.015
    kwargs = dict(transform=ax1.transAxes, color="black", clip_on=False, linewidth=1.5)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d * 4, 1 + d * 4), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d * 4, 1 + d * 4), **kwargs)

    return ax1, ax2
