# python
import os, json, re, asyncio, functools, pathlib, base64, random
from openai import AsyncOpenAI, OpenAI
from datetime import datetime
from retrieval import Retriever
from config import configs
from templete import *
import cv2
import av


def sample_av1_video(video_path, sample_rate=1):
    container = av.open(video_path)

    sampled_frames = []
    frame_count = 0
    sampled_count = 0

    for frame in container.decode(video=0):
        rgb_frame = frame.to_ndarray(format="bgr24")
        print(f"Read frame with shape: {rgb_frame.shape}")

        # 按采样率提取帧
        if frame_count % sample_rate == 0:
            sampled_frames.append(rgb_frame)
            sampled_count += 1

        frame_count += 1

    return sampled_frames


def sample_video(
    input_video_path,
    sample_rate=1,
    save_frames=False,
    verbose=False,
    num: int | None = None,
):
    """
    读取视频并按指定采样率提取帧

    参数:
        input_video_path (str): 输入视频文件路径
        output_folder (str): 保存采样帧的文件夹路径(如果为None则不保存)
        sample_rate (int): 采样率(每n帧采样1帧)，默认为1(每帧都采样)
        save_frames (bool): 是否将采样帧保存为图像文件，默认为False
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {input_video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if num is not None:
        sample_rate = total_frames // num

    if verbose:
        print(f"视频信息: {os.path.basename(input_video_path)}")
        print(f"分辨率: {width}x{height}")
        print(f"帧率(FPS): {fps:.2f}")
        print(f"总帧数: {total_frames}")
        print(f"采样率: 每{sample_rate}帧采样1帧")

    sampled_frames = []
    frame_count = 0
    sampled_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 按采样率提取帧
        if frame_count % sample_rate == 0:
            sampled_frames.append(frame)
            sampled_count += 1

        frame_count += 1

    cap.release()

    print(f"实际采样帧数: {sampled_count}/{total_frames}")
    return sampled_frames
