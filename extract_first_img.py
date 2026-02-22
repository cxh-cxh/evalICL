import os, json, re, asyncio, functools, pathlib, base64, random, glob
from openai import AsyncOpenAI, OpenAI
from datetime import datetime
from retrieval import Retriever
from config import configs
from templete import *
import cv2


def read_first_img(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {input_video_path}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"视频信息: {os.path.basename(input_video_path)}")
    print(f"分辨率: {width}x{height}")
    print(f"帧率(FPS): {fps:.2f}")
    print(f"总帧数: {total_frames}")

    ret, frame = cap.read()

    return frame


video_root = "/mnt/sdb/benchmarking/videos/pi05_box_4"
video_meta = os.path.join(video_root, "meta.json")
with open(video_meta, "r") as f:
    video_meta = json.load(f)

save_path = "images/box_4"
os.makedirs(save_path, exist_ok=True)


for id, paths in enumerate(video_meta.values()):
    front_img = read_first_img(
        os.path.join(video_root, paths[0]["front"].split("/")[-1])
    )
    side_img = read_first_img(os.path.join(video_root, paths[0]["side"].split("/")[-1]))
    cv2.imwrite(os.path.join(save_path, "front_" + str(id) + ".png"), front_img)
    cv2.imwrite(os.path.join(save_path, "side_" + str(id) + ".png"), side_img)

import collections
from natsort import natsorted


front_imgs = natsorted(glob.glob(f"front_*.png", root_dir=save_path))
side_imgs = natsorted(glob.glob(f"side_*.png", root_dir=save_path))

data = collections.defaultdict(dict)
# if os.path.exists(os.path.join(root, "info.json")):
#     with open(os.path.join(root, "info.json"), "r") as f:
#         data.update(json.load(f))
data["num"] = len(front_imgs)
data["name"] = save_path.split("/")[-1]

for i in range(len(front_imgs)):
    data[str(i)]["front"] = front_imgs[i]
    data[str(i)]["side"] = side_imgs[i]

with open(os.path.join(save_path, "info.json"), "w") as f:
    json.dump(data, f, indent=4)
