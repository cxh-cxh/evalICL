import os, glob, time
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
import torch
import cv2
import einops
import json, h5py
import collections
from natsort import natsorted
import pdb
from video_sampler import sample_video
import numpy as np


ckpt_path = "/mnt/sdb/benchmarking/train/pi0_2025_9_t10_generalist/checkpoints/400000/pretrained_model"
task_prompt = [""]
policy = PI0Policy.from_pretrained(ckpt_path)  # ACTPolicy DiffusionPolicy PI0Policy
policy.eval()
policy.to(torch.device("cuda"))

model_name = "pi0_drawer"

hdf5_path = "/home/rhos/ICL/data/video_emb.hdf5"
if os.path.exists(hdf5_path):
    database = h5py.File(hdf5_path, "r+")
else:
    database = h5py.File(hdf5_path, "w")


def video_emb(front_imgs, side_imgs):
    batch = {}
    batch_size = len(front_imgs)
    batch["observation.images.view_front"] = torch.tensor(np.stack(front_imgs))
    batch["observation.images.view_side"] = torch.tensor(np.stack(side_imgs))
    batch["observation.state"] = torch.zeros(batch_size, 6)
    for name in batch:
        if "image" in name:
            batch[name] = batch[name].type(torch.float32) / 255
            batch[name] = batch[name].permute(0, 3, 1, 2).contiguous()
        batch[name] = batch[name].to(torch.device("cuda"))
    if model_name.startswith("pi0"):
        batch = policy.normalize_inputs(batch)
        images, img_masks = policy.prepare_images(batch)
        front_img_emb = policy.model.paligemma_with_expert.embed_image(images[0])
        front_img_emb = front_img_emb.to(dtype=torch.bfloat16)
        front_img_emb_dim = front_img_emb.shape[-1]
        front_img_emb = front_img_emb * torch.tensor(
            front_img_emb_dim**0.5,
            dtype=front_img_emb.dtype,
            device=front_img_emb.device,
        )
        side_img_emb = policy.model.paligemma_with_expert.embed_image(images[1])
        side_img_emb = side_img_emb.to(dtype=torch.bfloat16)
        side_img_emb_dim = side_img_emb.shape[-1]
        side_img_emb = side_img_emb * torch.tensor(
            side_img_emb_dim**0.5,
            dtype=front_img_emb.dtype,
            device=front_img_emb.device,
        )
    elif model_name.startswith("act"):
        batch = policy.normalize_inputs(batch)
        if len(policy.expected_image_keys) > 0:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[k] for k in policy.expected_image_keys], dim=-4
            )
        all_cam_features = []
        for cam_index in range(batch["observation.images"].shape[-4]):
            cam_features = policy.model.backbones[cam_index](
                batch["observation.images"][:, cam_index]
            )["feature_map"]
            cam_features = policy.model.encoder_img_feat_input_proj(
                cam_features
            )  # (B, C, h, w)
            all_cam_features.append(cam_features)
        front_img_emb = all_cam_features[0]
        side_img_emb = all_cam_features[1]
    elif model_name.startswith("dp"):
        batch = policy.normalize_inputs(batch)
        if len(policy.expected_image_keys) > 0:
            batch = dict(
                batch
            )  # shallow copy so that adding a key doesn't modify the original
        front_img_emb = policy.diffusion.rgb_encoder(
            batch["observation.images.view_front"]
        )
        side_img_emb = policy.diffusion.rgb_encoder(
            batch["observation.images.view_side"]
        )
    return front_img_emb.mean(dim=0), side_img_emb.mean(dim=0)


video_root = "/mnt/sdb/benchmarking/videos/pi0_2026_1_drawer_aug_drawer/"
video_meta = "meta.json"

env_name = "drawer_env1"

data = collections.defaultdict(dict)
# if os.path.exists(os.path.join(root, "imgemb.json")):
#     with open(os.path.join(root, "imgemb.json"), "r") as f:
#         data.update(json.load(f))

with open(os.path.join(video_root, video_meta), "r") as f:
    meta = json.load(f)

for id, paths in meta.items():
    for path in paths:
        front_imgs = sample_video(path["front"], sample_rate=15)
        side_imgs = sample_video(path["side"], sample_rate=15)

        front_group_path = (
            "/" + model_name + "/" + env_name + "/" + path["front"].split("/")[-1]
        )
        side_group_path = (
            "/" + model_name + "/" + env_name + "/" + path["side"].split("/")[-1]
        )

        if front_group_path in database and side_group_path in database:
            print(front_group_path, side_group_path, "exists")
        else:
            front_img_emb, side_img_emb = video_emb(front_imgs, side_imgs)
            database[front_group_path] = (
                front_img_emb.to(dtype=torch.float32).detach().cpu().numpy().tolist()
            )
            database[side_group_path] = (
                side_img_emb.to(dtype=torch.float32).detach().cpu().numpy().tolist()
            )
