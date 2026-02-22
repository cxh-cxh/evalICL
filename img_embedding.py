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

ckpt_path = "/mnt/sdb/benchmarking/train/pi0_2025_8_t10003_sim_100000/checkpoints/400000/pretrained_model"
# ckpt_path = "/mnt/sda/cyx/outputs/train/dp_2025_8_t10003_cotrain_new_aug_seed_100000/checkpoints/225000/pretrained_model"
test_cases_path = "/home/rhos/ICL/images"
task_prompt = [""]
policy = PI0Policy.from_pretrained(ckpt_path)  # ACTPolicy DiffusionPolicy PI0Policy
policy.eval()
policy.to(torch.device("cuda"))

model_name = "pi0_t10003_sim"
# model_name = "pi0_t10"

hdf5_path = "/home/rhos/ICL/data/img_emb.hdf5"
if os.path.exists(hdf5_path):
    database = h5py.File(hdf5_path, "r+")
else:
    database = h5py.File(hdf5_path, "w")


def img_emb(front_img, side_img):
    batch = {}
    batch["observation.images.view_front"] = front_img
    batch["observation.images.view_side"] = side_img
    batch["observation.state"] = torch.zeros(1, 6)
    for name in batch:
        if "image" in name:
            batch[name] = batch[name].type(torch.float32) / 255
            batch[name] = batch[name].permute(2, 0, 1).contiguous()
        batch[name] = batch[name].unsqueeze(0)
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
    return front_img_emb, side_img_emb


root = "/home/rhos/ICL/images"
infos = natsorted(glob.glob(f"t10003_sim*/info.json", root_dir=root))

data = collections.defaultdict(dict)
# if os.path.exists(os.path.join(root, "imgemb.json")):
#     with open(os.path.join(root, "imgemb.json"), "r") as f:
#         data.update(json.load(f))

for info in infos:
    with open(os.path.join(test_cases_path, info), "r") as f:
        info_ = json.load(f)
    name = info_["name"]
    num = info_["num"]
    for i in range(num):
        front_img_path = info_[str(i)]["front"]
        front_img = torch.tensor(
            cv2.imread(os.path.join(test_cases_path, name, front_img_path))
        )
        side_img_path = info_[str(i)]["side"]
        side_img = torch.tensor(
            cv2.imread(os.path.join(test_cases_path, name, side_img_path))
        )

        front_group_path = "/" + model_name + "/" + name + "/" + front_img_path
        side_group_path = "/" + model_name + "/" + name + "/" + side_img_path

        if front_group_path in database and side_group_path in database:
            print(front_group_path, side_group_path, "exists")
        else:
            front_img_emb, side_img_emb = img_emb(front_img, side_img)
            database[front_group_path] = (
                front_img_emb.to(dtype=torch.float32).detach().cpu().numpy().tolist()
            )
            database[side_group_path] = (
                side_img_emb.to(dtype=torch.float32).detach().cpu().numpy().tolist()
            )


# with open(os.path.join(root, "imgemb.json"), "w") as f:
#     json.dump(data, f, indent=4)
