import os, glob, time
import json
import collections
from natsort import natsorted

root = "task40_env3"

front_imgs = natsorted(glob.glob(f"front_*.png", root_dir=root))

data = collections.defaultdict(dict)
# if os.path.exists(os.path.join(root, "info.json")):
#     with open(os.path.join(root, "info.json"), "r") as f:
#         data.update(json.load(f))
data["num"] = len(front_imgs)
data["name"] = root

for i in range(len(front_imgs)):
    data[str(i)]["front"] = front_imgs[i]

with open(os.path.join(root, "info.json"), "w") as f:
    json.dump(data, f, indent=4)
