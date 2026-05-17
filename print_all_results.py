import json
import os, glob
import numpy as np

func_names = ["linear", "mult_sqrt", "l2"]
keys = [
    # "pi0_t10003_full",
    # "pi0_t10003_sim_full",
    # "pi0_t7_full",
    # "pi0_t10_full",
    # "pi05_task40_full",
    # "pi0_drawer_full",
    # "pi05_box_no_train_data",
    "pi0_t10003",
    "dp_t10003",
    "act_t10003",
]

func_name_dict = {"linear": "分段线性", "mult_sqrt": "几何平均", "l2": "L2距离"}

data_ = {}

for func_name in func_names:
    with open(f"./collate/result_{func_name}.json") as f:
        data_[func_name] = json.load(f)
        for key in keys:
            print(func_name, key)
            print(
                "old:",
                f"{data_[func_name][key]['old'][-1]:.3f}",
                "new:",
                f"{data_[func_name][key]['new'][-1]:.3f}",
            )
