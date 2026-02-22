# Given img path, retrieve k result items

import h5py, json
import numpy as np

# from scipy.spatial import cKDTree
import faiss
from typing import List, Dict
import os


class Retriever:
    def __init__(self, test_names: List[str], model_name, img_emb_path):
        self.model_name = model_name
        self.img_emb = h5py.File(img_emb_path, "r")
        self.front_img_emb = []
        self.test_results = []
        for test_name in test_names:
            eval_data_path = "data/" + test_name + ".jsonl"
            img_info_path = "images/" + test_name + "/info.json"
            with open(img_info_path, "r") as f:
                info = json.load(f)
            with open(eval_data_path, "r") as f:
                eval_data = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                        eval_data.append(ex)
                        self.front_img_emb.append(
                            self.img_emb[
                                "/"
                                + model_name
                                + "/"
                                + test_name
                                + "/"
                                + info[ex["index"]]["front"]
                            ][:].reshape(-1)
                        )
                        self.test_results.append(ex)
                    except:
                        pass
        self.front_img_emb = np.stack(self.front_img_emb)
        self.index = faiss.IndexFlatL2(self.front_img_emb.shape[1])
        self.index.add(self.front_img_emb)

    def __init__(
        self,
        starting_test_records: List[dict],
        starting_test_names: List[str],
        model_name,
        img_emb_path,
        is_video=False,
    ):
        self.model_name = model_name
        self.img_emb = h5py.File(img_emb_path, "r")
        self.front_img_emb = []
        self.test_results = []
        self.test_names = []
        self.is_video = is_video
        for test_name, test_record in zip(starting_test_names, starting_test_records):
            img_info_path = "images/" + test_name + "/info.json"
            if not self.is_video:
                with open(img_info_path, "r") as f:
                    info = json.load(f)
            self.test_results.append(test_record)
            self.test_names.append(test_name)
            self.front_img_emb.append(
                self.img_emb[
                    (
                        "/"
                        + model_name
                        + "/"
                        + test_name
                        + "/"
                        + (
                            test_record["video"]["front"]
                            if self.is_video
                            else info[test_record["index"]]["front"]
                        )
                    )
                ][:].reshape(-1)
            )
        if len(self.front_img_emb) > 0:
            self.front_img_emb = np.stack(self.front_img_emb)
            self.index = faiss.IndexFlatL2(self.front_img_emb.shape[1])
            self.index.add(self.front_img_emb)
        else:
            self.index = faiss.IndexFlatL2()

    def retrieve(self, test_record, test_name, k=5, increament=False):
        img_info_path = "images/" + test_name + "/info.json"
        if not self.is_video:
            with open(img_info_path, "r") as f:
                info = json.load(f)
        query_point = self.img_emb[
            "/"
            + self.model_name
            + "/"
            + test_name
            + "/"
            + (
                test_record["video"]["front"]
                if self.is_video
                else info[test_record["index"]]["front"]
            )
        ][:].reshape(1, -1)

        if self.index.d != query_point.shape[1]:
            self.index = faiss.IndexFlatL2(query_point.shape[1])

        distances, indices = self.index.search(query_point, k=k)

        if increament:
            self.test_results.append(test_record)
            self.test_names.append(test_name)
            self.index.add(query_point)
        return (
            indices,
            [self.test_results[index] for index in indices[0] if index >= 0],
            [self.test_names[index] for index in indices[0] if index >= 0],
        )


if __name__ == "__main__":
    test = Retriever(
        ["t10003_env1"],
        model_name="pi0_t10003",
        img_emb_path="data/img_emb.hdf5",
    )
    print(test.retrieve("/pi0_t10003/t10003_env1/front_1.png"))
