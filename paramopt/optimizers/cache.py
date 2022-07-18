from pathlib import Path

import numpy as np


class PredictionCache:

    def __init__(self, name) -> None:
        self.name = name
        self.data = {
            "label": [],
            "mean": [],
            "std": [],
            "acquisition": [],
            "next_combination": []
        }

    def stack(self, label, mean, std, acquisition, next_combination):
        self.data["label"].append(label)
        self.data["mean"].append(mean)
        self.data["std"].append(std)
        self.data["acquisition"].append(acquisition)
        self.data["next_combination"].append(next_combination)

    def save(self, directory):
        np.savez((Path(directory)/self.name).with_suffix(".npz"), **self.data)
