"""Data loader modules.

Classes
---------
CustomData(torch.utils.data.Dataset)

"""

from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import gzip


class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for data in dictionaries.
    """

    def __init__(self, data_file):
        with gzip.open(data_file, "rb") as handle:
            dict_data = pickle.load(handle)

        cut_val = 135 # remove front nans

        self.x1 = dict_data["x"][cut_val:,0]
        self.x2 = dict_data["x"][cut_val:,1]
        self.x3 = dict_data["x"][cut_val:,2]
        self.target = dict_data["y"][cut_val:-1]

        assert not np.any(np.isnan(self.x1))
        assert not np.any(np.isnan(self.x2))
        assert not np.any(np.isnan(self.x3))
        assert not np.any(np.isnan(self.target))

        # Print shapes for debugging
        print(f"X1 shape: {self.x1.shape}")
        print(f"Target shape: {self.target.shape}")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        x1 = self.x1[idx, ...]
        x2 = self.x2[idx,...]
        x3 = self.x3[idx,...]

        target = self.target[idx]

        if np.isnan(x1).any() or np.isnan(x2).any() or np.isnan(x3).any() or np.isnan(target).any():
            print(f"NaN found in input data at index {idx}")
        if np.isinf(x1).any() or np.isinf(x2).any() or np.isinf(x3).any() or np.isinf(target).any():
            print(f"Infinite value found in input data at index {idx}")

        return (
            [
                torch.tensor(x1, dtype=torch.float32),
                torch.tensor(x2, dtype=torch.float32),
                torch.tensor(x3, dtype=torch.float32)
            ],
            torch.tensor(target, dtype=torch.float32),
        )