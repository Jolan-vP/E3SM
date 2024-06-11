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

        cut_val = 121 # remove front nans
        
        self.x1 = dict_data["x"][cut_val:,0]
        self.x2 = dict_data["x"][cut_val:,1]
        self.x3 = dict_data["x"][cut_val:,2]
        self.target = dict_data["y"][cut_val:-1]

        # Print shapes for debugging
        print(f"X1 shape: {self.x1.shape}")
        print(f"Input unit shape: {self.target.shape}")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        x1 = self.x1[idx, ...]
        x2 = self.x2[idx,...]
        x3 = self.x3[idx,...]

        target = self.target[idx]

        return (
            [
                torch.tensor(x1, dtype=torch.float32),
                torch.tensor(x2, dtype=torch.float32),
                torch.tensor(x3, dtype=torch.float32)
            ],
            torch.tensor(target, dtype=torch.float32),
        )