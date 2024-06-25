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
from sklearn.preprocessing import StandardScaler


class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for data in dictionaries.
    """

    def __init__(self, data_file, front_cutoff, back_cutoff):
        with gzip.open(data_file, "rb") as handle:
            dict_data = pickle.load(handle)

        self.input = dict_data["x"][front_cutoff:-back_cutoff,:]
        self.target = dict_data["y"][front_cutoff:-back_cutoff]
        # # Normalize Inputs
        # scaler = StandardScaler()
        # self.input = scaler.fit_transform(self.input)

        assert not np.any(np.isnan(self.input))
        assert not np.any(np.isnan(self.target))

        # Print shapes for debugging
        print(f"X1 shape: {self.input.shape}")
        print(f"Target shape: {self.target.shape}")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        input = self.input[idx, ...]

        target = self.target[idx]

        return (
            [
                torch.tensor(input, dtype=torch.float32)
            ],
            torch.tensor(target, dtype=torch.float32),
        )