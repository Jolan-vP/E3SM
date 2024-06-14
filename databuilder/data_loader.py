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

        front_cutoff = 121  # remove front nans
        back_cutoff = 32    # remove back nans

        self.input = dict_data["x"][front_cutoff:-back_cutoff,:]
        self.target = dict_data["y"][front_cutoff:-back_cutoff-1]

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