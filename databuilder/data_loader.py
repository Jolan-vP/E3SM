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

        self.input = dict_data["x"]
        self.target = dict_data["y"]

        if "input_unit" in dict_data:
            self.input_unit = dict_data["input_unit"]
        else:
            # Handle the case where input_unit is not available
            self.input_unit = np.zeros_like(self.target)  # or another appropriate default
        
        # Print shapes for debugging
        print(f"Input shape: {self.input.shape}")
        print(f"Input unit shape: {self.input_unit.shape}")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        input = self.input[idx, ...]
        input_unit = self.input_unit[idx]

        target = self.target[idx]

        return (
            [
                torch.tensor(input, dtype=torch.float32),
                torch.tensor(input_unit, dtype=torch.float32),
            ],
            torch.tensor(target, dtype=torch.float32),
        )