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
import xarray as xr


class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for data in dictionaries.
    """

    def __init__(self, data_file, lagtime, smoothing_length):
        dict_data = xr.open_dataset(data_file)

        # Cut data to ensure proper lag & alignment: 
        # Remove Lag-length BACK nans from Input
        self.input = dict_data["x"][:-lagtime]

        # Remove Lag-length FRONT nans from Target
        self.target = dict_data["y"][lagtime:]

        # Remove Smoothing-length FRONT nans from BOTH Input and Target
        self.input = self.input[smoothing_length:]
        self.target = self.target[smoothing_length:]

        assert not np.any(np.isnan(self.input))
        assert not np.any(np.isnan(self.target))

        # Print shapes for debugging
        print(f"X shape: {self.input.shape}")
        print(f"Target shape: {self.target.shape}")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        input = torch.tensor(self.input[idx, ...].data)

        # Configure Channel Dimension to be in position 1
        input = torch.permute(input, [2, 0, 1])

        target = self.target[idx].values    
        
        return (
            [ torch.tensor(input, dtype=torch.float32)],
            torch.tensor(target, dtype=torch.float32),
        )