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
from utils.filemethods import open_data_file as open_data_file
from utils.utils import trim_nans
from utils.utils import filter_months
import calendar
from datetime import date, timedelta

class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for data in dictionaries.
    """

    def __init__(self, data_file, config):
        config = config["databuilder"]
        lagtime = config["lagtime"]
        smoothing_length = config["averaging_length"]
        selected_months = config["averaging_length"]
        input_years = config["input_years"]
        front_nans = config["front_cutoff"]
        back_nans = config["back_cutoff"]

        dict_data = open_data_file(data_file)

        # If there are leading or ending nans, cut the inputs evenly so there are no longer nans
        trimmed_data = {key: value[front_nans : -back_nans] for key, value in dict_data.items() }
        print(type(trimmed_data["y"]))
        # Cut data to ensure proper lag & alignment: 
        # Remove Lag-length BACK nans from Input
        input = trimmed_data["x"][:-lagtime]
        
        # Remove Lag-length FRONT nans from Target
        target = trimmed_data["y"][lagtime:]
   
        if selected_months is not None: 
            input_filtered, target_filtered = filter_months(input, target, selected_months, lagtime)
        else: 
            print("Using input and target data from all year round")

        # Remove Smoothing-length FRONT nans from BOTH Input and Target
        self.input = input_filtered[smoothing_length:]
        self.target = target_filtered[smoothing_length:]

        assert not np.any(np.isnan(self.input))
        assert not np.any(np.isnan(self.target))

        # Print shapes for debugging
        print(f"X shape: {self.input.shape}")
        print(f"Target shape: {self.target.shape}")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        input = torch.tensor(self.input[idx, ...].data)

        target = self.target[idx]    
        
        return (
            [ torch.tensor(input, dtype=torch.float32)],
            torch.tensor(target, dtype=torch.float32),
        )