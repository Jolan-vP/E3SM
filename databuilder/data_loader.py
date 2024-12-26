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
import matplotlib as plt

class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for data in dictionaries.
    """

    def __init__(self, data_file, config):
        # # config = config["databuilder"]
        # lagtime = config["databuilder"]["lagtime"]
        # smoothing_length = config["databuilder"]["averaging_length"]
        # selected_months = config["databuilder"]["target_months"]
        # input_years = config["databuilder"]["input_years"]
        # front_nans = config["databuilder"]["front_cutoff"]
        # back_nans = config["databuilder"]["back_cutoff"]

        # dict_data = open_data_file(data_file)

        # # If there are leading or ending nans, cut the inputs evenly so there are no longer nans
        # trimmed_data = {key: value[front_nans : -back_nans] for key, value in dict_data.items() }
     
        # # Cut data to ensure proper lag & alignment: 
        # # Remove Lag-length BACK nans from Input
        # input = trimmed_data["x"][:-lagtime]
        
        # # Remove Lag-length FRONT nans from Target
        # target = trimmed_data["y"][lagtime:]
   
        # if selected_months is not None: 
        #     input_filtered, target_filtered = filter_months(selected_months, lagtime, input = input, target = target)
        # else: 
        #     print("Using input and target data from all year round")

        # # # Check to make sure data has been properly filtered! 
        # # input_filtered_times = input_filtered.time.values
        # # fn = str(config["perlmutter_figure_dir"]) + str(config["expname"]) + "/filtered_input_times_CHECK.pkl"
        # # with gzip.open(fn, "wb") as fp:
        # #     pickle.dump(input_filtered_times, fp)

        # # Remove Smoothing-length FRONT nans from BOTH Input and Target
        # input = input_filtered[smoothing_length:]
        # target = target_filtered[smoothing_length:]

        input, target = universaldataloader(data_file, config, target_only = False)

        # Ensure that the inputs and targets are now numpy arrays not xarray objects for the model
        self.input = input.values
        self.target= target.values
    
        assert not np.any(np.isnan(self.input))
        assert not np.any(np.isnan(self.target))

        assert len(self.input) == len(self.target)

        # Print shapes for debugging
        print(f"Input shape: {self.input.shape}")
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
    

def universaldataloader(data_file, config, target_only = False): 
     # config = config["databuilder"]
    lagtime = config["databuilder"]["lagtime"]
    smoothing_length = config["databuilder"]["averaging_length"]
    selected_months = config["databuilder"]["target_months"]
    input_years = config["databuilder"]["input_years"]
    front_nans = config["databuilder"]["front_cutoff"]
    back_nans = config["databuilder"]["back_cutoff"]

    dict_data = open_data_file(data_file)

    # If there are leading or ending nans, cut the inputs evenly so there are no longer nans
    trimmed_data = {key: value[front_nans : -back_nans] for key, value in dict_data.items() }
    
    # Cut data to ensure proper lag & alignment: 
    # Remove Lag-length BACK nans from Input
    input = trimmed_data["x"][:-lagtime]
    
    # Remove Lag-length FRONT nans from Target
    target = trimmed_data["y"][lagtime:]

    if target_only is False: 
        if selected_months is not None: 
            input_filtered, target_filtered = filter_months(selected_months, lagtime, input = input, target = target)
        else: 
            print("Using input and target data from all year round")

        # # Check to make sure data has been properly filtered! 
        # input_filtered_times = input_filtered.time.values
        # fn = str(config["perlmutter_figure_dir"]) + str(config["expname"]) + "/filtered_input_times_CHECK.pkl"
        # with gzip.open(fn, "wb") as fp:
        #     pickle.dump(input_filtered_times, fp)

        # Remove Smoothing-length FRONT nans from BOTH Input and Target
        input = input_filtered[smoothing_length:]
        target = target_filtered[smoothing_length:]

        return input, target 
    
    elif target_only is True: 
        if selected_months is not None: 
            target_filtered = filter_months(selected_months, lagtime, input = None, target = target)
        else: 
            print("Using input and target data from all year round")

        # # Check to make sure data has been properly filtered! 
        # input_filtered_times = input_filtered.time.values
        # fn = str(config["perlmutter_figure_dir"]) + str(config["expname"]) + "/filtered_input_times_CHECK.pkl"
        # with gzip.open(fn, "wb") as fp:
        #     pickle.dump(input_filtered_times, fp)

        # Remove Smoothing-length FRONT nans from BOTH Input and Target
        target = target_filtered[smoothing_length:]

        return target