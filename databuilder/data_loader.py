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
from databuilder.sampleclass import SampleDict
import analysis.analysis_metrics as am

class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for data in dictionaries.
    """

    def __init__(self, data_file, config, which_set = None):
    
        dict_data = open_data_file(data_file)

        self.input = dict_data["x"].values
        self.target= dict_data["y"].values

        # Normalize data using TRAINING stats: 
        if which_set == "training":
            i_std = np.std(self.input, axis = 0)
            i_mean = np.mean(self.input, axis = 0)
            stats = {
                'input_std': i_std,
                'input_mean': i_mean
            }
            am.save_pickle(stats, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/train_stats.pkl")
            print("Saved training stats")

            self.input = (self.input - i_mean) / i_std

        elif which_set == "validation": 
            stats = open_data_file(str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/train_stats.pkl")
            i_std = stats['input_std']
            i_mean = stats['input_mean']

            # self.input = self.input[::3]
            # self.target = self.target[::3]

        elif which_set == "testing":
            stats = open_data_file(str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/train_stats.pkl")
            i_std = stats['input_std']
            i_mean = stats['input_mean']

            self.input = (self.input - i_mean) / i_std

        assert not np.any(np.isnan(self.input))
        assert not np.any(np.isnan(self.target))

        assert len(self.input) == len(self.target)

        # Print shapes for debugging
        print(f"Input shape: {self.input.shape}")
        print(f"Target shape: {self.target.shape}")

        # SHUFFLE TARGET DATA : 
        if config["data_loader"]["shuffle_target"] == "True":
            print("Shuffling target data")
            np.random.seed(config["seed_list"][0])
            np.random.shuffle(self.target)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):

        input = self.input[idx, ...]
        
        target = self.target[idx]
        
        return (
            [ torch.tensor(input, dtype=torch.float32)],
            torch.tensor(target, dtype=torch.float32),
        )
    

def universaldataloader(data_file, config, target_only = False, repackage = False): 
    config_db = config["databuilder"]
    lagtime = config_db["lagtime"]
    smoothing_length = config_db["averaging_length"]
    selected_months = config_db["target_months"]
    front_nans = config_db["front_cutoff"]
    back_nans = config_db["back_cutoff"]

    data = open_data_file(data_file)
    # print(type(data))
    # print(data['x'].shape)
    # print(data['y'].shape)

    # assign nan-less and lagged input and target variables: 
    if isinstance(data, xr.Dataset):
        # Access 'x' and 'y' directly since they behave like a dictionary
        input = data['x']
        target = data['y']

        # Handle front/back NaNs
        input_trimmed = input[front_nans : -back_nans]
        target_trimmed = target[front_nans : -back_nans]

        # Apply lagtime adjustment
        input = input_trimmed[:-lagtime]
        target = target_trimmed[lagtime:]

        print(f"input shape post lag: {input.shape}")
        print(f"target shape post lag: {target.shape}")

    elif isinstance(data, dict):
        # If there are leading or ending nans, cut the inputs evenly so there are no longer nans
        trimmed_data = {key: value[front_nans : -back_nans] for key, value in data.items() }
        
        # Remove Lag-length BACK nans from Input
        input = trimmed_data["x"][:-lagtime]
        
        # Remove Lag-length FRONT nans from Target
        target = trimmed_data["y"][lagtime:]

    else: 
        # assume that if it is not a dictionary passed, then only a target is passed with no inputs
        target = data[front_nans : -back_nans]
        target = target[lagtime:]

    # use assigned target and input variables as inputs for filter months function to select target months
    if target_only is False: 
        if selected_months != "None": 
            print(f"Filtering by months: {selected_months}")
            input_filtered, target_filtered = filter_months(selected_months, lagtime, input = input, target = target)
            print(f"input filtered shape: {input_filtered.shape}")
            print(f"target filtered shape: {target_filtered.shape}")
        else: 
            print("Using input and target data from all year round")
            input_filtered = input
            target_filtered = target

        # Remove Smoothing-length FRONT nans from BOTH Input and Target
        input_mod_final = input_filtered[smoothing_length:]
        target_mod_final = target_filtered[smoothing_length:]
        
        print(f" input_mod_final time: {input_mod_final.time}")
        # print(f"input_mod_final shape: {input_mod_final.shape}")
        # print(f"target_mod_final shape: {target_mod_final.shape}")

        if repackage == False:
            return input, target 
        
        else: 
            if len(data['x'].shape)  == 4: 
                data_dict = xr.Dataset({
                    "x": (["time", "lat", "lon", "channel"], input_mod_final.data),  
                    "y": (["time"], target_mod_final.data),
                    })
            if len(data['x'].shape)  == 3: 
                data_dict = xr.Dataset({
                    "x": (["time", "lat", "lon"], input_mod_final.data),  
                    "y": (["time"], target_mod_final.data),
                    })
            elif len(data['x'].shape) == 2: 
                data_dict = xr.Dataset({
                    "x": (["time", "channel"], input_mod_final.data),  
                    "y": (["time"], target_mod_final.data),
                    })
            else: 
                print("Input data shape is not expected. Must be either Map or Simple Inputs")
                raise ValueError
            
            return data_dict
    
    elif target_only is True: 
        if selected_months != "None": 
            target_filtered = filter_months(selected_months, lagtime, input = None, target = target)
        else: 
            print("Using input and target data from all year round")
            target_filtered = target

        # Remove Smoothing-length FRONT nans from BOTH Input and Target
        target = target_filtered[smoothing_length:]

        return target
    


# GARBAGE HEAP: -----------------------------------------


  # # Check to make sure data has been properly filtered! 
        # input_filtered_times = input_filtered.time.values
        # fn = str(config["perlmutter_figure_dir"]) + str(config["expname"]) + "/filtered_input_times_CHECK.pkl"
        # with gzip.open(fn, "wb") as fp:
        #     pickle.dump(input_filtered_times, fp)