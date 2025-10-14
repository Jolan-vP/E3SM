"""Data loader modules.

Classes
---------
CustomData(torch.utils.data.Dataset)

"""

import gc
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
import matplotlib.pyplot as plt
from databuilder.sampleclass import SampleDict
import analysis.analysis_metrics as am

class CustomData(torch.utils.data.Dataset):
    """
    Custom dataset for data in dictionaries.
    """

    def __init__(self, data_file, config, which_set = None):
     
        dict_data = open_data_file(data_file)

        self.input = dict_data["x"]
        self.target= dict_data["y"]

        # STANDARDIZE data using FIXED CLIMATOLOGY stats: 
        climo_stats = open_data_file(str(config["databuilder"]["climatology_stats_path"]))
        
        # PRECT, TS, Z500   -or-    PRECT, TS
        for i in range(len(config["databuilder"]["input_vars"])):
            i_mean_input = climo_stats[config["databuilder"]["input_vars"][i]][0] 
            i_std_input = climo_stats[config["databuilder"]["input_vars"][i]][1]
            self.input[..., i] = (self.input[..., i] - i_mean_input) / i_std_input

        i_mean_target = climo_stats[config["databuilder"]["target_var"]][2] 
        i_std_target = climo_stats[config["databuilder"]["target_var"]][3]

        self.target = (self.target - i_mean_target) / i_std_target

        print(f"shape of self.input: {self.input.shape}")
        #TODO: EVENTUALLY FIX!!
        if self.input.shape[1] > 180: 
            self.input = self.input[:, :180, ...]
            print(f"MODIFIED SELF.INPUT SHAPE: {self.input.shape}")

        # SCALE TARGET BY DAY OF YEAR VARIANCE: 
        if config["data_loader"]["scale_target_variance"]: 
            print("Scaling target by day of year variance")
            daily_target_grouped_var = self.target.groupby('time.dayofyear').var('time')
            scaled_target = self.target.groupby('time.dayofyear') / daily_target_grouped_var
            scaled_target = scaled_target.sortby('time')
            self.target = scaled_target


        self.input = self.input.values
        self.target = self.target.values

        assert not np.any(np.isnan(self.input))
        assert not np.any(np.isnan(self.target))

        assert len(self.input) == len(self.target)


        # SHUFFLE TARGET DATA : 
        # if config["data_loader"]["shuffle_target"] == "True":
        #     print("Shuffling target data")
        #     np.random.seed(config["seed_list"][0])
        #     np.random.shuffle(self.target)


        # # DEBUGGING: Print first few samples before shuffling
        # print("Before shuffle - first 3 input samples:")
        # print(self.input[:3, 0, 0]) 

        # # SHUFFLE INPUT DATA : 
        # if which_set == 'training':
        #     print("Shuffling input data")
        #     np.random.seed(config["seed_list"][0])
        #     indices = np.arange(len(self.input))
        #     #TODO: check indices changing w/ each seed
        #     np.random.shuffle(indices)
        #     self.input = self.input[indices]
        #     self.target = self.target[indices]

        # print("After shuffle - first 3 input samples:")
        # print(self.input[:3, 0, 0])

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
        print("Data is an xarray dataset")
        # Access 'x' and 'y' directly since they behave like a dictionary
        input = data['x']
        target = data['y']
        
        # Handle front/back NaNs
        if front_nans > 0 or back_nans > 0:
            input_trimmed = input[front_nans : -back_nans]
            target_trimmed = target[front_nans : -back_nans]

            del data  # Clear original
            gc.collect()
        else: 
            input_trimmed = input
            target_trimmed = target

        # Apply lagtime adjustment
        input = input_trimmed[:-lagtime]
        target = target_trimmed[lagtime:]

        del input_trimmed, target_trimmed  # Clear trimmed variables
        gc.collect()

        print(f"input shape post lag: {input.shape}")
        print(f"target shape post lag: {target.shape}")

    elif isinstance(data, dict):
        print("Data is a dictionary")

        # If there are leading or ending nans, cut the inputs evenly so there are no longer nans
        if front_nans > 0 or back_nans > 0:
            trimmed_data = {key: value[front_nans : -back_nans] for key, value in data.items() }
        else: 
            trimmed_data = {key: value for key, value in data.items() }

        # Remove Lag-length BACK nans from Input
        input = trimmed_data["x"][:-lagtime]
        
        # Remove Lag-length FRONT nans from Target
        target = trimmed_data["y"][lagtime:]
        

    else: 
        print("Data is not a dictionary or xarray dataset")
        # assume that if it is not a dictionary passed, then only a target is passed with no inputs
        if front_nans > 0 or back_nans > 0:
            target = data[front_nans : -back_nans]
        else: 
            target = data
            
        target = target[lagtime:]

    # use assigned target and input variables as inputs for filter months function to select target months
    if target_only is False: 
        if selected_months != "None": 
            print(f"Filtering by months: {selected_months}")
            input_filtered, target_filtered = filter_months(selected_months, lagtime, input = input, target = target)
             
            del input, target  # Clear original
            gc.collect()

        else: 
            print("Using input and target data from all year round")
            input_filtered = input
            target_filtered = target

        # Remove Smoothing-length FRONT nans from BOTH Input and Target
        input_mod_final = input_filtered[smoothing_length:]
        target_mod_final = target_filtered[smoothing_length:]

        del input_filtered, target_filtered  # Clear filtered variables
        gc.collect()
        
        # print(f"input_mod_final shape: {input_mod_final.shape}")
        # print(f"target_mod_final shape: {target_mod_final.shape}")

        if repackage == False:
            return input, target 
        
        else: 
            print("Repackaging data into xarray dataset")

            # check if there are nans and what location the nans are: 
            if np.any(np.isnan(input_mod_final)):
                print("Input data has nans")
                # print location of nans
                print(np.where(np.isnan(input_mod_final)))

            # ALL TRIMMED DATA HAS COORDINATES OF THE TARGET DATA
            if len(input_mod_final.shape) == 4: 
                data_dict = xr.Dataset({
                    "x": (["time", "lat", "lon", "channel"], input_mod_final.data),  
                    "y": (["time"], target_mod_final.data),
                    }, coords = {
                        "time": target_mod_final.coords["time"],
                        "lat": input_mod_final.coords["lat"],
                        "lon": input_mod_final.coords["lon"],
                        "channel": input_mod_final.coords["channel"]
                    })
            elif len(input_mod_final.shape) == 3: 
                data_dict = xr.Dataset({
                    "x": (["time", "lat", "lon"], input_mod_final.data),  
                    "y": (["time"], target_mod_final.data),
                    }, coords = {
                        "time": target_mod_final.coords["time"],
                        "lat": input_mod_final.coords["lat"],
                        "lon": input_mod_final.coords["lon"],
                    })
            elif len(input_mod_final.shape) == 2: 
                data_dict = xr.Dataset({
                    "x": (["time", "channel"], input_mod_final.data),  
                    "y": (["time"], target_mod_final.data),
                    }, coords = {
                        "time": target_mod_final.coords["time"],
                        "channel": input_mod_final.coords["channel"]
                    })
            else: 
                print("Data shape is not expected. Must be either Map or Simple Inputs")

                raise ValueError
            
            # print(f"target time: {data_dict['y'].sel(time = slice('1860-01-01','1861-01-01'))}") 
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
    




def select_years(data, config):
    print("Processing Training")
    f_dict_train = self._process_data(train_ds)
    f_dict_train['x'] = f_dict_train['x'].sel(time = slice(str(self.config["train_years"][0]), str(self.config["train_years"][1])))
    f_dict_train['y'] = f_dict_train['y'].sel(time = slice(str(self.config["train_years"][0]), str(self.config["train_years"][1])))
    print("Processing validation")
    f_dict_val = self._process_data(validate_ds)
    f_dict_val['x'] = f_dict_val['x'].sel(time = slice(str(self.config["val_years"][0]), str(self.config["val_years"][1])))
    f_dict_val['y'] = f_dict_val['y'].sel(time = slice(str(self.config["val_years"][0]), str(self.config["val_years"][1])))
    print("Processing testing")
    f_dict_test = self._process_data(test_ds)
    f_dict_test['x'] = f_dict_test['x'].sel(time = slice(str(self.config["test_years"][0]), str(self.config["test_years"][1])))
    f_dict_test['y'] = f_dict_test['y'].sel(time = slice(str(self.config["test_years"][0]), str(self.config["test_years"][1])))   


# GARBAGE HEAP: -----------------------------------------


  # # Check to make sure data has been properly filtered! 
        # input_filtered_times = input_filtered.time.values
        # fn = str(config["perlmutter_figure_dir"]) + str(config["expname"]) + "/filtered_input_times_CHECK.pkl"
        # with gzip.open(fn, "wb") as fp:
        #     pickle.dump(input_filtered_times, fp)