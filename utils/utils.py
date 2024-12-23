"""
Utilities: Classes and Functions

Functions: --------------------
get_config(exp_name)



"""
import json
import pandas as pd
import torch
import numpy as np
import calendar
from datetime import date, timedelta

def get_config(exp_name):

    basename = "exp"

    with open("configs/config_" + exp_name[len(basename) :] + ".json") as f:
        config = json.load(f)

    assert config["expname"] == basename + exp_name[len(basename) :], "Exp_Name must be equal to config[exp_name]"

    # add additional attributes for easier use later
    config["fig_dpi"] = config["fig_dpi"]
    config["data_dir"] = config["data_dir"]

    return config


def prepare_device(device="mps"):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    if device == "mps":
        if torch.backends.mps.is_available():
            print("torch.backends.mps is available")
            x = torch.ones(1, device=device)
            print (x)
            device = torch.device("mps")
        else:
            print("Warning: MPS device not found." "Training will be performed on CPU.")
            device = torch.device("cpu")
    elif device == "cpu":
        print("somwhere device = cpu")
        device = torch.device("cpu")
    else:
        raise NotImplementedError

    return device

def trim_nans(data_dict):
    """
    Removes leading and trailing NaNs consistently across all keys.
    
    Args:
        data_dict (dict): A dictionary of data (keys are column names, values are arrays/lists).
    
    Returns:
        dict: A dictionary with trimmed data, with no leading or trailing NaNs.
    """
    # Convert all values to numpy arrays for uniformity
    arrays = {key: np.asarray(value) for key, value in data_dict.items()}
    
    # Determine the valid range across all keys
    start_idx = 0
    end_idx = float('inf')
    
    for array in arrays.values():
        # Find first and last non-NaN indices
        non_nan_indices = np.where(~np.isnan(array))[0]
        if non_nan_indices.size > 0:  # Ensure the array has non-NaN values
            start_idx = max(start_idx, non_nan_indices[0])  # Max of start indices
            end_idx = min(end_idx, non_nan_indices[-1])  # Min of end indices
        else:
            raise ValueError("One of the columns contains only NaNs.")
    
    # Slice all arrays to the valid range [start_idx, end_idx]
    cleaned_data = {key: array[start_idx:end_idx + 1] for key, array in arrays.items()}
    
    return cleaned_data

def save_torch_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_torch_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model


class MetricTracker:
    def __init__(self, *keys):

        self.history = dict()
        for k in keys:
            self.history[k] = []
        self.reset()

    def reset(self):
        for key in self.history:
            self.history[key] = []

    def update(self, key, value):
        if key in self.history:
            self.history[key].append(value)

    def result(self):
        for key in self.history:
            self.history[key] = np.nanmean(self.history[key])

    # def print(self, idx=None):
    #     for key in self.history.keys():
    #         if idx is None:
    #             print(f"  {key} = {self.history[key]:.5f}")
    #         else:
    #             print(f"  {key} = {self.history[key][idx]:.5f}")
    
    def print(self, idx=None):
        for key in self.history.keys():
            if idx is None:
                if isinstance(self.history[key], list):
                    print(f"  {key} = " + " | ".join([f"{v:.5f}" for v in self.history[key]]))
                else:
                    print(f"  {key} = {self.history[key]:.5f}")
            else:
                if isinstance(self.history[key], list):
                    if isinstance(self.history[key][idx], (int, float)):
                        print(f"  {key}[{idx}] = {self.history[key][idx]:.5f}")
                    else:
                        print(f"  {key}[{idx}] = " + " | ".join([f"{v:.5f}" for v in self.history[key][idx]]))
                else:
                    print(f"  {key}[{idx}] = {self.history[key][idx]:.5f}")



def daysinmonth(month):
    import calendar

    # Generate the number of days for each month of a non-leap year
    days_in_month_no_leap = [
        28 if month == 2 else calendar.monthrange(2023, month)[1] 
        for month in range(1, 13) ]

    days_array_no_leap = np.array(days_in_month_no_leap)

    return days_array_no_leap[month]

def specifydates(target_months, lagtime):

    # Helper function to format date as 'MM-DD'
    def format_date(month, day):
        return f"{str(month).zfill(2)}-{str(day).zfill(2)}"

    # Target start date: First day of the first target month
    target_start_month = target_months[0]
    target_start_date = format_date(target_start_month, 1)

    # Input start date: 14 days before the target start date
    input_start_obj = date(2023, target_start_month, 1) - timedelta(days=lagtime)
    input_start_date = format_date(input_start_obj.month, input_start_obj.day)

    # Target end date: Last day of the last target month
    target_end_month = target_months[-1]
    last_day_of_target_end_month = calendar.monthrange(2023, target_end_month)[1]  # Get the last day of the month
    target_end_date = format_date(target_end_month, last_day_of_target_end_month)

    # Input end date: 14 days before the target end date
    input_end_obj = date(2023, target_end_month, last_day_of_target_end_month) - timedelta(days=lagtime)
    input_end_date = format_date(input_end_obj.month, input_end_obj.day)

    # Print the results
    print("Target Start:", target_start_date)  # e.g., '04-01'
    print("Input Start:", input_start_date)    # e.g., '03-18'
    print("Target End:", target_end_date)      # e.g., '09-30'
    print("Input End:", input_end_date)        # e.g., '09-16'

    return target_start_date, target_end_date, input_start_date, input_end_date


def filter_months(input, target, selected_months, lagtime):
    """
    Filters an xarray dataset or DataArray to include only the specified months
    for both input and target arrays

    Parameters:
    - data: xarray.Dataset or xarray.DataArray with a 'time' coordinate (input and target).
    - selected_months: List of integers (1-12) representing the months to keep.
    - lagtime: Optional integer (days) for lagging the start and end dates.

    Returns:
    - Filtered xarray object with only the desired months.
    """
    # Extract month and day from the 'time' coordinate
    months = target["time"].dt.month

    # Filter for selected months
    target_filtered = target.where(months.isin(selected_months), drop=True)

    # Adjust input selection based on lagtime
    _, _, input_start_date, input_end_date = specifydates(selected_months, lagtime)
    
    input_condition = (
    (input.time.dt.month == input_start_date[1:2]) & (input.time.dt.day >= input_start_date[3:]) | 
    (input.time.dt.month == input_end_date[1:2]) & (input.time.dt.day <= input_end_date[3:])
    )

    input_filtered = input.sel(time = input_condition, drop=True)

    return input_filtered, target_filtered

