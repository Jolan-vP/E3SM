"""
Functions for working with generic files.

Functions: -----------------------
get_model_name(settings)
get_netcdf_da(filename)
save_pred_obs(pred_vector, filename)

"""
import xarray as xr
import pickle
import gzip
import numpy as np
import pandas as pd
import os

def get_netcdf_da(filename):
    ds = xr.open_dataset(filename)
    return ds

def open_data_file(data_file):

    if "." in data_file:
        if data_file.endswith(".pkl"):
            # Open the file using gzip and pickle
            with gzip.open(data_file, "rb") as fp:
                data = pickle.load(fp)
            # print(f"Opened pickle file: {data_file}")
        elif data_file.endswith(".nc"):
            # Open the file as a NetCDF dataset using xarray
            data = xr.open_dataset(data_file)
            # print(f"Opened NetCDF file: {data_file}")
        elif data_file.endswith(".txt"):
            # Open the file as a text file
            with open(data_file, "r") as fp:
                data = pd.read_csv(fp, sep='\s+', header=0)
            # print(f"Opened text file as Pandas DF: {data_file}")
    else:
        #assume datafile is a passed variable
        data = data_file
        # print(f"Data passed directly as {type(data)} rather than filename.")

    return data

def create_folder(folder_name):
    """Creates a new folder with the given name.

    Args:
        folder_name: The name of the folder to create.
    """
    try:
        os.mkdir(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    except FileExistsError:
        print(f"Folder '{folder_name}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

def filter_dates(dates1, dates2, valid_times, target_months):
    # Ensure valid_times is a set for faster lookup
    valid_times_set = set(valid_times)

    def is_valid_date(d):
        try:
            return d in valid_times_set and d.month in target_months
        except Exception:
            return False

    # Handle xarray.DataArray
    if isinstance(dates1, xr.DataArray):
        dates1_np = dates1.values
        dates2_np = dates2.values
    else:  # assume numpy.ndarray
        dates1_np = dates1
        dates2_np = dates2

    # Apply filtering
    dates1_filtered = np.array([d for d in dates1_np if is_valid_date(d)])
    dates2_filtered = np.array([d for d in dates2_np if is_valid_date(d)])

    return dates1_filtered, dates2_filtered