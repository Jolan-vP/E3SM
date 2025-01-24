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

def get_netcdf_da(filename):
    ds = xr.open_dataset(filename)
    return ds

def open_data_file(data_file):

    if "." in data_file:
        if data_file.endswith(".pkl"):
            # Open the file using gzip and pickle
            with gzip.open(data_file, "rb") as fp:
                data = pickle.load(fp)
            print(f"Opened pickle file: {data_file}")
        elif data_file.endswith(".nc"):
            # Open the file as a NetCDF dataset using xarray
            data = xr.open_dataset(data_file)
            print(f"Opened NetCDF file: {data_file}")
    else:
        #assume datafile is a passed variable
        data = data_file
        print(f"Data passed directly as {type(data)} rather than filename.")

    return data

