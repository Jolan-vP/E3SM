""""
Module: Loading data, masking, removing seasonal cycle, isolating abnormalities

"""

import os
import pickle
import numpy as np
import xarray as xr
import importlib as imp
import gzip

import fileops as fileops



##?? What is DA? ******
def maskout_landocean(da, mask_type, directory):
    if mask_type is None:
        return da

    mask = xr.open_dataset(directory + "/landfrac.bilin.nc")["LANDFRAC"][0,:,:]
    if mask_type == "land":
        da_mask = da * xr.where(mask > 0.5, 1.0, 0.0)
    elif mask_type == "ocean":
        da_mask = da * xr.where(mask> 0.5, 0.0, 1.0)
    else:
        raise NotImplementedError()

    return da_mask

#--------------------------------------------------------------------------------------------

## Open and store .nc files based on which member is of interest and the associated settings
def load_inputs(directory, settings):
    x_train, x_val = None, None #?? what do these refer to initially?

    for index_variable, variable in enumerate(settings["input_variables"]):
        assert len(settings["input_mask"]) == len(settings["input_variables"]) #?? why do input mask and input vars contents relate to one another?
        assert len(settings["input_region"]) == len(settings["input_variables"])

        # training data
        x_t = fileops.open_netcdf(settings, directory, settings["training_ens"], variable)
        x_t = maskout_landocean(x_t, settings["input_mask"][index_variable], directory)
        x_t, __, __ = extract_region(x_t, settings["input_region"][index_variable])
        x_t = x_t.expand_dims(dim="channel", axis=1)
        x_t = rolling_ave(settings, x_t)

        # validation data
        # x_v = fileops.open_netcdf(settings, directory, settings["validation_ens"], variable)
        # x_v = maskout_landocean(x_t, settings["input_mask"][index_variable], directory)
        # x_v, __, __ = extract_region(x_t, settings["input_mask"][index_variable], directory)
        # x_v = x_t.expand_dims(dim="channel", axis=1)
        # x_v = rolling_ave(settings, x_t)

        # remove seasonal cycle
        x_t = trend_remove_seasonalcycle(x_t)
        print(x_t)
        # x_v = trend_remove_seasonalcycle(x_v)
    return x_train, x_t #, x_val

#--------------------------------------------------------------------------------------------

# specify which region of the world is used for...... ?? Is extract region used both for the incoming variables for anomalies??
# Is this function also used to select the location that prediction skill is to be tested on? 
def extract_region(data, region=None, lat=None, lon=None, drop=True):
    if region is None:
        min_lon, max_lon = [0, 360]
        min_lat, max_lat = [-90, 90]
    else:
        min_lat, max_lat = region[:2]
        min_lon, max_lon = region[2:]
    
    if isinstance(data, xr.DataArray):
        mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
        mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
        data_masked = data.where(mask_lon & mask_lat, drop=True)
        return (
            data_masked, 
            data_masked["lat"].to_numpy().astype(np.float32),
            data_masked["lon"].to_numpy().astype(np.float32),
        )
    else:
        raise NotImplementedError("Data must be Xarray")


def trend_remove_seasonalcycle(x):
    if len(x.shape) == 1:  
        # Here are we removing the day-of-year that does not contain information in the "time" variable? 
        return x.groupby("time.dayofyear").map(subtract_trend).dropna("time") 
    else:
        x_out = x.copy()  # What does this do, and why?
        inc = 45          # increment? or something else?
        for loop_index in np.arange(0, x.shape[2] // inc + 1): 
            start = inc * loop_index
            end = np.min([inc * (loop_index +1), x_out.shape[2]])
            if start ==end:
                break

            stacked = x[:, :, start:end, :].stack(z=("lat", "lon", "channel"))
            x_out[:, :, start:end] = stacked.groupby("time.dayofyear").map(subtract_trend).unstack()
    return x_out.dropna("time")


def subtract_trend(x):
    detrendOrder = 3 
    curve = np.polynomial.polynomial.polyfit(np.arange(0, x.shape[0]), x, detrendOrder)
    trend = np.polynomial.polynomial.polyval(np.arange(0, x.shape[0]), curve, tensor=True)

    try: 
        detrend = x - np.swapaxes(trend, 0, 1)
    except:
        detrend = x - trend

    return detrend.astype("float32")

def rolling_ave(settings, x):
    if settings["averaging_length"] == 0 :
        return x
    else: 
        if len(x.shape) == 1:
            return x.rolling(time=settings["averaging_length"]).mean()
        else:
            x_out = x.copy() #?? This is the same code as the trend_remove_seasonalcycle. Why increment of 45??
            inc = 45
            for loop_index in np.arange(0, x.shape[2] // inc + 1): 
                start = inc * loop_index
                end = np.min([inc * (loop_index +1), x_out.shape[2]])
                if start ==end:
                    break

            x_out[:, :, start:end, :] = x[:, :, start:end, :].rolling(time=settings["averaging_length"]).mean()
            return x_out

#-----------------------------------------------------------------------------------------------------------

# def get_data(directory, settings):
#      #?? what purpose does pre-saved exp have? Why do we need to even call it? Is it in case the kernel needs to restart? 
#     if settings["presaved_exp"] is not None:  
#         print("Loading the pre-saved training, validation, and testing data.")

#         data_savename = "Jolan's directory to presaved training data/" + settings["presaved_exp"] + "_train.pkl"
#         with gzip.open(data_savename, "rb") as fp: #?? Rb as FP (rebuild given file path?)
#             x_train = pickle.load(fp)
#             labels+train = pickle.load(fp)

#             x_mean = pickle.load(fp)
#             x_std = pickle.load(fp)

#             lat = pickle.load(fp)
#             lon = pickle.load(fp)

#         data_savename = "Jolan's directory to presaved test data"
#         ........


# def get_labels


# def balance_data
