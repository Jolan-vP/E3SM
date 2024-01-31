""""
Functions for Loading data, Masking, removing seasonal cycle, isolating abnormalities

"""

import os
import pickle
import numpy as np
import xarray as xr
import importlib as imp
import file_methods as file_methods
import gzip


# open file

# specify target variables 

# store extracted data

def load_inputs(directory, settings, member):

    x_train, y_train = None, None

    for index_variable, variable in enumerate(settings["input_variables"]):
        # assert
        # assert

        # training data
        x_t = file_methods.open_netcdf(directory, settings["training_ens"], member)
        print(type(x_t))
        # x_t = maskout land ocean
        # x_t = extract region
        # x_t = expan dims
        # x_t = rolling_ave(settings, x_t)

        # validation data
        x_v = file_methods.open_netcdf(directory, settings["validation_ens"], member)
        # x_v = maskout land ocean
        # x_v = extract region
        # x_v = expan dims
        # x_v = rolling_ave(settings, x_v)

        # remove seasonal cycle
#         x_t = trend_remove_seasonalcycle(x_t)
#         x_v = trend_remove_seasonalcycle(x_v)
    return x_t, x_v

# def trend_remove_seasonalcycle(data):
#     # average per day of year

# def subtract_trend(data):
#     #try polynomial

# ------------------------------------------------------------------------
#### SDSE notes??

## SEASONAL CYCLE
# find the average per day of year 

# subtract from extracted data and store as anomalies


## TREND
# identify trend using time series analysis (Berkeley SDSE notes)

# remove trend from extracted data