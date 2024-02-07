"""
Functions for working with files

Functions: 
- open_netcdf

"""

import xarray as xr
import json
import pickle
import tensorflow as tf
# import custom_metrics

__author__ = "Jolan von Plutzner adapted from code wrtiten by Elizabeth A Barnes and Noah Diffenbaugh"
__version__ = "25 January 2024"

def open_netcdf(settings, directory, member, var):

    if settings["reduce_data"]:
        index = 10_000
        if member == "0101":
            return xr.open_dataset(directory + "ens1/bilinear/v2.LR.historical_" + member + ".eam.h1.1850-2014.nc")[var][:index, :, :]
        elif member == "0151":
            return xr.open_dataset(directory + "ens2/bilinear/v2.LR.historical_" + member + ".eam.h1.1850-2014.nc")[var][:index, :, :]
        elif member == "0201":
            return xr.open_dataset(directory + "ens3/bilinear/v2.LR.historical_" + member + ".eam.h1.1850-2014.nc")[var][:index, :, :]
        else:
            raise NotImplementedError()
    else:
        if member == "0101":
            return xr.open_dataset(directory + "ens1/bilinear/v2.LR.historical_" + member + ".eam.h1.1850-2014.nc")[var]
        elif member == "0151":
            return xr.open_dataset(directory + "ens2/bilinear/v2.LR.historical_" + member + ".eam.h1.1850-2014.nc")[var]
        elif member == "0201":
            return xr.open_dataset(directory + "ens3/bilinear/v2.LR.historical_" + member + ".eam.h1.1850-2014.nc")[var]
        else:
            raise NotImplementedError()


