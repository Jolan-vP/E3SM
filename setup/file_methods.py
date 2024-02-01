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

def open_netcdf(directory, settings, member): #add Settings to this function as they develop!
    if member == "0101":
        return xr.open_dataset(directory + "v2.LR.historical_0101.eam.h1.1850-2014.nc")
    if member == "0151":
        return xr.open_dataset(directory + "v2.LR.historical_0151.eam.h1.1850-2014.nc")
    if member == "0201":
        return xr.open_dataset(directory + "v2.LR.historical_0201.eam.h1.1850-2014.nc")
    else:
        raise NotImplementedError
    


    
    # Should I develop a less extensive version of the code that just does what I need at the moment, and add functionality as I go?
    # how often do we want to reduce data? is this used?