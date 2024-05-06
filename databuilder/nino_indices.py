"""
Compute E3SMv2 Nino1+2, 3, 4, 3.4 Indices: 
(1) Define the lat/lon box
(2) Take area weighted average
(3) Subtract the climatology (NCL)
(4) Do 5-month running mean  (NCL)
(6) Divide by standard deviation  (NCL)
 Contact: Po-Lun.Ma@pnnl.gov

"""

import xarray as xr
import os
import pandas as pd
import numpy as np
import nc_time_axis
import cftime
from scipy import stats
from scipy import integrate
import scipy as scipy
from sklearn import preprocessing
import numpy.ma as ma

import gzip
from netCDF4 import Dataset,num2date
import time

import importlib as imp
from glob import glob
import numpy.random
from random import choices
import random as random

def NinoIndices(member, averaginglength):
    ddir = "/pscratch/sd/q/qinyi/E3SMv2_init/v2.LR.historical_" + str(member) +"/archive/atm/hist"
    edir = "/pscratch/sd/p/plma/shared/for_jolan"
    #ddir = "/Users/C830793391/BIG_DATA/E3SM_Data/ens1/bilinear OA HW1/"

    # (1) Define Lat Lon Boxes
    nino_boxbounds = np.array([[-5, 5, 190, 240],  # NINO 3+4
                        [-10, 0, 270, 280], # NINO 1+2
                        [-5, 5, 210, 270],  # NINO 3
                        [-5, 5, 160, 210]])  # NINO 4

    # Open files: Files are separated by month - must gather all files and concatenate 
    ds = xr.open_mfdataset(ddir + 'v2.LR.historical_'+ str(member) + '.eam.h0.*.nc', parallel = True)
    #ds = xr.open_mfdataset(ddir + 'v2.LR.historical_0101.eam.h1.*.nc')

    # (2) Take Area Weighted Average of TS over the lat lon box regions for EACH Index:
    TS3_4 = _extractregion(ds["TS"], nino_boxbounds[0,:])
    TS1_2 = _extractregion(ds["TS"], nino_boxbounds[1,:])
    TS3 = _extractregion(ds["TS"], nino_boxbounds[2,:])
    TS4 = _extractregion(ds["TS"], nino_boxbounds[3,:])

    temp_dict = {"Nino34": TS3_4, 
                 "Nino12": TS1_2, 
                 "Nino2": TS3, 
                 "Nino3": TS4}
    
    for key in temp_dict:
        weights = np.cos(np.deg2rad(temp_dict[key].lat))
        temp_dict[key] = temp_dict[key].weighted(weights)
        temp_dict[key] = temp_dict[key].mean(("lat", "lon"))
        
        # (3) Subtract Climatology (remove seasonal cycle)
        temp_dict[key] = trend_remove_seasonal_cycle(temp_dict[key])

        # (4) 5 Month Running Mean of TS 
        temp_dict[key] = rolling_ave(temp_dict[key], averaginglength)

        # (5) Divide by standard deviation 
        temp_dict[key] = temp_dict[key] / xr.DataArray.std(temp_dict[key])
        
        # Save .nc file output
        temp_dict[key].to_netcdf(ddir + "/nino.member" + str(member) + "." + key + ".nc")

    return temp_dict

# -------------------------------------------------------------------------------------------
def subtract_trend(x): 
        
    detrendOrder = 3

    curve = np.polynomial.polynomial.polyfit(np.arange(0, x.shape[0]), x, detrendOrder)
    trend = np.polynomial.polynomial.polyval(np.arange(0, x.shape[0]), curve) 

    try: 
        detrend = x - np.swapaxes(trend, 0, 1)
    except:
        detrend = x - trend

    return detrend 

    
def trend_remove_seasonal_cycle(da):

    if len(np.array(da.shape)) == 1: 
        print("Shape is 1")
        return da.groupby("time.dayofyear").map(subtract_trend).dropna("time")
        
    else: 
        da_copy = da.copy()

        inc = 45 # 45 degree partitions in longitude to split up the data
    
        for iloop in np.arange(0, da_copy.shape[2] // inc + 1):
            start = inc * iloop
            end = np.min([inc * (iloop + 1), da_copy.shape[2]])
            if start == end:
                break

            stacked = da[:, :, start:end].stack(z=("lat", "lon"))

            da_copy[:, :, start:end] = stacked.groupby("time.dayofyear").map(subtract_trend).unstack()
    
    return da_copy.dropna("time")


def rolling_ave(da, averaginglength):
    if len(da.shape) == 1: 
        return da.rolling(time = averaginglength).mean()
    else: 
        da_copy = da.copy()
        inc = 45
        for iloop in np.arange(0, da.shape[2] // inc + 1): 
            start = inc * iloop
            end = np.min([inc *(iloop + 1), da_copy.shape[2]])
            if start == end: 
                break

            da_copy[:, :, start:end] = da[:, :, start:end].rolling(time = averaginglength).mean()

    return da_copy
    

def _extractregion(da, boxbounds): 
    min_lat, max_lat = (boxbounds[0], boxbounds[1])
    min_lon, max_lon = (boxbounds[2], boxbounds[3])

    if isinstance(da, xr.DataArray):
        mask_lon = (da.lon >= min_lon) & (da.lon <= max_lon)
        mask_lat = (da.lat >= min_lat) & (da.lat <= max_lat)
        data_masked = da.where(mask_lon & mask_lat, drop=True)
    return (
        data_masked #,
        #data_masked["lat"].to_numpy().astype(np.float32),
        #data_masked["lon"].to_numpy().astype(np.float32),
    )