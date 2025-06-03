#!/usr/bin/env python3.10

import sys
import os
os.environ['PROJ_DATA'] = "/pscratch/sd/p/plutzner/proj_data"
import xarray as xr
import torch
import torchinfo
import random
import numpy as np
import importlib as imp
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
import json
import pickle
import gzip
import scipy
from scipy import stats
#import matplotlib.colors as mcolorsxx

# %load_ext autoreload
# %autoreload 2
import utils
import analysis.analysis_metrics as analysis_metrics
import utils.filemethods as filemethods
import databuilder.data_loader as data_loader
from databuilder.data_loader import universaldataloader
import databuilder.data_generator as data_generator
from databuilder.data_generator import ClimateData
import model.loss as module_loss
import model.metric as module_metric
from databuilder.data_generator import multi_input_data_organizer
import databuilder.data_loader as data_loader
from utils.filemethods import open_data_file

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"pytorch version = {torch.__version__}")

# Open MJO Satellite-based RMM1/RMM2L: 
mjo_1979_2023_fn = '/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/rmm.74toRealtime.txt'
mjo_1979_2023 = open_data_file(mjo_1979_2023_fn)
MJOdf = mjo_1979_2023.copy()

print(f"MJO array: {MJOdf}")
MJOdf.columns = MJOdf.columns.str.strip().str.rstrip(',')
MJOdf['date'] = pd.to_datetime(dict(year=MJOdf.year, month=MJOdf.month, day=MJOdf.day))
MJOdf.set_index('date', inplace=True)

RMM1 = MJOdf["RMM1"]
RMM2 = MJOdf["RMM2"]

# Open ERA20C MJO RMM1/RMM2
mjo_era20c_fn = '/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/WH04_RMM_ERA20C.nc'
mjo_era20c = xr.open_dataset(mjo_era20c_fn)
print(f"mjo_era20c time: {mjo_era20c.time}")

# Merge RMM1 and RMM2 into new Xarray object with "RMM1" and "RMM2" as variables: 
# ERA20C from 1900 - 1978 and MJO Sat OBS from 1979 - 2023
RMM1_1900_1978 = mjo_era20c['RMM1'].sel(time=slice('1900-01-01', '1978-12-31'))
RMM2_1900_1978 = mjo_era20c['RMM2'].sel(time=slice('1900-01-01', '1978-12-31'))

RMM1_1979_2023 = MJOdf['RMM1'].loc['1979-01-01': '2023-12-31']
RMM2_1979_2023 = MJOdf['RMM2'].loc['1979-01-01': '2023-12-31']
# convert post 1979 RMM1 and RMM2 to xarray DataArray
time_1979_2023 = pd.date_range(start='1979-01-01', end='2023-12-31', freq='D') 
RMM1_1979_2023 = xr.DataArray(RMM1_1979_2023.values, coords = [time_1979_2023], dims=['time'])
RMM2_1979_2023 = xr.DataArray(RMM2_1979_2023.values, coords = [time_1979_2023], dims=['time'])

plt.figure(figsize=(12, 6))
plt.plot(RMM1_1900_1978.time, RMM1_1900_1978, label='RMM1 ERA20C (1900-1978)', color='blue')
plt.plot(RMM1_1979_2023.time, RMM1_1979_2023, label='RMM1 MJO Sat OBS (1979-2023)', color='orange')
plt.title('RMM1 Time Series')
plt.xlabel('Time')
plt.ylabel('RMM1')
plt.legend()
plt.show()  
plt.savefig('/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/RMM1_time_series.png', dpi=300)

plt.figure(figsize=(12, 6))
plt.plot(RMM2_1900_1978.time, RMM2_1900_1978, label='RMM2 ERA20C (1900-1978)', color='blue')
plt.plot(RMM2_1979_2023.time, RMM2_1979_2023, label='RMM2 MJO Sat OBS (1979-2023)', color='orange')
plt.title('RMM2 Time Series')
plt.xlabel('Time')
plt.ylabel('RMM2')
plt.legend()
plt.show()
plt.savefig('/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/RMM2_time_series.png', dpi=300)

# Variance of RMM1 and RMM1 in 10 year chunks: 
RMM1_1900_1978_var = RMM1_1900_1978.rolling(time=3650).var()
RMM1_1979_2023_var = RMM1_1979_2023.rolling(time=3650).var()

RMM2_1900_1978_var = RMM2_1900_1978.rolling(time=3650).var()
RMM2_1979_2023_var = RMM2_1979_2023.rolling(time=3650).var()

plt.figure(figsize=(12, 6))
plt.plot(RMM1_1900_1978.time, RMM1_1900_1978_var, label='RMM1 ERA20C (1900-1978) Variance', color='blue')
plt.plot(RMM1_1979_2023.time, RMM1_1979_2023_var, label='RMM1 MJO Sat OBS (1979-2023) Variance', color='orange')
plt.title('RMM1 Variance Time Series')
plt.xlabel('Time')
plt.ylabel('RMM1 Variance')
plt.legend()
plt.show()
plt.savefig('/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/RMM1_variance_time_series.png', dpi=300)

plt.figure(figsize=(12, 6))
plt.plot(RMM2_1900_1978.time, RMM2_1900_1978_var, label='RMM2 ERA20C (1900-1978) Variance', color='blue')
plt.plot(RMM2_1979_2023.time, RMM2_1979_2023_var, label='RMM2 MJO Sat OBS (1979-2023) Variance', color='orange')
plt.title('RMM2 Variance Time Series')
plt.xlabel('Time')
plt.ylabel('RMM2 Variance')
plt.legend()
plt.show()
plt.savefig('/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/RMM2_variance_time_series.png', dpi=300)

# Merge pre and post 1979 RMM1 and pre and post 1979 RMM2 
RMM1_combined = xr.concat([RMM1_1900_1978, RMM1_1979_2023], dim='time')
RMM2_combined = xr.concat([RMM2_1900_1978, RMM2_1979_2023], dim='time')
combined_time = xr.concat([RMM1_1900_1978.time, RMM1_1979_2023.time], dim='time')
# TODO: # Combine RMM1 and RMM2 into a single xarray dataset with variables "RMM1" and "RMM2"
mjo_combined = xr.Dataset({
    'RMM1': RMM1_combined,
    'RMM2': RMM2_combined, 
    }, coords={'time': combined_time})

print(f"mjo_combined: {mjo_combined}")
print(f"mjo_combined RMM1: {mjo_combined['RMM1']}")
print(f"mjo_combined RMM2: {mjo_combined['RMM2']}")

# Save the combined dataset to a netCDF file
mjo_combined_fn = '/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_combined_ERA20C_MJO_SatOBS_1900_2023.nc'
mjo_combined.to_netcdf(mjo_combined_fn)
print(f"Saved combined MJO dataset to {mjo_combined_fn}")