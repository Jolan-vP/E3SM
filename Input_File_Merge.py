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
from cftime import DatetimeNoLeap
from datetime import datetime
from sklearn.metrics import mean_squared_error
#import matplotlib.colors as mcolorsxx

# %load_ext autoreload
# %autoreload 2
import utils
import utils.filemethods as filemethods
import databuilder.data_loader as data_loader
from utils.filemethods import open_data_file
from utils import utils


"""
(1) Open 0101 and 0151 data files containing Z500, PRECT, TS
(2) Check the files for correct magnitudes and variables
(3) Shift 0101 data by -60225 days to 1685-1849
(4) Merge the two datasets into a single dataset with continuous time (1685 - 2014)
(5) Save the merged dataset to a new NetCDF file
(6) Print the shape of the merged dataset
(7) Print the time range of the merged dataset
(8) Print the variable names in the merged dataset

"""

# def merge_input_files(file_0101, file_0151, input_file, output_file):
    # # Open the 0101 data file
    # ds_0101 = open_data_file(file_0101)

    # # Select only 1850-2014 (not 2015)
    # ds_0101 = ds_0101.sel(time=slice('1850-01-01', '2014-12-31'))
    
    # # Open the 0151 data file
    # ds_0151 = open_data_file(file_0151)

    # # Check the variables in both datasets
    # print("Variables in 0101 dataset:", list(ds_0101.data_vars))
    # print("Variables in 0151 dataset:", list(ds_0151.data_vars))

    # # Shift the time in the 0101 dataset by -60225 days
    # # Datetime no leap is used to avoid leap years in the time series
    # print("Time range of 0101 dataset before shift:", ds_0101['time'].values.min(), "to", ds_0101['time'].values.max())
    # ds_0101['time'] = xr.cftime_range(start='1685-01-01', 
    #                                    end='1849-12-31', 
    #                                    freq='D',
    #                                    calendar='noleap')

    # print("Time range of 0101 dataset after shift:", ds_0101['time'].values.min(), "to", ds_0101['time'].values.max())
     
    # # Create the merged time coordinate first
    # merged_time = xr.concat([ds_0101['time'], ds_0151['time']], dim='time')

    # print(f"Merged Time: {merged_time}")
    # print(f"Merged Time middle: {merged_time.isel(time = slice(60220, 60230))}")
    
    # # Concatenate each variable individually
    # merged_vars = {}
    # vars = ["PRECT", "TS", "Z500"]

    # for var in vars:
    #     print(f"Concatenating variable: {var}")
    #     merged_vars[var] = xr.concat([ds_0101[var], ds_0151[var]], dim='time')
    
    # # Create the merged dataset
    # ds_merged = xr.Dataset(merged_vars, coords={'time': merged_time, 
    #                                             'lat': ds_0101['lat'], 
    #                                             'lon': ds_0101['lon']})
    
    # # Copy attributes
    # ds_merged.attrs = ds_0151.attrs  # Use 0151 attributes as base
    
    # # # Save with compression
    # # encoding = {var: {'zlib': True, 'complevel': 6} for var in ds_merged.data_vars}
    # # ds_merged.to_netcdf(output_file, encoding=encoding)


    # # Merge the two datasets along the time dimension
    # ds_merged = xr.concat([ds_0101, ds_0151], dim='time')

    # # Print the shape of the merged dataset
    # print("Shape of merged dataset:", ds_merged.sizes)

    # # Print the time range of the merged dataset
    # print("Time range of merged dataset:", ds_merged['time'].values.min(), "to", ds_merged['time'].values.max())

    # # Print the variable names in the merged dataset
    # print("Variables in merged dataset:", list(ds_merged.data_vars))

    # # Convert m/s precip to mm/day (*86400000)
    # if 'PRECT' in ds_merged.data_vars:
    #     ds_merged['PRECT'] = ds_merged['PRECT'] * 86400000
     
    # print(f"Conversion of PRECT to mm/day: {ds_merged['PRECT'].isel(time = 450).values}" )

    # # eliminate time bounds variable - it is not needed
    # if 'time_bnds' in ds_merged.data_vars:
    #     ds_merged = ds_merged.drop_vars('time_bnds')

    # # Plot variables across entire time coordinate to check for correct magnitudes
    # plt.figure(figsize=(10, 5))
    # for var in vars:
    #     ds_merged[var].sel(lon = 100.5, lat = 40.5).plot()
    #     plt.title(f"Variable: {var}")
    #     plt.xlabel("Time")
    #     plt.ylabel(var)
    #     plt.show()
    
    # plt.savefig("/pscratch/sd/p/plutzner/E3SM/bigdata/merged_input_dataset_checkplot.png", dpi=300, format='png')

    # print("Forcing Compute")
    # ds_merged.compute()

    # print("Saving merged dataset to file...")
    # # Save the merged dataset to a new NetCDF file
    # ds_merged.to_netcdf(output_file)

    # ------------------------------------------------------------

#     # # Open ds_merged: 
#     ds_merged = open_data_file(input_file)

#     # eliminate bounds variables and dimension - they are not needed
#     bounds_vars = ['time_bnds', 'lon_bnds', 'lat_bnds']
#     existing_bounds = [var for var in bounds_vars if var in ds_merged.data_vars]
    
#     print("dropping vars")
#     if existing_bounds:
#         ds_merged = ds_merged.drop_vars(existing_bounds)
    
#     print("dropping bnds")
#     # Drop the bnds dimension if it exists
#     if 'bnds' in ds_merged.dims:
#         ds_merged = ds_merged.drop_dims('bnds')

#     # Reorder dimensions so lat comes before lon
#     ds_merged = ds_merged.transpose('time', 'lat', 'lon')
    
#     print(f"Shape of merged dataset: {ds_merged.sizes}")
#     print(f"merged vars: {list(ds_merged.data_vars)}")
#     print(f"merged coords: {list(ds_merged.coords)}")

#      # # Save the merged dataset to a new NetCDF file
#     ds_merged.to_netcdf(output_file) 

# merge_input_files(
#     file_0101='/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.P_T_Z5.v2.LR.historical_0101.eam.h1.1850-2014.nc',
#     file_0151='/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.P_T_Z5.v2.LR.historical_0151.eam.h1.1850-2014.nc',
#     input_file = '/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.P_T_Z5.v2.LR.historical_merged.eam.h1.1685-2014.nc',
#     output_file='/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.P_T_Z5.v2.LR.historical_merged_ens1ens2.eam.h1.1685-2014.nc'
# )

# merged_file = '/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.P_T_Z5.v2.LR.historical_merged.eam.h1.1685-2014.nc'

# # Open ds_merged: 
# ds_merged = open_data_file(merged_file)

# print(f"Shape of merged dataset: {ds_merged.sizes}")
# print(f"merged vars: {list(ds_merged.data_vars)}")
# print(f"merged coords: {list(ds_merged.coords)}")
# print(f"max precip: {ds_merged['PRECT'].max().values}")
# print(f"min precip: {ds_merged['PRECT'].min().values}")
# print(f"max TS: {ds_merged['TS'].max().values}")
# print(f"min TS: {ds_merged['TS'].min().values}")
# print(f"max Z500: {ds_merged['Z500'].max().values}")
# print(f"min Z500: {ds_merged['Z500'].min().values}")



# ------- 0201 dataset: 

P_0201 = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.P.v2.LR.historical_0201.eam.h1.1850-2014_precip_mmday.nc')
Z5_TZ_0201 = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.T_Z5.v2.LR.historical_0201.eam.h1.1850-2014.nc')

# Check Precip : 

print(f"Shape of P_0201 dataset: {P_0201.sizes}")
print(f"Magnitude of precip: {P_0201['PRECT'].max().values} to {P_0201['PRECT'].min().values}")

print(f"shape of Z5_TZ_0201 dataset: {Z5_TZ_0201.sizes} ")
print(f"magnitude of Z500: {Z5_TZ_0201['Z500'].max().values} to {Z5_TZ_0201['Z500'].min().values}")

# Check if the datasets have the same time coordinate
if np.array_equal(P_0201['time'].values, Z5_TZ_0201['time'].values):
    print("Time coordinates match.")
else:
    print("Time coordinates do not match.")

# Merge two datasets, adding the variables into the same xarray object: 
ds_merged_0201 = xr.merge([P_0201, Z5_TZ_0201])

# eliminate bounds variables and dimension - they are not needed
bounds_vars = ['time_bnds', 'lon_bnds', 'lat_bnds']
existing_bounds = [var for var in bounds_vars if var in ds_merged_0201.data_vars]

print("dropping vars")
if existing_bounds:
    ds_merged_0201 = ds_merged_0201.drop_vars(existing_bounds)

print("dropping bnds")
# Drop the bnds dimension if it exists
if 'bnds' in ds_merged_0201.dims:
    ds_merged_0201 = ds_merged_0201.drop_dims('bnds')

# Reorder dimensions so lat comes before lon
ds_merged_0201 = ds_merged_0201.transpose('time', 'lat', 'lon')

# Print the shape of the merged dataset
print("Shape of merged 0201 dataset:", ds_merged_0201.sizes)

# Print the time range of the merged dataset
print("Time range of merged 0201 dataset:", ds_merged_0201['time'].values.min(), "to", ds_merged_0201['time'].values.max())

# Print the variable names in the merged dataset
print("Variables in merged 0201 dataset:", list(ds_merged_0201.data_vars))

# Save dataset: 
ds_merged_0201.to_netcdf('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.P_T_Z5.v2.LR.historical_0201.eam.h1.1850-2014_precip_mmday.nc')


ds_merged_saved_0201 = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.P_T_Z5.v2.LR.historical_0201.eam.h1.1850-2014_precip_mmday.nc')

plt.figure(figsize=(10, 5))
ds_merged_saved_0201['PRECT'].sel(lon = 100.5, lat = 40.5).plot()
plt.title(f"Variable: PRECT")
plt.xlabel("Time")
plt.ylabel("PRECT (mm/day)")
plt.savefig("/pscratch/sd/p/plutzner/E3SM/bigdata/merged_0201_input_dataset_checkplot.png", dpi=300, format='png')