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

# https://github.com/victoresque/pytorch-template/tree/master

# Try regridding ERA5 data to E3SM grid: 
# open exp083 input test data (ERA5 obs)
ERA5_full_data = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_1x1_input_vars_1940-2023.nc')

# open exp075 input test data (E3SM model)
exp075_input = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp075_trimmed_test_dat.nc')

# compare grids: 
print(f"ERA5_full_data grid: {ERA5_full_data['tp'].shape}, {ERA5_full_data['tp'].lat.shape}, {ERA5_full_data['tp'].lon.shape}")
print(f"exp075_input grid: {exp075_input['x'].shape}, {exp075_input['x'].lat.shape}, {exp075_input['x'].lon.shape}")


# regrid ERA5_full_data to exp075_input grid
# add cyclic padding to ERA5_full_data
cyclic_padding = ERA5_full_data.isel(lon=-1).copy()
cyclic = cyclic_padding.assign_coords(lon=ERA5_full_data.lon[-1] + 360)

ERA5_cyclic = xr.concat([cyclic, ERA5_full_data], dim='lon')

ERA5_full_data_regrid = ERA5_cyclic.interp(
    lat=exp075_input['x'].lat,
    lon=exp075_input['x'].lon,
    method="linear"
)

print("NaNs by lon:", ERA5_full_data_regrid.isnull().sum(dim="lat").max())
print("NaNs by lat:", ERA5_full_data_regrid.isnull().sum(dim="lon").max())

# check if regridding worked
print(f"ERA5_full_data_regrid grid: {ERA5_full_data_regrid['tp'].shape}, {ERA5_full_data_regrid['tp'].lat.shape}, {ERA5_full_data_regrid['tp'].lon.shape}")

# save regridded dataset
ERA5_full_data_regrid.to_netcdf('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_1x1_input_vars_1940-2023_regrid.nc')