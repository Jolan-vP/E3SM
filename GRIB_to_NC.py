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
from databuilder.data_loader import universaldataloader
import databuilder.data_generator as data_generator
from databuilder.data_generator import ClimateData
import model.loss as module_loss
import model.metric as module_metric
from databuilder.data_generator import multi_input_data_organizer
import databuilder.data_loader as data_loader
from utils.filemethods import open_data_file
from trainer.trainer import Trainer
from model.build_model import TorchModel
from base.base_model import BaseModel
from utils import utils
from shash.shash_torch import Shash
import analysis.calc_climatology as calc_climatology
from analysis import analysis_metrics
from utils.utils import filter_months
import analysis
from analysis import CRPS
from analysis import ENSO_indices_calculator
import analysis.analysis_metrics as analysis_metrics
from analysis.calc_climatology import precip_regime
from utils.filemethods import create_folder
from databuilder.data_generator import uniform_dist
from captum.attr import IntegratedGradients, Saliency
from XAI.captum import compute_attributions, average_attributions, visualize_average_attributions
from utils import utils
from model.metric import iqr_basic
# from analysis.nino_indices import identify_nino_phases

# CONVERT ERA5 GRIB FILES TO NETCDF: 

def grib_to_netcdf(input_file, output_file):
    """
    Convert a GRIB file to NetCDF format using xarray.
    
    Parameters:
    input_file (str): Path to the input GRIB file.
    output_file (str): Path where the output NetCDF file will be saved.
    """
    try:
        # Open the GRIB file using xarray
        ds = xr.open_dataset(input_file, engine='cfgrib')

        print(f"ds time: {ds['time'].values}")
        
        # Save the dataset to a NetCDF file
        ds.to_netcdf(output_file)
        print(f"Converted {input_file} to {output_file}")
        
    except Exception as e:
        print(f"Error converting {input_file} to NetCDF: {e}")
    
ERA5_Z500_GRIB1 = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/Z500/8449dc42c66949da451e7377e9ef9acf.grib'
ERA5_Z500_GRIB2 = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/Z500/71afb81b29c3a5d3d6f74375173667b7.grib'
ERA5_Z500_GRIB3 = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/Z500/dff6a21d88ba8a0cf6f81cc35cbd8c6d.grib'


ERA5_Z500_1940_1970 = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_025deg_Z500_1940-1970.nc'
ERA5_Z500_1971_1999 = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_025deg_Z500_1971-1999.nc'
ERA5_Z500_2000_2024 = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_025deg_Z500_2000-2024.nc'

# grib_to_netcdf(ERA5_Z500_GRIB1, ERA5_Z500_1940_1970)
# grib_to_netcdf(ERA5_Z500_GRIB2, ERA5_Z500_1971_1999)
# grib_to_netcdf(ERA5_Z500_GRIB3, ERA5_Z500_2000_2024)

# MERGE THE THREE NETCDF FILES INTO ONE:

def resample_netcdf_file(input_file, output_file):
    """
    Take daily average of 6 hourly Z500 data and merge into one NetCDF file.
    Merge multiple NetCDF files into a single NetCDF file using xarray.
    
    Parameters:
    input_files (list of str): List of paths to the input NetCDF files.
    output_file (str): Path where the merged NetCDF file will be saved.
    """

    # Open dataset
    ds = xr.open_dataset(input_file)
    # Daily average Z500:
    if 'z' in ds.data_vars:
        ds['z'] = ds['z'].resample(time='1D').mean()
    else:
        print(f"Warning: 'z' variable not found in {ds.filepath()}")

    # Save the merged dataset to a new NetCDF file
    ds.to_netcdf(output_file)
    print(f"Saved file into {output_file}")

# ERA5_Z500_1940_2024 = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_Z500_1940-2024.nc'


ERA5_Z500_1940_1970_daily = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_025deg_Z500_1940-1970_daily.nc'
ERA5_Z500_1971_1999_daily = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_025deg_Z500_1971-1999_daily.nc'
ERA5_Z500_2000_2024_daily = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_025deg_Z500_2000-2024_daily.nc'


resample_netcdf_file(ERA5_Z500_1940_1970, ERA5_Z500_1940_1970_daily)
# resample_netcdf_file(ERA5_Z500_1971_1999, ERA5_Z500_1971_1999_daily)
# resample_netcdf_file(ERA5_Z500_2000_2024, ERA5_Z500_2000_2024_daily











# def merge_netcdf_files(input_files, output_file):
#     """
#     Take daily average of 6 hourly Z500 data and merge into one NetCDF file.
#     Merge multiple NetCDF files into a single NetCDF file using xarray.
    
#     Parameters:
#     input_files (list of str): List of paths to the input NetCDF files.
#     output_file (str): Path where the merged NetCDF file will be saved.
#     """
#     try:
#         # Open multiple datasets 
#         ds_list = [xr.open_dataset(f) for f in input_files]
#         # Daily average Z500:
#         for ds in ds_list:
#             if 'Z500' in ds.data_vars:
#                 ds['Z500'] = ds['Z500'].resample(time='1D').mean()
#             else:
#                 print(f"Warning: 'Z500' variable not found in {ds.filepath()}")

#         merged_ds = xr.concat(ds_list, dim='time')
        
#         # Save the merged dataset to a new NetCDF file
#         merged_ds.to_netcdf(output_file)
#         print(f"Merged files into {output_file}")
        
#     except Exception as e:
#         print(f"Error merging files: {e}")

# ERA5_Z500_1940_2024 = '/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_Z500_1940-2024.nc'

# merge_netcdf_files([ERA5_Z500_1940_1970, ERA5_Z500_1971_1999, ERA5_Z500_2000_2024], ERA5_Z500_1940_2024)

