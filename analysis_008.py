import sys
import os

import analysis.ENSO_indices_calculator
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
import gzip
import scipy
from scipy import stats
#import matplotlib.colors as mcolorsxx

import utils
import utils.filemethods as filemethods
from utils import utils
from shash.shash_torch import Shash
import analysis
from analysis import analysis_metrics
import analysis.calc_climatology as calc_climatology
import analysis.CRPS as CRPS

# ------------------------------------------------------------------

config = utils.get_config("exp008")
seed = config["seed_list"][0]

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

lagtime = config["databuilder"]["lagtime"] 
smoothing_length = config["databuilder"]["averaging_length"]  

# -------------------------------------------------------------------

# Open Model Outputs
model_output_pred = '/Users/C830793391/Documents/Research/E3SM/saved/output' + str(config["expname"]) + '_output_testset.pkl'
output = analysis_metrics.load_pickle(model_output_pred)

# Open Target Data
target = xr.open_dataset('/Users/C830793391/BIG_DATA/E3SM_Data/presaved/Network Inputs/' + str(config["expname"]) + '_d_test_1850-1900.nc')
target = target["y"][lagtime:]
target = target[smoothing_length:]

# Open Climatology Data: TRAINING DATA
climatology_filename = '/Users/C830793391/BIG_DATA/E3SM_Data/presaved/Network Inputs/' + str(config["expname"]) + '_d_train_1850-1900.nc'
climatology_da = xr.open_dataset(climatology_filename)
climatology = climatology_da["y"][lagtime:]
climatology = climatology[smoothing_length:]

# Compare SHASH predictions to climatology histogram
x = np.arange(-15, 15, 0.01)

p = calc_climatology.deriveclimatology(output, climatology, x, number_of_samples=50, config=config, climate_data = False)

# ----------------------------- CRPS ----------------------------------

# Comput CRPS for climatology
CRPS_climatology = CRPS.calculateCRPS(output, target, x, config, climatology)

# Compute CRPS for all predictions 
CRPS_network = CRPS.calculateCRPS(output, target, x, config, climatology = None)

analysis_metrics.save_pickle(CRPS_climatology, str(config["output_dir"]) + "/" + str(config["expname"]) + "/CRPS_climatology_values.pkl")
analysis_metrics.save_pickle(CRPS_network, str(config["output_dir"]) + "/" + str(config["expname"]) + "/CRPS_network_values.pkl")

CRPS_climatology = analysis_metrics.load_pickle(str(config["output_dir"]) + "/" + str(config["expname"]) + "/CRPS_climatology_values.pkl")
CRPS_network = analysis_metrics.load_pickle(str(config["output_dir"]) + "/" + str(config["expname"]) + "/CRPS_network_values.pkl")

# Compare CRPS scores for climatology vs predictions (Is network better than climatology on average?)
CRPS.CRPScompare(CRPS_network, CRPS_climatology)

# ----------------------------- ENSO ----------------------------------

# Calculate ENSO Indices: 
monthlyENSO = xr.open_dataset('/Users/C830793391/BIG_DATA/E3SM_Data/presaved/ENSO_ne30pg2_HighRes/nino.member0201.nc')
Nino34 = monthlyENSO.nino34
# select a slice of only certain years
Nino34 = Nino34.sel(time=slice ( str(config["databuilder"]["input_years"][0]) + '-01-01', str(config["databuilder"]["input_years"][1]) + '-12-31'))
Nino34 = Nino34.values

enso_indices_daily = analysis.ENSO_indices_calculator.identify_nino_phases(Nino34, threshold=0.4, window=6, lagtime = lagtime, smoothing_length = smoothing_length)

# Separate CRPS scores by ENSO phases 
elnino, lanina, neutral, CRPS_elnino, CRPS_lanina, CRPS_neutral = analysis.ENSO_indices_calculator.ENSO_CRPS(enso_indices_daily, CRPS_network, climatology, x, output, config)

# Compare Distributions? 
p = calc_climatology.deriveclimatology(output, climatology, x, number_of_samples=50, config=config, climate_data = False)

# Calculate precipitation anomalies during each ENSO Phase + Plot

# Open raw target data
nc_file = xr.open_dataset('/Users/C830793391/BIG_DATA/E3SM_Data/ens3/PRECT.v2.LR.historical_0201.eam.h1.1850-2014.nc')
prect_global = nc_file.PRECT.sel(time = slice(str(config["databuilder"]["input_years"][0]) + '-01-01', str(config["databuilder"]["input_years"][1])))

min_lat, max_lat = config["databuilder"]["target_region"][:2]
min_lon, max_lon = config["databuilder"]["target_region"][2:]

if isinstance(prect_global, xr.DataArray):
    mask_lon = (prect_global.lon >= min_lon) & (prect_global.lon <= max_lon)
    mask_lat = (prect_global.lat >= min_lat) & (prect_global.lat <= max_lat)
    prect_regional = prect_global.where(mask_lon & mask_lat, drop=True)

prect_regional = prect_regional.mean(dim=['lat', 'lon']).values[lagtime:]
prect_regional = prect_regional[smoothing_length:]

print(f"raw target prect_regional shape: {prect_regional.shape}")

target_raw = prect_regional * 86400 * 1000  # Convert to mm/day

print(f"mean raw target: {np.mean(target_raw)}")
print(f"median raw target: {np.median(target_raw)}")
print(f"std raw target: {np.std(target_raw)}")

# Discard plot of CRPS vs IQR Percentile, CRPS vs Anomalies & true precip
sample_index = analysis_metrics.discard_plot(output, target_raw, CRPS_network, CRPS_climatology, config, target_type = 'raw')

anomalies_by_ENSO_phase = analysis_metrics.anomalies_by_ENSO_phase(elnino, lanina, neutral, target, target_raw, sample_index, config)

# Spread-Skill Ratio
spread_skill_plot = analysis_metrics.spread_skill(output, target, config)