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

import utils
import utils.filemethods as filemethods
from utils import utils
from shash.shash_torch import Shash
from analysis import analysis_metrics
import analysis.climatology as climatology
import analysis.CRPS as CRPS

# ------------------------------------------------------------------

config = utils.get_config("exp007")
seed = config["seed_list"][0]

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

front_cutoff = config["databuilder"]["front_cutoff"] # remove front nans : 74 ENSO - two front nans before daily interpolation = 60 days, daily interpolation takes 1/2 the original time step = 15 days TOTAL = ~75
back_cutoff = config["databuilder"]["back_cutoff"]  # remove back nans : 32 ~ 1 month of nans

# -------------------------------------------------------------------

# Open Model Outputs
model_output_pred = '/Users/C830793391/Documents/Research/E3SM/saved/output/exp007_FILENAME.pkl'
output = analysis_metrics.load_pickle(model_output_pred)

# Open Target Data
target = analysis_metrics.load_pickle('file')


# Compare SHASH predictions to climatology histogram
x = np.arange(-7, 11, 0.01)

p = climatology.deriveclimatology(output, target, x, number_of_samples=17, config=config, climate_data = False)
print(p.shape)

# ----------------------------- CRPS ----------------------------------


# Compute CRPS for all predictions 
CRPS_network = CRPS.calculateCRPS(output, target, x, config, climatology = None)

# Comput CRPS for climatology
CRPS_climatology = CRPS.calculateCRPS(output, target, x, config, climatology_filename)

# Compare CRPS scores for climatology vs predictions (Is network better than climatology on average?)
CRPS_forecast, CRPS_climatology = CRPS.CRPScompare(CRPS_network, CRPS_climatology)

# Discard plot of CRPS vs IQR Percentile, CRPS vs Anomalies & true precip
sample_index = analysis_metrics.discard_plot(output, target, CRPS_forecast, CRPS_climatology, config)

# ----------------------------- ENSO ----------------------------------

# Calculate ENSO Indices: 
monthlyENSO = xr.open_dataset('/Users/C830793391/BIG_DATA/E3SM_Data/presaved/ENSO_ne30pg2_HighRes/nino.member0201.nc')
Nino34 = monthlyENSO.nino34.values

enso_indices_daily = analysis_metrics.identify_nino_phases(Nino34, threshold=0.4, window=6, front_cutoff=0, back_cutoff=0)

# Separate CRPS scores by ENSO phases 
elnino, lanina, neutral, CRPS_elnino, CRPS_lanina, CRPS_neutral = CRPS.ENSO_CRPS(enso_indices_daily, CRPS_forecast, config)

# Compare Distributions? 


# Calculate precipitation anomalies during each ENSO Phase + Plot

# Open raw target data
nc_file = xr.open_dataset('/Users/C830793391/BIG_DATA/E3SM_Data/ens3/PRECT.v2.LR.historical_0201.eam.h1.1850-2014.nc')
prect_global = nc_file.PRECT

min_lat, max_lat = config["databuilder"]["target_region"][:2]
min_lon, max_lon = config["databuilder"]["target_region"][2:]

if isinstance(prect_global, xr.DataArray):
    mask_lon = (prect_global.lon >= min_lon) & (prect_global.lon <= max_lon)
    mask_lat = (prect_global.lat >= min_lat) & (prect_global.lat <= max_lat)
    prect_regional = prect_global.where(mask_lon & mask_lat, drop=True)

prect_regional = prect_regional.mean(dim=['lat', 'lon']).values[front_cutoff + 7 : - (back_cutoff+8)]
print(f"prect_regional shape: {prect_regional.shape}")

target_raw = prect_regional * 86400 * 1000  # Convert to mm/day

print(np.mean(target_raw))
print(np.median(target_raw))
print(np.std(target_raw))

anomalies_by_ENSO_phase = analysis_metrics.anomalies_by_ENSO_phase(elnino, lanina, neutral, target, target_raw, sample_index, config)

# Spread-Skill Ratio
spread_skill_plot = analysis_metrics.spread_skill(output, target, config)