#!/usr/bin/env python3.10

import sys
sys.path.insert(0, '/pscratch/sd/p/plutzner/E3SM')
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

# import utils
import utils.filemethods as filemethods
import databuilder.data_loader as data_loader
from databuilder.data_loader import universaldataloader
import databuilder.data_generator as data_generator
import model.loss as module_loss
import model.metric as module_metric
import databuilder.data_loader as data_loader
from utils.filemethods import open_data_file

# from utils import utils
from shash.shash_torch import Shash
import analysis.calc_climatology as calc_climatology
from analysis import analysis_metrics
from utils.utils import filter_months
import analysis
from analysis import CRPS
from analysis import ENSO_indices_calculator
from analysis.calc_climatology import precip_regime
from analysis.ENSO_indices_calculator import idealENSOphases

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"pytorch version = {torch.__version__}")

# ------------------------------------------------------------
# Determine Ideal ENSO Phases for Column Model Runs: 

ENSO_ens1 = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/ENSO_ne30pg2_HighRes/nino.member0101.nc')
ENSO_ens2 = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/ENSO_ne30pg2_HighRes/nino.member0151.nc')
ENSO_ens3 = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/ENSO_ne30pg2_HighRes/nino.member0201.nc')

nino34_ens1 = ENSO_ens1['nino34']
nino34_ens2 = ENSO_ens2['nino34']
nino34_ens3 = ENSO_ens3['nino34']

# open ens2 input data
ens2_input_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp029_d_val.pkl')
ens2_TS = ens2_input_data['x'][..., 1]

# saveplot = '/pscratch/sd/p/plutzner/E3SM/COLUMN MODEL RUN PROJECT/ENSO_index_ens1.png'
# idealENSOphases(nino34_ens1, ens = 'ens1', percentile = 70, numberofeachphase= 1, plotfn = saveplot )

saveplot = '/pscratch/sd/p/plutzner/E3SM/COLUMN MODEL RUN PROJECT/'
idealENSOphases(nino34_ens2, ens2_TS, ens = 'ens2', percentile = 70, numberofeachphase= 1, plotfn = saveplot )

# saveplot = '/pscratch/sd/p/plutzner/E3SM/COLUMN MODEL RUN PROJECT/ENSO_index_ens3.png'
# idealENSOphases(nino34_ens3, ens = 'ens3', percentile = 70, numberofeachphase= 1, plotfn = saveplot )
