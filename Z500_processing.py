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

# ----CONFIG AND CLASS SETUP----------------------------------------------
config = utils.get_config("exp069")
print(config["expname"])
seed = config["seed_list"][0]

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

imp.reload(utils)
imp.reload(filemethods)
imp.reload(data_generator)
imp.reload(data_loader)

data = ClimateData(
    config["databuilder"], 
    expname = config["expname"],
    seed=seed,
    data_dir = config["perlmutter_data_dir"], 
    figure_dir=config["perlmutter_figure_dir"],
    target_only = False, 
    fetch=False,
    verbose=False
)

# ---- DATA PROCESSING ----------------------------------------------------------------

# d_train, d_val, d_test = data.fetch_data()

# # check data output
# print(d_train)
# print(d_train['y'].shape)
# print(d_train['y'])

s_dict_savename1 = config["perlmutter_data_dir"] + "Z500_processed_anomalies.v2.LR.historical_0101.eam.h1.1850-2014.pkl"
s_dict_savename2 = config["perlmutter_data_dir"] + "Z500_processed_anomalies.v2.LR.historical_0151.eam.h1.1850-2014.pkl"
s_dict_savename3 = config["perlmutter_data_dir"] + "Z500_processed_anomalies.v2.LR.historical_0201.eam.h1.1850-2014.pkl"

# save the data as pickle: 
# analysis_metrics.save_pickle(d_train, s_dict_savename1)
# analysis_metrics.save_pickle(d_val, s_dict_savename2)
# analysis_metrics.save_pickle(d_test, s_dict_savename3)

# open processed data files:
Z500_train = analysis_metrics.load_pickle(s_dict_savename1)
Z500_val = analysis_metrics.load_pickle(s_dict_savename2)
Z500_test = analysis_metrics.load_pickle(s_dict_savename3)

# Z500_train = xr.open_dataset(s_dict_savename1)
# Z500_val = xr.open_dataset(s_dict_savename2)
# Z500_test = xr.open_dataset(s_dict_savename3)

trimmed_trainfn = config["perlmutter_data_dir"] + "Z500_trimmed_processed_anomalies.v2.LR.historical_0101.eam.h1.1850-2014.nc"
trimmed_valfn = config["perlmutter_data_dir"] + "Z500_trimmed_processed_anomalies.v2.LR.historical_0151.eam.h1.1850-2014.nc"
trimmed_testfn = config["perlmutter_data_dir"] + "Z500_trimmed_processed_anomalies.v2.LR.historical_0201.eam.h1.1850-2014.nc"

Z500_train_trimmed = universaldataloader(s_dict_savename1, config, target_only = False, repackage = True)
Z500_train_trimmed.to_netcdf(trimmed_trainfn)

Z500_val_trimmed = universaldataloader(s_dict_savename2, config, target_only = False, repackage = True)
Z500_val_trimmed.to_netcdf(trimmed_valfn)

Z500_test_trimmed = universaldataloader(s_dict_savename3, config, target_only = False, repackage = True)
Z500_test_trimmed.to_netcdf(trimmed_testfn)