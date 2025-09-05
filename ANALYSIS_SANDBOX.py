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
from databuilder.data_generator import uniform_dist, adjust_data_split
import captum.attr
from captum.attr import IntegratedGradients, Saliency
from XAI.captum import compute_attributions, average_attributions, visualize_average_attributions
from utils import utils
from model.metric import iqr_basic
from analysis import combined_experiment_analytics as cea
# from analysis.nino_indices import identify_nino_phases

# print(f"python version = {sys.version}")
# print(f"numpy version = {np.__version__}")
# print(f"xarray version = {xr.__version__}")
# print(f"pytorch version = {torch.__version__}")

# ------------------------------------------------------------------------------------

# keyword = "E3SM-short_E3SM-long_OBS"

# keyword = "OBS-OBS_E3SM-short_OBS"

# keyword = "E3SM-long_E3SM-short_E3SM"

# keyword = "OBS-OBS_OBS-E3SM"

exps = {
    "OBS(OBS)": ["exp173", "exp174", "exp175", "exp176", "exp177", "exp178", "exp179", "exp180", "exp181", "exp182", "exp183", "exp184"],
    "E3SM-short(OBS)": ["exp189", "exp195", "exp196", "exp197", "exp198", "exp199"]
    # "E3SM-long(OBS)": ["exp186", "exp187", "exp188", "exp203", "exp204", "exp205"]
    # "E3SM-short(E3SM)": ["exp185", "exp190", "exp191", "exp192", "exp193", "exp194"],
    # "E3SM-long(E3SM)": ["exp154", "exp157", "exp158", "exp200", "exp201", "exp202"], 
    # "OBS(E3SM)": ["exp206", "exp207", "exp208", "exp209", "exp210", "exp211", "exp212", "exp213", "exp214", "exp215", "exp216", "exp217"]
}

# DISCARD PLOTS: # ------------------------------------------------------------------------------------  

# cea.combined_success_discard(exps, keyword = "OBS-OBS_OBS-E3SM_E3SM-short-OBS")

# cea.combined_CRPS_IQR_discard(exps, keyword = "OBS-OBS_OBS-E3SM_E3SM-short-OBS")

# cea.IQR_distributions(exps, keyword = "OBS-OBS_OBS-E3SM_E3SM-short-OBS")
# # 
# COMPOSITE MAPPING: # ------------------------------------------------------------------------------------  

# Individual experiment plots: 
# inde_exps = {
#     "E3SM-short(OBS)": "exp196",
#     "OBS(OBS)": "exp181"
#     # "E3SM-long(OBS)": ["exp186", "exp187", "exp188"]
# }

# cea.composite_inputmap_target(inde_exps, confidence_level= 20, keyword = "OBS_OBS")

cea.COMPARE_composite_inputmap_target(exps, confidence_level_low= 20, confidence_level_high= 40, keyword = "OBS-OBS_E3SM-short_OBS")

## XAI / CAPTUM# ------------------------------------------------------------------------------------  

# SELECT ONE EXP TYPE AT A TIME: 
cea.XAI_confidence_compositing(exps, confidence_level_low = 20, confidence_level_high = 40, xai_method = 'integrated_gradients', keyword = "E3SM-short_OBS")

