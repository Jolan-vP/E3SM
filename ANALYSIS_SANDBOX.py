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

# ---- Z500 Regime Analysis ----
E3SM_baseline = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.P_T_Z5.v2.LR.historical_0101.eam.h1.1850-2014_precip_mmday.nc')
E3SM_Z500 = E3SM_baseline.sel(time = slice('1981-01-01', '2010-12-31'))
E3SM_Z500 = E3SM_Z500.Z500
# print(f"shape E3SM_Z500 = {E3SM_Z500.shape}")

ERA5_baseline = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_raw_climatology_TP_SKT_Z_1981-2010.nc')
ERA5_Z500 = ERA5_baseline['z'] / 9.81
# print(f"shape ERA5_Z500 = {ERA5_Z500.shape}")

# analysis.calc_climatology.Z500_regime(E3SM_Z500, ERA5_Z500) 
# ------------------------------------------------------------------------------------

exps = {
    # "OBS(OBS)":  ["exp173", "exp174", "exp175", "exp176", "exp177", "exp178", "exp179", "exp180", "exp181", "exp182", "exp183", "exp184"],
    # "E3SM(OBS)": ["exp189", "exp195", "exp196", "exp197", "exp198", "exp199", "exp263", "exp264", "exp265", "exp266", "exp267", "exp268"],
    # "E3SM(E3SM)":["exp185", "exp190", "exp191", "exp192", "exp193", "exp194", "exp257", "exp258", "exp259", "exp260", "exp261", "exp262"],
    # "OBS(E3SM)": ["exp206", "exp207", "exp208", "exp209", "exp210", "exp211", "exp212", "exp213", "exp214", "exp215", "exp216", "exp217"] 
    
    # "E3SM-long(OBS)": ["exp186", "exp187", "exp188", "exp203", "exp204", "exp205"], 
    # "E3SM-long(E3SM)": ["exp154", "exp157", "exp158", "exp200", "exp201", "exp202"],
    
# SCALED TARGET: 
    "OBS(OBS)sv":   ["exp218", "exp220", "exp221", "exp226", "exp227", "exp228", "exp229", "exp230", "exp231", "exp232", "exp233", "exp234"],
    "E3SM(OBS)sv":  ["exp222", "exp246", "exp247", "exp248", "exp249", "exp250", "exp251", "exp252", "exp253", "exp254", "exp255", "exp256"],
    "E3SM(E3SM)sv": ["exp219", "exp235", "exp236", "exp237", "exp238", "exp239", "exp240", "exp241", "exp242", "exp243", "exp244", "exp245"],
    "OBS(E3SM)sv":  ["exp223", "exp224", "exp225", "exp269", "exp270", "exp271", "exp272", "exp273", "exp274", "exp275", "exp276", "exp277"]

    # "E3SM-long(E3SM)sv" : ["exp278", "exp279", "exp280", "exp281", "exp282", "exp283", "exp284", "exp285", "exp286", "exp287", "exp288", "exp289"]
}

# DISCARD PLOTS: # ------------------------------------------------------------------------------------  

# cea.combined_success_discard(exps, keyword = "all_exps_no_scaling") # SCALED IQR OR SCALED TARGET?

# cea.combined_CRPS_IQR_discard(exps, keyword = "all_exps_no_scaling") # SCALED IQR OR SCALED TARGET?
# cea.combined_CRPS_IQR_discard(exps, keyword = "all_exps_scaled_target") # SCALED IQR OR SCALED TARGET?

 # SCALED IQR: 
# cea.CRPS_discard_scaled_IQR(exps, keyword = "all_exps_scaled_IQR")
# cea.combined_success_discard_scaled_IQR(exps, keyword = "all_exps_scaled_IQR")

# cea.variance_analysis_success_plot(exps, keyword = "all_exps_scaled_IQR")

# cea.IQR_distributions_STEP_hist(exps, keyword = "ID_OOD_E3SM_OBS")
# cea.IQR_distributions_STACKED_hist(exps, keyword = "ID_OOD_E3SM_OBS")
# cea.IQR_distributions_STACKED_hist(exps, keyword = "E3SM_long_short")

# COMPOSITE MAPPING: # ------------------------------------------------------------------------------------  

# Individual experiment plots: 
# inde_exps = {
#     "E3SM-short(OBS)": "exp196",
#     "OBS(OBS)": "exp181"
#     # "E3SM-long(OBS)": ["exp186", "exp187", "exp188"]
# }
# cea.composite_inputmap_target(inde_exps, confidence_level= 20, keyword = "OBS_OBS")
# cea.COMPARE_composite_inputmap_target(exps, confidence_level_low= 20, confidence_level_high= 40, keyword = "OBS-OBS_E3SM-short_OBS")

## XAI / CAPTUM# ------------------------------------------------------------------------------------  

# cea.XAI_confidence_compositing(exps, confidence_level_low = 20, confidence_level_high = 40, xai_method = 'integrated_gradients', keyword = "E3SM-short_OBS")

## TELECONNECTIONS ANALYSIS # ------------------------------------------------------------------------------------

# cea.m2m_sample_transfer(exps, selection_method = 'simple_iqr_percentage', confidence = 20, keyword = "scaled_target")

# cea.IQR_only_analysis(exps, selection_method = 'simple_iqr_percentage', confidence = 20, keyword = "scaled_target")

# cea.teleconnection_bias_analysis(exps, confidence_level_low = 100, confidence_level_high = 0, keyword = "OBS-E3SM_scaled_target")
# cea.anom_var_distributions(exps, keyword = "E3SM-short_OBS")
# cea.m2m_sample_transfer(exps, selection_method = 'scaled_iqr_by_percentage', confidence = 10, keyword = "OBS-OBS_scaled_IQR")
# cea.m2m_sample_transfer(exps, selection_method = 'scaled_iqr_by_percentage', confidence = 10, keyword = "E3SM-long_E3SM")
# cea.m2m_sample_transfer_individual(exps, selection_method = 'high_low_crps', confidence = 50, keyword = "OBS-OBS")
# cea.m2m_sample_transfer(exps, selection_method = 'no_scaling', confidence = 20, keyword = "no_scaling")
# cea.m2m_sample_transfer(exps, selection_method = 'simple_iqr_percentage', confidence = 20, keyword = "scaled_target")


# EPISTEMIC UNCERTAINTY ANALYSIS # ------------------------------------------------------------------------------------

# SCALED TARGET
# cea.variance_OIQR_analysis(exps, scale_target = True,  scale_IQR = False, keyword = "scaled_target") 

cea.dual_filtering_var_IQR(exps, keyword = None)

# cea.variance_OM_analysis(exps, scale_target = False, scale_IQR = False, keyword = "scaled_target") 
# # NO SCALING
# cea.variance_OM_analysis(exps, scale_target = False, scale_IQR = False, keyword = "no_scaling")
# cea.variance_OIQR_analysis(exps, scale_target = False,  scale_IQR = False, keyword = "no_scaling")
# SCALED IQR
# cea.variance_OM_analysis(exps, scale_target = False, scale_IQR = True, keyword = "scaled_IQR") 
# cea.variance_OIQR_analysis(exps, scale_target = False, scale_IQR = True, keyword = "scaled_IQR") 