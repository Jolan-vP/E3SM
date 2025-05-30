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
from databuilder.data_generator import uniform_dist
from captum.attr import IntegratedGradients, Saliency
from XAI.captum import compute_attributions, average_attributions, visualize_average_attributions
from utils import utils
from model.metric import iqr_basic
# from analysis.nino_indices import identify_nino_phases

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"pytorch version = {torch.__version__}")

# https://github.com/victoresque/pytorch-template/tree/master

# ----CONFIG AND CLASS SETUP----------------------------------------------
config = utils.get_config("exp079")
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

if config["arch"]["type"] == "basicnn":
    target_treatment = True
else:
    target_treatment = False

data = ClimateData(
    config["databuilder"], 
    expname = config["expname"],
    seed=seed,
    data_dir = config["perlmutter_data_dir"], 
    figure_dir=config["perlmutter_figure_dir"],
    target_only = target_treatment, 
    fetch=False,
    verbose=False
)

# Create directories for the experiment
output_folder_name = str(config["perlmutter_output_dir"]) + str(config["expname"])
figure_folder_name = str(config["perlmutter_figure_dir"]) + str(config["expname"])

create_folder(output_folder_name)
create_folder(figure_folder_name)

# ---- DATA PROCESSING ----------------------------------------------------------------
# # # Check if input data is being processed from scratch or if it is being loaded from a previous experiment

if config["input_data"] == "None": # Then input data must be processed FROM SCRATCH
    print("Processing input data from scratch")
    print(f"This is a {config['arch']['type']} model")

    d_train, d_val, d_test = data.fetch_data()

    # ---- FOR SIMPLE INPUTS ONLY : ----------------------------------------------
    if config["arch"]["type"] == "basicnn":
        # print(d_train['y'].shape)
        target_savename1 = str(config["perlmutter_data_dir"]) + str(config["expname"]) + "_d_train_TARGET_1850-2014.pkl"
        with gzip.open(target_savename1, "wb") as fp:
            pickle.dump(d_train, fp)

        target_savename2 = str(config["perlmutter_data_dir"]) + str(config["expname"]) + "_d_val_TARGET_1850-2014.pkl"
        with gzip.open(target_savename2, "wb") as fp:
            pickle.dump(d_val, fp)

        target_savename3 = str(config["perlmutter_data_dir"]) + str(config["expname"]) + "_d_test_TARGET_1850-2014.pkl"
        with gzip.open(target_savename3, "wb") as fp:
            pickle.dump(d_test, fp)

        d_train, d_val, d_test = multi_input_data_organizer(config, target_savename1, target_savename2, target_savename3, MJO = True, ENSO = True, other = False)
          
        # confirm metadata is stored for both input and target
        print(f" s_dict_train INPUT time {d_train['x'].time}")
        print(f" s_dict_train TARGET time {d_train['y'].time}")

        # confirm input structure: 
        print(f"input shape: {d_train['x'].shape}")
    else: 
        pass

#     # Save full input data for the experiment: ----------------------------------
    s_dict_savename1 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_d_train.pkl"
    s_dict_savename2 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_d_val.pkl"
    s_dict_savename3 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_d_test.pkl"

    with gzip.open(s_dict_savename1, "wb") as fp:
        pickle.dump(d_train, fp)

    with gzip.open(s_dict_savename2, "wb") as fp:
        pickle.dump(d_val, fp)

    with gzip.open(s_dict_savename3, "wb") as fp:
        pickle.dump(d_test, fp)

    # Trim input data: Lead/lag, month selection ----------------------------------
    trimmed_trainfn = config["perlmutter_inputs_dir"] + str(config["expname"]) + "_trimmed_" + "train_dat.nc"
    trimmed_valfn = config["perlmutter_inputs_dir"] + str(config["expname"]) + "_trimmed_" + "val_dat.nc"
    trimmed_testfn = config["perlmutter_inputs_dir"] + str(config["expname"]) + "_trimmed_" + "test_dat.nc"

    train_dat_trimmed = universaldataloader(s_dict_savename1, config, target_only = False, repackage = True)
    train_dat_trimmed.to_netcdf(trimmed_trainfn)
    print(f"Data saved to {trimmed_trainfn}")

    val_dat_trimmed = universaldataloader(s_dict_savename2, config, target_only = False, repackage = True)
    val_dat_trimmed.to_netcdf(trimmed_valfn)
    print(f"Data saved to {trimmed_valfn}")

    test_dat_trimmed = universaldataloader(s_dict_savename3, config, target_only = False, repackage = True)
    test_dat_trimmed.to_netcdf(trimmed_testfn)
    print(f"Data saved to {trimmed_testfn}")

elif "exp" in config["input_data"]: 
    trimmed_trainfn = str(config["perlmutter_inputs_dir"]) + str(config["input_data"]) + "_trimmed_" + "train_dat.nc"
    trimmed_valfn = str(config["perlmutter_inputs_dir"]) + str(config["input_data"]) + "_trimmed_" + "val_dat.nc"
    trimmed_testfn = str(config["perlmutter_inputs_dir"]) + str(config["input_data"]) + "_trimmed_" + "test_dat.nc"
    
# # # --- Setup the Data for Training ---------------------------------------------
lagtime = config["databuilder"]["lagtime"] 
smoothing_length = config["databuilder"]["averaging_length"]

trainset = data_loader.CustomData(trimmed_trainfn, config, which_set = 'training')
valset = data_loader.CustomData(trimmed_valfn, config, which_set = 'validation')
testset = data_loader.CustomData(trimmed_testfn, config, which_set = 'testing')

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=config["data_loader"]["batch_size"],
    shuffle=True,
    drop_last=False
)

val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=config["data_loader"]["batch_size"],
    shuffle=False,
    drop_last=False
)

# --- Setup the Model ----------------------------------------------------

# # Check if model already exists: 
# if os.path.exists(str(config["perlmutter_model_dir"]) + str(config["expname"]) + '.pth'):
#     print("Model already exists")
#     response = input("Would you like to load the model? (yes) \n or retrain from epoch 0 (no): ")

#     if response == "yes":
#         path = str(config["perlmutter_model_dir"]) + str(config["expname"]) + '.pth'
#         load_model_dict = torch.load(path)
#         state_dict = load_model_dict["model_state_dict"]
#         std_mean = load_model_dict["training_std_mean"]
#         model = TorchModel(
#             config=config["arch"],
#             target_mean=std_mean["trainset_target_mean"],
#             target_std=std_mean["trainset_target_std"],
#         )
#         model.load_state_dict(state_dict)

#     elif response == "no": # Model is being run from epoch 0 for the first time: 
#         model = TorchModel(
#             config=config["arch"],
#             target_mean=trainset.target.mean(axis=0),
#             target_std=trainset.target.std(axis=0),
#         )
#         std_mean = {"trainset_target_mean": trainset.target.mean(axis=0), "trainset_target_std": trainset.target.std(axis=0)}
# else: 
#     model = TorchModel(
#             config=config["arch"],
#             target_mean=trainset.target.mean(axis=0),
#             target_std=trainset.target.std(axis=0),
#         )
#     std_mean = {"trainset_target_mean": trainset.target.mean(axis=0), "trainset_target_std": trainset.target.std(axis=0)}

# model.freeze_layers(freeze_id="None")
# optimizer = getattr(torch.optim, config["optimizer"]["type"])(
#     model.parameters(), **config["optimizer"]["args"]
# )
# criterion = getattr(module_loss, config["criterion"])()
# metric_funcs = [getattr(module_metric, met) for met in config["metrics"]]

# # Build the trainer
# device = utils.prepare_device(config["device"])
# trainer = Trainer(
#     model,
#     criterion,
#     metric_funcs,
#     optimizer,
#     max_epochs=config["trainer"]["max_epochs"],
#     data_loader=train_loader,
#     validation_data_loader=val_loader,
#     device=device,
#     config=config,
# )

# # # Visualize the model
# torchinfo.summary(
#     model,
#     [   trainset.input[: config["data_loader"]["batch_size"]].shape ],
#     verbose=1,
#     col_names=("input_size", "output_size", "num_params"),
# )

# # TRAIN THE MODEL
# model.to(device)
# trainer.fit(std_mean)

# # Save the Model
# path = str(config["perlmutter_model_dir"]) + str(config["expname"]) + ".pth"
# torch.save({
#             "model_state_dict" : model.state_dict(),
#             "training_std_mean" : std_mean,
#              }, path)

# # Load the Model
# path = str(config["perlmutter_model_dir"]) + str(config["expname"]) + '.pth'

# load_model_dict = torch.load(path)

# state_dict = load_model_dict["model_state_dict"]
# std_mean = load_model_dict["training_std_mean"]

# model = TorchModel(
#     config=config["arch"],
#     target_mean=std_mean["trainset_target_mean"],
#     target_std=std_mean["trainset_target_std"],
# )

# model.load_state_dict(state_dict)
# model.eval()

# # Evaluate Training Metrics
# print(trainer.log.history.keys())

# print(trainer.log.history.keys())

# plt.figure(figsize=(20, 4))
# for i, m in enumerate(("loss", *config["metrics"])):
#     plt.subplot(1, 4, i + 1)
#     plt.plot(trainer.log.history["epoch"], trainer.log.history[m], label=m)
#     plt.plot(
#         trainer.log.history["epoch"], trainer.log.history["val_" + m], label="val_" + m
#     )
#     plt.axvline(
#        x=trainer.early_stopper.best_epoch, linestyle="--", color="k", linewidth=0.75
#     )
#     plt.title(m)
#     plt.legend()
# plt.tight_layout()
# plt.savefig(config["perlmutter_figure_dir"] + str(config["expname"]) + "/" + str(config["expname"]) + "training_metrics.png", format = 'png', dpi = 200) 

# # # # ------------------------------ Model Inference -------------------------------------------------------------
# # # # -------------------------------------------------------------------------------------------------------------

# if config["data_source"] == config["inference_data"]:
#     # Load the Model
#     path = str(config["perlmutter_model_dir"]) + str(config["expname"]) + '.pth'

#     load_model_dict = torch.load(path)

#     state_dict = load_model_dict["model_state_dict"]
#     std_mean = load_model_dict["training_std_mean"]

#     model = TorchModel(
#         config=config["arch"],
#         target_mean=std_mean["trainset_target_mean"],
#         target_std=std_mean["trainset_target_std"],
#     )

#     model.load_state_dict(state_dict)
#     model.eval()
    
#     device = utils.prepare_device(config["device"])

#     with torch.inference_mode():
#         print(device)
#         output = model.predict(dataset=testset, batch_size=128, device=device) # The output is the batched SHASH distribution parameters
    
#     # Save Model Outputs
#     model_output = str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + '_network_SHASH_parameters.pkl'
#     analysis_metrics.save_pickle(output, model_output)
#     print(output[:20]) # look at a small sample of the output data

# elif config["data_source"] != config["inference_data"]: 
#     print("PERFORMING INFERENCE ON OUT OF DISTRIBUTION DATA")
#     # specify which model experiment you'd like to use to make the inference: 
#     model_exp = config["trained_model"]
#     ood_config = utils.get_config(str(model_exp))
#     device = utils.prepare_device(ood_config["device"])
    
#     # Load the Model
#     path = str(ood_config["perlmutter_model_dir"]) + str(model_exp) + '.pth'

#     load_model_dict = torch.load(path)

#     state_dict = load_model_dict["model_state_dict"]
#     std_mean = load_model_dict["training_std_mean"]

#     model = TorchModel(
#         config=ood_config["arch"],
#         target_mean=std_mean["trainset_target_mean"],
#         target_std=std_mean["trainset_target_std"],
#     )

#     with torch.inference_mode():
#         print(device)
#         output = model.predict(dataset=testset, batch_size=128, device=device) # The output is the batched SHASH distribution parameters
    
#     # Save Model Outputs
#     ood_model_output = str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/' + str(model_exp) + 'T_' + str(config["expname"]) + '_OOD_INFERENCE_network_SHASH_parameters.pkl'
#     analysis_metrics.save_pickle(output, ood_model_output)
#     print(output[:20]) # look at a small sample of the output data

# # ------------------------------ Evaluate Network Predictions -------------------------------------------------
# # -------------------------------------------------------------------------------------------------------------
if config["input_data"] != "None": 
    input_trainfn = str(config["perlmutter_inputs_dir"]) + str(config["input_data"]) + "_trimmed_" + "train_dat.nc"
    input_valfn = str(config["perlmutter_inputs_dir"]) + str(config["input_data"]) + "_trimmed_" + "val_dat.nc"
    input_testfn = str(config["perlmutter_inputs_dir"]) + str(config["input_data"]) + "_trimmed_" + "test_dat.nc"
else: 
    input_trainfn = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_trimmed_" + "train_dat.nc"
    input_valfn = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_trimmed_" + "val_dat.nc"
    input_testfn = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_trimmed_" + "test_dat.nc"

lagtime = config["databuilder"]["lagtime"] 
smoothing_length = config["databuilder"]["averaging_length"]  
selected_months = config["databuilder"]["target_months"]
front_cutoff = config["databuilder"]["front_cutoff"] 
back_cutoff = config["databuilder"]["back_cutoff"] 

# # # # ## -------------------------------------------------------------------------------------------------
if config["data_source"] == config["inference_data"]: 
    # Save Model Outputs
    model_output = str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + '_network_SHASH_parameters.pkl'

elif config["data_source"] != config["inference_data"]: 
    model_exp = config["trained_model"]
    # Open OOD Model Outputs
    model_output = str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/' + str(model_exp) + 'T_' + str(config["expname"]) + '_OOD_INFERENCE_network_SHASH_parameters.pkl'

# Open Model Outputs
output = analysis_metrics.load_pickle(model_output)
print(f"output shape: {output.shape}")

# Open Target Data
test_inputs = open_data_file(input_testfn)
target = test_inputs['y']
# print(f"target time: {target.time.values[:300]}")
print(f"UDL target shape: {target.shape}")

# Open Climatology Data: TRAINING DATA
train_inputs = open_data_file(input_trainfn)
climatology = train_inputs['y']
print(f"UDL climatology shape {climatology.shape}")

if config["expname"] == "exp083":
    print("Adjusting climatology for OOD Inference Data")
    # Split ERA5 data into climatology and test data: 
    climatology_years = [1940, 1980]
    analysis_years = [1981, 2023]
    climatology = target.sel(time=slice(str(climatology_years[0]) + '-01-01', str(climatology_years[1]) + '-12-31'))
    target = target.sel(time=slice(str(analysis_years[0]) + '-01-01', str(analysis_years[1]) + '-12-31'))
    # trim output to just the indices corresponding with analysis years
    analysis_years_time = target.time.sel(time = slice(str(analysis_years[0]) + '-01-01', str(analysis_years[1]) + '-12-31'))
    analysis_years_indices = np.where(np.isin(target.time.values, analysis_years_time.values))[0]
    output = output[analysis_years_indices, ...]
    print(f"modified target length: {target.shape}")
    print(f"climatology length: {climatology.shape}")

# print(f"test input time: {test_inputs['x'].sel(time = slice('1851-01-01', '1852-01-01')).time}")
# print(f"test target time: {target.time.sel(time = slice('1851-01-01', '1852-01-01')).time}")

# # # Compare SHASH predictions to climatology histogram
p = calc_climatology.deriveclimatology(output, climatology, number_of_samples=50, config=config, climate_data = False)

# # # # ----------------------------- CRPS ------------------------------------------------------------------------
# # # # -------------------------------------------------------------------------------------------------------------
x = np.linspace(-10, 12, 1000)
x_wide = np.arange(-25, 25, 0.01)

# # Compute CRPS for climatology
# CRPS_climatology = CRPS.calculateCRPS(output, target, x_wide, config, climatology)

# # Compute CRPS for all predictions 
# CRPS_network = CRPS.calculateCRPS(output, target, x_wide, config, climatology = None)

# analysis_metrics.save_pickle(CRPS_climatology, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_CRPS_climatology_values.pkl")
# analysis_metrics.save_pickle(CRPS_network, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_CRPS_network_values.pkl")

CRPS_climatology = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_CRPS_climatology_values.pkl")
CRPS_network = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_CRPS_network_values.pkl")

# Compare CRPS scores for climatology vs predictions (Is network better than climatology on average?)
CRPS.CRPScompare(CRPS_network, CRPS_climatology, config)

# # # # # ----------------------------- ENSO ----------------------------------------------------------------------
# # # -------------------------------------------------------------------------------------------------------------

# # Calculate ENSO Indices from Monthly ENSO Data:
if config["data_source"] == "E3SM": 
    # Calculate ENSO Indices from Daily ENSO Data: 
    dailyENSOfn = '/pscratch/sd/p/plutzner/E3SM/bigdata/ENSO_Data/E3SM/ENSO_ne30pg2_HighRes/nino.member0201_daily_linterp_shifted.nc'

elif config["data_source"] == "ERA5":
    dailyENSOfn = '/pscratch/sd/p/plutzner/E3SM/bigdata/ENSO_Data/OBS/nino34.long.anom_daily_linterp_shifted.nc'
    
# Calculate enso indices for daily data based on EVALUATION DAY
daily_enso_timestamps = ENSO_indices_calculator.identify_nino_phases(dailyENSOfn, config, threshold=0.4, window=6, lagtime = lagtime, smoothing_length = smoothing_length)

# # save daily timestamps dictionary: 
enso_savename = str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_daily_enso_timestamps.pkl"
analysis_metrics.save_pickle(daily_enso_timestamps, enso_savename)
daily_enso_timestamps = analysis_metrics.load_pickle(enso_savename)

# elnino_dates = np.array([d.item() for d in daily_enso_timestamps["El Nino"]])
# lanina_dates = np.array([d.item() for d in daily_enso_timestamps["La Nina"]])
# neutral_dates = np.array([d.item() for d in daily_enso_timestamps["Neutral"]])

elnino_dates = np.array(daily_enso_timestamps["El Nino"])
lanina_dates = np.array(daily_enso_timestamps["La Nina"])
neutral_dates = np.array(daily_enso_timestamps["Neutral"])

# Separate CRPS scores by ENSO phases 
analysis.ENSO_indices_calculator.ENSO_CRPS(daily_enso_timestamps, CRPS_network, target.time, config)

# # # Compare Distributions? 
# # p = calc_climatology.deriveclimatology(output, climatology, number_of_samples=30, config=config, climate_data = False)

# ## Calculate precipitation anomalies during each ENSO Phase + Plot -----------------------------------------
# print("Calculating Raw Precip Test Data")
# # Open raw TESTING target data
# if config["inference_data"] == "E3SM":
#     nc_file = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc')
#     prect_global = nc_file.PRECT.sel(time = slice(str(config["databuilder"]["input_years"][0]) + '-01-01', str(config["databuilder"]["input_years"][1])))
# elif config["inference_data"] == "ERA5" or config["inference_data"] == "None":
#     nc_file = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5/ERA5_1x1_input_vars_1940-2023_regrid.nc')
#     prect_global = nc_file.tp
#     prect_global = prect_global.sel(time = slice(str(analysis_years[0]) + '-01-01', str(analysis_years[1]) + '-12-31'))
# else: 
#     ValueError("Please specify a valid inference data source")

# min_lat, max_lat = config["databuilder"]["target_region"][:2]
# min_lon, max_lon = config["databuilder"]["target_region"][2:]

# # Convert longitudes from -180 to 180 range to 0 to 360 range
# if min_lon < 0:
#     min_lon += 360
# if max_lon < 0:
#     max_lon += 360

# if isinstance(prect_global, xr.DataArray):
#     mask_lon = (prect_global.lon >= min_lon) & (prect_global.lon <= max_lon)
#     mask_lat = (prect_global.lat >= min_lat) & (prect_global.lat <= max_lat)
#     prect_regional = prect_global.where(mask_lon & mask_lat, drop=True)

# # average around target region
# prect_regional = prect_regional.mean(dim=['lat', 'lon'])
# target_raw = universaldataloader(prect_regional, config, target_only = True, repackage = False)
# target_raw = target_raw * 86400 * 1000  # Convert to mm/day

# analysis_metrics.save_pickle(target_raw, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_target_raw.pkl")
target_raw = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_target_raw.pkl")
    
# # # DISCARD PLOTS: --------------------------------------------------------------------------------------------
# # # -------------------------------------------------------------------------------------------------------------
print("Discard Plots")
# Discard plot of CRPS vs IQR Percentile for INCREASING CONFIDENCE, CRPS vs Anomalies & true precip
sample_index_increasingconf_anoms, inconf_perc, inconf_crps = analysis_metrics.IQRdiscard_plot(
    output, target, CRPS_network, CRPS_climatology, target.time, config, target_type = 'anomalous', keyword = 'All Samples', analyze_months = False, most_confident= True)
# sample_index_increasingconf_raw = analysis_metrics.IQRdiscard_plot(output, target_raw, CRPS_network, CRPS_climatology, config, target_type = 'raw', keyword = 'All Samples', analyze_months = False, most_confident= True)
# print(f"shape of sample index : {sample_index_increasingconf_anoms.shape}")

# # # Discard plot of CRPS vs IQR Percentile for DECREASING CONFIDENCE
sample_index_decreasingconf_anoms, deconf_perc, deconf_crps = analysis_metrics.IQRdiscard_plot(
    output, target, CRPS_network, CRPS_climatology, target.time, config, target_type = 'anomalous', keyword = 'All Samples', analyze_months = False, most_confident= False)

if config["data_source"] == "E3SM":
    if config["arch"]["type"] == "basicnn":
        crps_SNN = CRPS_network
        CNN_expname = "exp075" # CHOOSE
        crps_CNN = open_data_file(str(config["perlmutter_output_dir"]) + str(CNN_expname) + '/' + CNN_expname + '_CRPS_network_values.pkl')
        CNN_inputs = open_data_file(str(config["perlmutter_inputs_dir"]) + str(CNN_expname) + "_trimmed_" + "test_dat.nc")
        CNN_target = CNN_inputs['y']
        SNN_target = target
    else:
        crps_CNN = CRPS_network
        SNN_expname = "exp076" # CHOOSE
        crps_SNN = open_data_file(str(config["perlmutter_output_dir"]) + str(SNN_expname) + '/' + SNN_expname + '_CRPS_network_values.pkl')
        SNN_inputs = open_data_file(str(config["perlmutter_inputs_dir"]) + str(SNN_expname) + "_trimmed_" + "test_dat.nc")
        SNN_target = SNN_inputs['y']
        CNN_target = target

elif config["data_source"] == "ERA5":
    if config["arch"]["type"] == "basicnn":
        crps_SNN = CRPS_network
        CNN_expname = "exp079" # CHOOSE
        SNN_expname = config["expname"]
        crps_CNN = open_data_file(str(config["perlmutter_output_dir"]) + str(CNN_expname) + '/' + CNN_expname + '_CRPS_network_values.pkl')
        CNN_inputs = open_data_file(str(config["perlmutter_inputs_dir"]) + str(CNN_expname) + "_trimmed_" + "test_dat.nc")
        CNN_target = CNN_inputs['y']
        SNN_target = target
    else:
        crps_CNN = CRPS_network
        SNN_expname = "exp078" # CHOOSE
        CNN_expname = config["expname"]
        crps_SNN = open_data_file(str(config["perlmutter_output_dir"]) + str(SNN_expname) + '/' + SNN_expname + '_CRPS_network_values.pkl')
        SNN_inputs = open_data_file(str(config["perlmutter_inputs_dir"]) + str(SNN_expname) + "_trimmed_" + "test_dat.nc")
        SNN_target = SNN_inputs['y']
        CNN_target = target

# # Discard plot of CRPS vs Target Magnitude; CNN, Simple NN, Climo
analysis_metrics.target_discardplot(CNN_target, CNN_expname, SNN_target, SNN_expname, crps_CNN, crps_SNN, CRPS_climatology, config, target_type = 'anomalous', keyword = 'All Samples')
## analysis_metrics.target_discardplot(target_raw, SNN_target, crps_SNN, crps_SNN, CRPS_climatology, config, target_type = 'raw', keyword = 'All Samples')

# # SUCCESS RATIO discard plot
percentiles, avg_success_ratio = analysis_metrics.IQR_success_discard_plot(output, CRPS_network, CRPS_climatology, config, keyword = None)

success_plot_dict = {
    "percentiles": percentiles,
    "avg_success_ratio": avg_success_ratio}
analysis_metrics.save_pickle(success_plot_dict, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_success_ratio.pkl")

# # # # ANALYSIS OF BEST PREDICTIONS --------------------------------------------------------------------------------
# # # # -------------------------------------------------------------------------------------------------------------
## Analysis of Comparatively Better CRPS Predictions  
print("Analysis of 10-30% most confident predictions")
# Composite Maps - Anomalies by ENSO Phase for Most Confident Low CRPS Predictions
# input_test_maps = open_data_file(input_testfn)
# input_test_maps = input_test_maps['x']

# # # Isolate the MOST confident of these low CRPS predictions from 'sample_index_anoms'
percent = 30 
percentile = 100 - percent # 30% most confident predictions 
percentile_index_high = int(sample_index_increasingconf_anoms.shape[1] - ((percentile/100) * sample_index_increasingconf_anoms.shape[1]))
# TAKE OUT THE MOST CONFIDENT 10% WHICH HAVE ABNORMAL VALUES: #TODO: DECIDE WHETHER TO KEEP THIS FEATURE OR NOT
percentile_index_10 = int(sample_index_increasingconf_anoms.shape[1] - ((10/100) * sample_index_increasingconf_anoms.shape[1]))
percentile_index_high = np.unique(np.array([percentile_index_high, percentile_index_10]))
IQR_subset_highconf = sample_index_increasingconf_anoms[:, percentile_index_high][sample_index_increasingconf_anoms[..., percentile_index_high] != 0].astype(int)
IQR_subset_highconf_dates = target.time.isel(time = IQR_subset_highconf)

# # # Isolate the LEAST confident of low CRPS predictions from 'sample_index_anoms'
# percentile_index_low = int(sample_index_decreasingconf_anoms.shape[1] - ((percentile/100) * sample_index_decreasingconf_anoms.shape[1]))
# IQR_subset_lowconf = sample_index_decreasingconf_anoms[:, percentile_index_low][sample_index_decreasingconf_anoms[..., percentile_index_low] != 0].astype(int)
# IQR_subset_lowconf_dates = target.time.isel(time = IQR_subset_lowconf)

# # # Look at predictions with CRPS that are just comparatively lower than climatological CRPS on a sample-by-sample basis
comparatively_low_CRPS = np.where(CRPS_network < CRPS_climatology)[0]
lowCRPS_highconfident = np.intersect1d(comparatively_low_CRPS, IQR_subset_highconf).astype(int)
# # print(f"Number of Comparatively low CRPS High Conf Samples: {lowCRPS_highconfident.shape}")

# lowCRPS_lowconfident = np.intersect1d(comparatively_low_CRPS, IQR_subset_lowconf).astype(int)
# print(f"Number of Comparatively low CRPS Low Conf Samples: {lowCRPS_lowconfident.shape}")

sub_elnino_dates, sub_lanina_dates, sub_neutral_dates = analysis_metrics.subsetanalysis_SHASH_ENSO(lowCRPS_highconfident, daily_enso_timestamps, output, climatology, target, target_raw, config, x, subset_keyword = 'Comparatively Low CRPS') 

# # ## Composite Maps - Anomalies by ENSO Phase for Comparatively Low, Confident CRPS Predictions
# # analysis_metrics.compositemapping(sub_elnino_dates, input_test_maps, config, keyword= "Comparatively Low CRPS High Conf El Nino Norm")
# # analysis_metrics.compositemapping(sub_lanina_dates, input_test_maps, config, keyword= "Comparatively Low CRPS High Conf La Nina Norm")

# # # Composite Maps - All anomalies for Comparatively Low, Confident CRPS Predictions
# # analysis_metrics.compositemapping(lowCRPS_highconfident, input_test_maps, config, keyword= "Comparatively Low CRPS High Conf Norm All")

# # # Low CRPS, High Confidence SHASH Predictions:
# # lowCRPS_highconf_params = output[lowCRPS_highconfident, ...]
# # analysis_metrics.plotSHASH(lowCRPS_highconf_params, climatology, config, keyword = "ComparativelyLowCRPS_HighConf")

# # # Low CRPS, Low Confidence SHASH Predictions:
# # lowCRPS_lowconf_params = output[lowCRPS_lowconfident, ...]
# # analysis_metrics.plotSHASH(lowCRPS_lowconf_params, climatology, config, keyword = "ComparativelyLowCRPS_LowConf")

# # # Composite Maps - **CONDITIONED ON CONFIDENCE** enso phase samples -----------------------------------
# # Convert to sets to speed things up
# # print("Composite Map Sets")
# # highconf_set = set(IQR_subset_highconf_dates.values)
# # lowconf_set = set(IQR_subset_lowconf_dates.values)

# # # # EL NINO - High Confidence
# # mask = np.array([d in highconf_set for d in elnino_dates])
# # high_conf_EN_dates = elnino_dates[mask]
# # # # EL NINO - Low Confidence
# # mask = np.array([d in lowconf_set for d in elnino_dates])
# # low_conf_EN_dates  = elnino_dates[mask]

# # # # analysis_metrics.differenceplot(high_conf_EN_dates, low_conf_EN_dates, input_test_maps, target, CRPS_network, config, normalized = True, keyword= "Confidence Conditioned El Nino High-Low Confidence Norm")
# # # # analysis_metrics.differenceplot(high_conf_EN_dates, low_conf_EN_dates, input_test_maps, target, CRPS_network, config, normalized = False, keyword= "Confidence Conditioned El Nino High-Low Confidence")

# # # # LA NINA - High Confidence
# # mask = np.array([d in highconf_set for d in lanina_dates])
# # high_conf_LN_dates = lanina_dates[mask]
# # # # LA NINA - Low Confidence
# # mask = np.array([d in lowconf_set for d in elnino_dates])
# # low_conf_LN_dates  = elnino_dates[mask]

# # # # analysis_metrics.differenceplot(high_conf_LN_dates, low_conf_LN_dates, input_test_maps, target, CRPS_network, config, normalized = True, keyword= "Confidence Conditioned La Nina High-Low Confidence Norm")
# # # # analysis_metrics.differenceplot(high_conf_LN_dates, low_conf_LN_dates, input_test_maps, target, CRPS_network, config, normalized = False, keyword= "Confidence Conditioned La Nina High-Low Confidence")

# # # # Neutral - High Confidence
# # mask = np.array([d in highconf_set for d in neutral_dates])
# # high_conf_NE_dates = neutral_dates[mask]
# # # # Neutral - Low Confidence
# # mask = np.array([d in lowconf_set for d in neutral_dates])
# # low_conf_NE_dates  = neutral_dates[mask]
# # # analysis_metrics.differenceplot(high_conf_NE_dates, low_conf_NE_dates, input_test_maps, target, CRPS_network, config, normalized = True, keyword= "Confidence Conditioned Neutral High-Low Confidence Norm")
# # # analysis_metrics.differenceplot(high_conf_NE_dates, low_conf_NE_dates, input_test_maps, target, CRPS_network, config, normalized = False, keyword= "Confidence Conditioned Neutral High-Low Confidence")


# # # # ## Combined Phase Discard Plot: -------------------------------------------------------------------------
print("Combined Phase Discard Plot")

sample_index_increasingconf_anoms_EN, inconf_percEN, inconf_crpsEN = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, elnino_dates, 
                                                                        config, target_type = 'anomalous', keyword = 'Conditioned on El_Nino', analyze_months = False, most_confident= True)
sample_index_increasingconf_anoms_LN, inconf_percLN, inconf_crpsLN = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, lanina_dates, 
                                                                        config, target_type = 'anomalous', keyword = 'Conditioned on La_Nina', analyze_months = False, most_confident= True)
sample_index_increasingconf_anoms_NE, inconf_percNE, inconf_crpsNE = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, neutral_dates, 
                                                                        config, target_type = 'anomalous', keyword = 'Conditioned on Neutral', analyze_months = False, most_confident= True)

percentile_dict = {"EN": inconf_percEN, "LN": inconf_percLN, "NE": inconf_percNE}
crps_dict = {"EN": inconf_crpsEN, "LN": inconf_crpsLN, "NE": inconf_crpsNE}
plotting_data_dict = analysis_metrics.IQRdiscard_combined(percentile_dict, crps_dict, CRPS_climatology, config, keyword = None)

analysis_metrics.save_pickle(plotting_data_dict, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_combined_ENSO_IQR_discard_data.pkl")

# # # Z500 ANALYSIS Z500 Z500 Z500 ----------------------------------------------------------------------------
# # -------------------------------------------------------------------------------------------------------------

# TODO: ADD maps for OBS SNN? Because it's representative of what the simple indices are seeing? Good enough?

# # # Composite Maps with Z500 Geopotential Height: 
# # print("Z500 Analysis")
Z500_test_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/Z500_trimmed_processed_anomalies.v2.LR.historical_0201.eam.h1.1850-2014.nc')
Z500_test_data = Z500_test_data['x']
# analysis_metrics.compositemapping(lowCRPS_highconfident, Z500_test_data, config, keyword= "Comparatively Low CRPS High Conf Z500 All")
# analysis_metrics.compositemapping(lowCRPS_lowconfident, Z500_test_data, config, keyword= "Comparatively Low CRPS Low Conf Z500 All")

# Z500_complowCRPS_highconf_dates = Z500_test_data.time.sel(time = Z500_test_data.time[lowCRPS_highconfident])
# Z500_complowCRPS_lowconf_dates = Z500_test_data.time.sel(time = Z500_test_data.time[lowCRPS_lowconfident])
# print("High-Low All samples")
# analysis_metrics.differenceplot(Z500_complowCRPS_highconf_dates, Z500_complowCRPS_lowconf_dates, Z500_test_data, target, CRPS_network, config, normalized = True, keyword= "Comparatively Low CRPS High-Low Confidence Z500 Norm")
# analysis_metrics.differenceplot(Z500_complowCRPS_highconf_dates, Z500_complowCRPS_lowconf_dates, Z500_test_data, target, CRPS_network, config, normalized = False, keyword= "Comparatively Low CRPS High-Low Confidence Z500")

# print("Cond on El Nino")
# analysis_metrics.differenceplot(high_conf_EN_dates, low_conf_EN_dates, Z500_test_data, target, CRPS_network, config, normalized = True, keyword= "Z500 Confidence Conditioned El Nino High-Low Confidence Norm")
# analysis_metrics.differenceplot(high_conf_EN_dates, low_conf_EN_dates, Z500_test_data, target, CRPS_network, config, normalized = False, keyword= "Z500 Confidence Conditioned El Nino High-Low Confidence")

# print("Cond on La Nina")
# analysis_metrics.differenceplot(high_conf_LN_dates, low_conf_LN_dates, Z500_test_data, target, CRPS_network, config, normalized = True, keyword= "Z500 Confidence Conditioned La Nina High-Low Confidence Norm")
# analysis_metrics.differenceplot(high_conf_LN_dates, low_conf_LN_dates, Z500_test_data, target, CRPS_network, config, normalized = False, keyword= "Z500 Confidence Conditioned La Nina High-Low Confidence")

# print("Cond on Neutral")
# analysis_metrics.differenceplot(high_conf_NE_dates, low_conf_NE_dates, Z500_test_data, target, CRPS_network, config, normalized = True, keyword= "Z500 Confidence Conditioned Neutral High-Low Confidence Norm")
# analysis_metrics.differenceplot(high_conf_NE_dates, low_conf_NE_dates, Z500_test_data, target, CRPS_network, config, normalized = False, keyword= "Z500 Confidence Conditioned Neutral High-Low Confidence")

# MJO Phase Analysis: BOOSTRAPPING ----------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# phase_subsets = {
#     1: [2, 3, 4, 5], 
#     2: [6, 7, 8, 1]}
# MJO_phase_timestamps = analysis_metrics.mjo_subsetindices(
#     phase_subsets, input_test_maps, target, elnino_dates, lanina_dates, neutral_dates, CRPS_network, config, keyword = '2-5_6-1')

# phase_subsets = {
#     1: [3, 4, 5, 6], 
#     2: [7, 8, 1, 2]}
# MJO_subset = analysis_metrics.mjo_subsetindices(
#     phase_subsets, input_test_maps, target, elnino_dates, lanina_dates, neutral_dates, CRPS_network, config, keyword = '3-6_7-2')

# phase_subsets = {
#     1: [4, 5, 6, 7], 
#     2: [8, 1, 2, 3]}
# MJO_subset = analysis_metrics.mjo_subsetindices(
#     phase_subsets, input_test_maps, target, elnino_dates, lanina_dates, neutral_dates, CRPS_network, config, keyword = '4-7_8-3')

# phase_subsets = {
#     1: [5, 6, 7, 8], 
#     2: [1, 2, 3, 4]}
# MJO_subset = analysis_metrics.mjo_subsetindices(
#     phase_subsets, input_test_maps, target, elnino_dates, lanina_dates, neutral_dates, CRPS_network, config, keyword = '5-8_1-4')

# # PRECIP EXCEEDANCE THRESHOLD: ------------------------------------------------------------------------------
# # precip_thresh = 95

# # # # Isolate the time stamps of samples for which the target precip is at or above the 95th percentile 
# # # extreme_precip_dates = analysis_metrics.precip_exceedance_threshold(
# # #     target, output, precip_thresh, CRPS_network, CRPS_climatology, config, keyword = ' ')

# # index_increasingconf_exceedanceprecip_success = analysis_metrics.IQRdiscard_plot(
# #     output, target, CRPS_network, CRPS_climatology, extreme_precip_dates, config, target_type = 'anomalous', keyword = 'All Samples', analyze_months = False, most_confident= True)


# # # ENSO / MJO Tile Plots ! -------------------------------------------------------------------------------------
# # # -------------------------------------------------------------------------------------------------------------
# MJO_phase_timestamps = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/' + '2-5_6-1' + '_MJOphase_dates.pkl')

# def safe_parse_date(datestr):
#     try:
#         dt = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")
#         if dt.month == 2 and dt.day == 29:
#             return None  # Skip leap days for NoLeap calendar
#         return DatetimeNoLeap(dt.year, dt.month, dt.day)
#     except ValueError as e:
#         print(f"Skipping invalid date: {datestr} ({e})")
#         return None

# MJO_phase_timestamps_converted = {}
# for phase, datelist in MJO_phase_timestamps.items():
#     converted = [safe_parse_date(date) for date in datelist]
#     MJO_phase_timestamps_converted[phase] = [d for d in converted if d is not None]

# MJO_phase_timestamps = MJO_phase_timestamps_converted

# # CRPS! And 3d array of dates corresponding to each phase combination
# dates_array_by_phase = analysis_metrics.tiled_phase_analysis(MJO_phase_timestamps, elnino_dates, lanina_dates, neutral_dates, CRPS_network, target, config, keyword = 'CRPS')

# ## SUCCESS RATIO!
# success_ratio_all_samples = np.where(CRPS_network < CRPS_climatology, 1, 0)
# print(f"Success Ratio All Samples: {success_ratio_all_samples.shape}")
# analysis_metrics.tiled_phase_analysis(MJO_phase_timestamps, elnino_dates, lanina_dates, neutral_dates, success_ratio_all_samples, target, config, keyword = 'Success Ratio')

# # RMSE! 
# # mean of SHASH for all output parameters: 
# # Convert output columns to tensors
# mu = torch.tensor(output[:, 0], dtype=torch.float32)
# sigma = torch.tensor(output[:, 1], dtype=torch.float32)
# gamma = torch.tensor(output[:, 2], dtype=torch.float32)
# tau = torch.tensor(output[:, 3], dtype=torch.float32)  # Only if tau is included explicitly
# output_tensor = torch.stack((mu, sigma, gamma, tau), dim=1)

# # Instantiate Shash Class: 
# shash_instance = Shash(output_tensor)
# output_mean = shash_instance.mean()
# output_mean_np = output_mean.detach().cpu().numpy()  # Convert to numpy array
# print(f"output_mean shape: {output_mean_np.shape}")

# analysis_metrics.tiled_phase_analysis(MJO_phase_timestamps, elnino_dates, lanina_dates, neutral_dates, output_mean_np, target, config, keyword = 'RMSE')


# # ## COUNT + CONFIDENCE PLOTS : ENSO Comparison -----------------------------------------------------------------
# # # -------------------------------------------------------------------------------------------------------------
# histogram of IQR width binned for each ENSO phase
IQR_EN, IQR_LN, IQR_NE = analysis_metrics.countplot_IQR(output, target, elnino_dates, lanina_dates, neutral_dates, config, keyword = 'All_samples')

# # # Composite Maps: WHEN CNN IS BETTER THAN SNN: ----------------------------------------------------------------
# # # -------------------------------------------------------------------------------------------------------------

# load both CNN and SNN data: 
if config["inference_data"] == "E3SM":
    SNN_expname = "exp076"
    CNN_expname = "exp075"
elif config["inference_data"] == "ERA5":
    SNN_expname = "exp082"
    CNN_expname = "exp079"

crps_threshold = 0.5
# returns crps and time for which the CNN has lower CRPS than SNN and is below the threshold
CNN_crps_da_lower = analysis_metrics.CNN_SNN_ComparativeComposites(CNN_expname, SNN_expname, crps_threshold, Z500_test_data, config, keyword = "Z500")
# find index fo CNN_crps_da_lower.time within target.time: 
CNN_crps_da_lower_index = np.where(target.time == CNN_crps_da_lower.time)[0]
sub_EN_dates, sub_LN_dates, sub_NE_dates = analysis_metrics.subsetanalysis_SHASH_ENSO(CNN_crps_da_lower_index, daily_enso_timestamps, output, climatology, target, target_raw, config, x, subset_keyword = 'CNN < SNN')

# return crps and timestamps for samples where CNN crps is ABSOLUTE LOWEST relative to SNN (blue sliver)
crps_threshold = 0.44
CNN_crps_da_lowest = analysis_metrics.CNN_SNN_ComparativeComposites(CNN_expname, SNN_expname, crps_threshold, Z500_test_data, config, keyword = "Z500")
CNN_crps_da_lowest_index = np.where(target.time == CNN_crps_da_lowest.time)[0]
sublowest_EN_dates, sublowest_LN_dates, sublowest_NE_dates = analysis_metrics.subsetanalysis_SHASH_ENSO(CNN_crps_da_lowest_index, daily_enso_timestamps, output, climatology, target, target_raw, config, x, subset_keyword = 'Lowest CNN CRPS < SNN CRPS')

## PDF Histogram Plots: IQR, ANOMALIES, RAW DATA, CRPS (WHEN CNN IS BETTER THAN SNN --------------------------

# IQR PDFs
IQR_cnn_network_output = iqr_basic(output)
IQR_cnn_network_output_lowest = iqr_basic(output[CNN_crps_da_lowest_index])
IQR_data_list = [IQR_cnn_network_output, IQR_cnn_network_output_lowest]
IQR_data_labels = ["IQR", "CNN IQR All Samples", "CNN IQR (Lowest CRPS)"]

bins = np.linspace(min(IQR_data_list[0]), max(IQR_data_list[0]), 100)
utils.plot_hist(IQR_data_list, IQR_data_labels, bins, config)

# ANOMALIES PDFs
target_lowest_crps = target.isel(time = CNN_crps_da_lowest_index)
anomalies_data_list = [target, target_lowest_crps]
anomalies_data_labels = ["Precip Anomalies (mm/day)", "Target Anomalies All Samples", "Target Anomalies (Lowest CRPS)"]

bins = np.linspace(min(anomalies_data_list[0]), max(anomalies_data_list[0]), 100)
utils.plot_hist(anomalies_data_list, anomalies_data_labels, bins, config)

# RAW VALUES PDFs
target_raw_lowest_crps = target_raw.isel(time = CNN_crps_da_lowest_index)
raw_data_list = [target_raw, target_raw_lowest_crps]
raw_data_labels = ["Raw Precip (mm/day)", "Target Raw All Samples", "Target Raw (Lowest CRPS)"]

bins = np.linspace(min(raw_data_list[0]), max(raw_data_list[0]), 100)
utils.plot_hist(raw_data_list, raw_data_labels, bins, config)

# CRPS PDFs
crps_data_list = [CRPS_network, CNN_crps_da_lowest.values]
crps_data_labels = ["CRPS", "CNN CRPS All Samples", "CNN CRPS (Lowest CRPS)"]

bins = np.linspace(0, 2, 100)
utils.plot_hist(crps_data_list, crps_data_labels, bins, config)


## XAI - CAPTUM  ----------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# Use 3D array of dates_array_by_phase to generate XAI plots for specific phase combinations (according to their dates)

# # Open input testset and ensure that it is in tensor form
# input_test_maps = open_data_file(input_testfn)
# input_test_maps = input_test_maps['x']
# # print(f"input_test_maps shape : {input_test_maps.shape}")

# output_column = 1 # SIGMA IS SHASH PARAMETER OF INTEREST FOR XAI 

# # Compute Integrated Gradients Composites: 

# # CNN CRPS MJO4_EN
# # select dates from the 3D array and select only the ones that are not None
# MJO4_EN_dates = dates_array_by_phase[4, 0, :]
# MJO4_EN_dates = [date for date in MJO4_EN_dates if date is not None]
# avg_attributions_IG_MJO4_EN = average_attributions(model, input_test_maps, MJO4_EN_dates, device, output_column, config, method="integrated_gradients", keyword = "IG_MJO4_EN")


# MJO8_LN_dates = dates_array_by_phase[8, 1, :]
# MJO8_LN_dates = [date for date in MJO8_LN_dates if date is not None]
# avg_attributions_IG_MJO4_EN = average_attributions(model, input_test_maps, MJO8_LN_dates, device, output_column, config, method="integrated_gradients", keyword = "IG_MJO8_LN")



# -----------------------------------------------------------------------------------------------------------------

# Calculate precip regime for raw target input data
# Open raw INPUT target data
# nc_file = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc')
# precip_regime(nc_file, config)




























# GARBAGE LAND: -----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------- 

# # # Compute the average attributions
# # print("Computing Average Attributions - IG") 
# # print("Computing Average Attributions - DL")
# # avg_attributions_DL = average_attributions(model, input_test_maps, sub_elnino, device, output_column, config, method="deeplift")
# # print("Computing Average Attributions - S")
# # avg_attributions_S = average_attributions(model, input_test_maps, sub_elnino, device, output_column, config, method="saliency")
# # print(f"Average Attributions Shape: {avg_attributions_IG.shape}")

# # Load the average attributions
# avg_attributions_IG = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/average_attributions_integrated_gradients.pkl')
# avg_attributions_DL = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/average_attributions_deeplift.pkl')
# avg_attributions_S = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/average_attributions_saliency.pkl')

# avg_attributions_P_DL = avg_attributions_DL[..., 0]  # Precipitation channel
# avg_attributions_TS_DL = avg_attributions_DL[..., 1]  # Temperature channel

# avg_attributions_P_S = avg_attributions_S[..., 0]  # Precipitation channel
# avg_attributions_TS_S = avg_attributions_S[..., 1]  # Temperature channel

# # Visualize the average attributions for each channel
# # EL NINO --------
# # average input map for given indices:    
# ave_input_test_map_elnino_P = np.mean(input_test_maps[sub_elnino, ..., 0], axis=0)
# ave_input_test_map_elnino_TS = np.mean(input_test_maps[sub_elnino, ..., 1], axis=0)

# visualize_average_attributions(avg_attributions_P_IG, ave_input_test_map_elnino_P, config, keyword='El Nino Precip Anomalies - Integrated Gradients')
# visualize_average_attributions(avg_attributions_TS_IG, ave_input_test_map_elnino_TS, config, keyword='El Nino Skin Temp Anomalies - Integrated Gradients')

# visualize_average_attributions(avg_attributions_P_DL, ave_input_test_map_elnino_P, config, keyword='El Nino Precip Anomalies - DeepLift')
# visualize_average_attributions(avg_attributions_TS_DL, ave_input_test_map_elnino_TS, config, keyword='El Nino Skin Temp Anomalies - DeepLift')

# visualize_average_attributions(avg_attributions_P_S, ave_input_test_map_elnino_P, config, keyword='El Nino Precip Anomalies - Saliency')
# visualize_average_attributions(avg_attributions_TS_S, ave_input_test_map_elnino_TS, config, keyword='El Nino Skin Temp Anomalies - Saliency')

# # LA NINA --------
# ave_input_test_map_lanina_P = np.mean(input_test_maps[sub_lanina, ..., 0], axis=0)
# ave_input_test_map_lanina_TS = np.mean(input_test_maps[sub_lanina, ..., 1], axis=0)

# visualize_average_attributions(avg_attributions_P_IG, ave_input_test_map_lanina_P, config, keyword='La Nina Precip Anomalies - Integrated Gradients')
# visualize_average_attributions(avg_attributions_TS_IG, ave_input_test_map_lanina_TS, config, keyword='La Nina Skin Temp Anomalies - Integrated Gradients')

# Z500 Geopotential Height ------
# ave_input_test_map_elnino_Z500 = np.mean(Z500_complowCRPS_highconf[sub_elnino, ...], axis=0)
# visualize_average_attributions(None, ave_input_test_map_elnino_Z500, config, keyword='El Nino Z500 Anomalies')


# # Spread-Skill Ratio
# # spread_skill_plot = analysis_metrics.spread_skill(output, target, config)


# Composite Maps - **CONDITIONED ON ENSO** Phase Low and High Confidence Samples -----------------------------------

# # EL NINO - High Confidence

# percentile_index_EN = int(sample_index_increasingconf_anoms_EN.shape[1] - ((percentile/100) * sample_index_increasingconf_anoms_EN.shape[1]))
# IQR_subset_highconf_EN = sample_index_increasingconf_anoms_EN[..., percentile_index_EN][sample_index_increasingconf_anoms_EN[..., percentile_index_EN] != 0]

# # # EL NINO - Low Confidence
# sample_index_decreasingconf_anoms_EN, deconf_percEN, deconf_crpsEN = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, elnino_dates, 
#                                                                         config, target_type = 'anomalous', keyword = 'Conditioned on El_Nino', analyze_months = False, most_confident= False)
# percentile_index_EN = int(sample_index_decreasingconf_anoms_EN.shape[1] - ((percentile/100) * sample_index_decreasingconf_anoms_EN.shape[1]))
# IQR_subset_lowconf_EN = sample_index_decreasingconf_anoms_EN[..., percentile_index_EN][sample_index_decreasingconf_anoms_EN[..., percentile_index_EN] != 0]

# analysis_metrics.differenceplot(IQR_subset_highconf_EN, IQR_subset_lowconf_EN, input_test_maps, CRPS_network, config, normalized = True, keyword= "ENSO Conditioned El Nino High-Low Confidence Norm")
# analysis_metrics.differenceplot(IQR_subset_highconf_EN, IQR_subset_lowconf_EN, input_test_maps, CRPS_network, config, normalized = False, keyword= "ENSO Conditioned El Nino High-Low Confidence")

# # # LA NINA - High Confidence
# sample_index_increasingconf_anoms_LN, inconf_percLN, inconf_crpsLN = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, lanina_dates, 
#                                                                         config, target_type = 'anomalous', keyword = 'Conditioned on La_Nina', analyze_months = False, most_confident= True)
# percentile_index_LN = int(sample_index_increasingconf_anoms_LN.shape[1] - ((percentile/100) * sample_index_increasingconf_anoms_LN.shape[1]))
# IQR_subset_highconf_LN = sample_index_increasingconf_anoms_LN[..., percentile_index_LN][sample_index_increasingconf_anoms_LN[..., percentile_index_LN] != 0]

# # # LA NINA - Low Confidence
# sample_index_decreasingconf_anoms_LN, deconf_percLN, deconf_crpsLN  = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, lanina_dates,
#                                                                         config, target_type = 'anomalous', keyword = 'Conditioned on La_Nina', analyze_months = False, most_confident= False)
# percentile_index_LN = int(sample_index_decreasingconf_anoms_LN.shape[1] - ((percentile/100) * sample_index_decreasingconf_anoms_LN.shape[1]))
# IQR_subset_lowconf_LN = sample_index_decreasingconf_anoms_LN[..., percentile_index_LN][sample_index_decreasingconf_anoms_LN[..., percentile_index_LN] != 0]

# analysis_metrics.differenceplot(IQR_subset_highconf_LN, IQR_subset_lowconf_LN, input_test_maps, CRPS_network, config, normalized = True, keyword= "ENSO Conditioned La Nina High-Low Confidence Norm")
# analysis_metrics.differenceplot(IQR_subset_highconf_LN, IQR_subset_lowconf_LN, input_test_maps, CRPS_network, config, normalized = False, keyword= "ENSO Conditioned La Nina High-Low Confidence")

# # # Neutral - High Confidence
# sample_index_increasingconf_anoms_NE, inconf_percNE, inconf_crpsNE = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, neutral_dates, config, target_type = 'anomalous', 
#                                                                         keyword = 'Neutral', analyze_months = False, most_confident= True)
# percentile_index_NE = int(sample_index_increasingconf_anoms_NE.shape[1] - ((percentile/100) * sample_index_increasingconf_anoms_NE.shape[1]))
# IQR_subset_highconf_NE = sample_index_increasingconf_anoms_NE[..., percentile_index_NE][sample_index_increasingconf_anoms_NE[..., percentile_index_NE] != 0]

# # # Neutral - Low Confidence
# sample_index_decreasingconf_anoms_NE, deconf_percNE, deconf_crpsNE = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, neutral_dates, config, target_type = 'anomalous', 
#                                                                         keyword = 'Neutral', analyze_months = False, most_confident= False)
# percentile_index_NE = int(sample_index_decreasingconf_anoms_NE.shape[1] - ((percentile/100) * sample_index_decreasingconf_anoms_NE.shape[1]))
# IQR_subset_lowconf_NE = sample_index_decreasingconf_anoms_NE[..., percentile_index_NE][sample_index_decreasingconf_anoms_NE[..., percentile_index_NE] != 0]

# analysis_metrics.differenceplot(IQR_subset_highconf_NE, IQR_subset_lowconf_NE, input_test_maps, CRPS_network, config, normalized = True, keyword= "ENSO Conditioned Neutral High-Low Confidence Norm")
# analysis_metrics.differenceplot(IQR_subset_highconf_NE, IQR_subset_lowconf_NE, input_test_maps, CRPS_network, config, normalized = False, keyword= "ENSO Conditioned Neutral High-Low Confidence")


# bins = 100

# # IQR PDFs
# IQR_cnn_network_output = iqr_basic(output)
# IQR_pdf_all, IQR_all_BC = pdf_from_hist(IQR_cnn_network_output, bins = bins)

# IQR_cnn_network_output_lowest = iqr_basic(output[CNN_crps_da_lowest_index])
# IQR_pdf_lowest, IQR_lowest_BC = pdf_from_hist(IQR_cnn_network_output_lowest, bins = bins)

# IQR_pdfs = np.array([IQR_pdf_all, IQR_pdf_lowest])
# IQR_pdfs_BC = np.array([IQR_all_BC, IQR_lowest_BC])

# analysis_metrics.pdf_comparison(IQR_pdfs, IQR_pdfs_BC, config, keyword = "IQR (Lowest CRPS)")

# # ANOMALIES PDFs
# target_anomalies_pdf_all, target_anomalies_all_BC = pdf_from_hist(target, bins = bins)
# target_anomalies_pdf_lowest, target_anomalies_lowest_BC = pdf_from_hist(target.isel(time = CNN_crps_da_lowest_index), bins = bins)

# target_anomalies_pdfs = np.array([target_anomalies_pdf_all, target_anomalies_pdf_lowest])
# target_anomalies_pdfs_BC = np.array([target_anomalies_all_BC, target_anomalies_lowest_BC])

# analysis_metrics.pdf_comparison(target_anomalies_pdfs, target_anomalies_pdfs_BC, config, keyword = "Anomalies (Lowest CRPS)")

# # RAW VALUES PDFs
# raw_target_pdf_all, raw_target_all_BC = pdf_from_hist(target_raw, bins = bins)
# raw_target_pdf_lowest, raw_target_lowest_BC = pdf_from_hist(target_raw.isel(time = CNN_crps_da_lowest_index), bins = bins)

# raw_target_pdfs = np.array([raw_target_pdf_all, raw_target_pdf_lowest])
# raw_target_pdfs_BC = np.array([raw_target_all_BC, raw_target_lowest_BC])

# analysis_metrics.pdf_comparison(raw_target_pdfs, raw_target_pdfs_BC, config, keyword = "Raw Target (Lowest CRPS)")

# # CRPS PDFs
# crps_pdf_all, crps_all_BC = pdf_from_hist(CRPS_network, bins = bins)
# crps_pdf_lowest, crps_lowest_BC = pdf_from_hist(CNN_crps_da_lowest.values, bins = bins)

# crps_pdfs = np.array([crps_pdf_all, crps_pdf_lowest])
# crps_pdfs_BC = np.array([crps_all_BC, crps_lowest_BC])

# analysis_metrics.pdf_comparison(crps_pdfs, crps_pdfs_BC, config, keyword = "CRPS (Lowest CRPS)")