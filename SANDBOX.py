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
from analysis.calc_climatology import precip_regime
from utils.filemethods import create_folder
from databuilder.data_generator import uniform_dist
from captum.attr import IntegratedGradients, Saliency
from XAI.captum import compute_attributions, average_attributions, visualize_average_attributions

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"pytorch version = {torch.__version__}")

# https://github.com/victoresque/pytorch-template/tree/master

# ----CONFIG AND CLASS SETUP----------------------------------------------
config = utils.get_config("exp042")
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
    target_treatment = "True"
else:
    target_treatment = "False"

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
# Check if input data is being processed from scratch or if it is being loaded from a previous experiment

if config["input_data"] == "None": # Then input data must be processed FROM SCRATCH
    # print("Processing input data from scratch")

    # d_train, d_val, d_test = data.fetch_data()

    # # ---- FOR SIMPLE INPUTS ONLY : ----------------------------------------------
    # if config["arch"]["type"] == "basicnn":
    #     print(d_train['y'].shape)
    #     target_savename1 = str(config["perlmutter_data_dir"]) + str(config["expname"]) + "_d_train_TARGET_1850-2014.pkl"
    #     with gzip.open(target_savename1, "wb") as fp:
    #         pickle.dump(d_train, fp)

    #     target_savename2 = str(config["perlmutter_data_dir"]) + str(config["expname"]) + "_d_val_TARGET_1850-2014.pkl"
    #     with gzip.open(target_savename2, "wb") as fp:
    #         pickle.dump(d_val, fp)

    #     target_savename3 = str(config["perlmutter_data_dir"]) + str(config["expname"]) + "_d_test_TARGET_1850-2014.pkl"
    #     with gzip.open(target_savename3, "wb") as fp:
    #         pickle.dump(d_test, fp)

    #     d_train, d_val, d_test = multi_input_data_organizer(config, target_savename1, target_savename2, target_savename3, MJO = True, ENSO = True, other = False)

        # # confirm metadata is stored for both input and target
        # print(f" s_dict_train INPUT time {d_train['x'].time}")
        # print(f" s_dict_train TARGET time {d_train['y'].time}")

        # # confirm input structure: 
    #     # print(f"input shape: {d_train['x'].shape}")
    # else: 
    #     pass

    # Save full input data for the experiment: ----------------------------------
    s_dict_savename1 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_d_train.pkl"
    s_dict_savename2 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_d_val.pkl"
    s_dict_savename3 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_d_test.pkl"

    # with gzip.open(s_dict_savename1, "wb") as fp:
    #     pickle.dump(d_train, fp)

    # with gzip.open(s_dict_savename2, "wb") as fp:
    #     pickle.dump(d_val, fp)

    # with gzip.open(s_dict_savename3, "wb") as fp:
    #     pickle.dump(d_test, fp)

    # Trim input data: Lead/lag, month selection ----------------------------------
    trimmed_trainfn = config["perlmutter_inputs_dir"] + str(config["expname"]) + "_trimmed_" + "train_dat.nc"
    trimmed_valfn = config["perlmutter_inputs_dir"] + str(config["expname"]) + "_trimmed_" + "val_dat.nc"
    trimmed_testfn = config["perlmutter_inputs_dir"] + str(config["expname"]) + "_trimmed_" + "test_dat.nc"

    # train_dat_trimmed = universaldataloader(s_dict_savename1, config, target_only = False, repackage = True)
    # train_dat_trimmed.to_netcdf(trimmed_trainfn)
    # print(f"Data saved to {trimmed_trainfn}")

    # val_dat_trimmed = universaldataloader(s_dict_savename2, config, target_only = False, repackage = True)
    # val_dat_trimmed.to_netcdf(trimmed_valfn)
    # print(f"Data saved to {trimmed_valfn}")

    # test_dat_trimmed = universaldataloader(s_dict_savename3, config, target_only = False, repackage = True)
    # test_dat_trimmed.to_netcdf(trimmed_testfn)
    # print(f"Data saved to {trimmed_testfn}")

elif "exp" in config["input_data"]: 
    trimmed_trainfn = str(config["perlmutter_inputs_dir"]) + str(config["input_data"]) + "_trimmed_" + "train_dat.nc"
    trimmed_valfn = str(config["perlmutter_inputs_dir"]) + str(config["input_data"]) + "_trimmed_" + "val_dat.nc"
    trimmed_testfn = str(config["perlmutter_inputs_dir"]) + str(config["input_data"]) + "_trimmed_" + "test_dat.nc"
    
# --- Setup the Data for Training ---------------------------------------------
lagtime = config["databuilder"]["lagtime"] 
smoothing_length = config["databuilder"]["averaging_length"]

# trainset = data_loader.CustomData(trimmed_trainfn, config, which_set = 'training')
# valset = data_loader.CustomData(trimmed_valfn, config, which_set = 'validation')
# testset = data_loader.CustomData(trimmed_testfn, config, which_set = 'testing')

# train_loader = torch.utils.data.DataLoader(
#     trainset,
#     batch_size=config["data_loader"]["batch_size"],
#     shuffle=True,
#     drop_last=False
# )

# val_loader = torch.utils.data.DataLoader(
#     valset,
#     batch_size=config["data_loader"]["batch_size"],
#     shuffle=False,
#     drop_last=False
# )

# # --- Setup the Model ----------------------------------------------------

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

# # model.freeze_layers(freeze_id="tau")
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

# # # ------------------------------ Model Inference ----------------------------------

# with torch.inference_mode():
#     print(device)
#     output = model.predict(dataset=testset, batch_size=128, device=device) # The output is the batched SHASH distribution parameters

# # Save Model Outputs
# model_output = str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + '_network_SHASH_parameters.pkl'
# analysis_metrics.save_pickle(output, model_output)
# print(output[:20]) # look at a small sample of the output data

# ------------------------------ Evaluate Network Predictions ----------------------------------
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

## -------------------------------------------------------------------

# Open Model Outputs
model_output = str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + '_network_SHASH_parameters.pkl'
output = analysis_metrics.load_pickle(model_output)
print(f"output shape: {output.shape}")

# Open Target Data
test_inputs = open_data_file(input_testfn)
target = test_inputs['y']
print(f"UDL target shape: {target.shape}")

# Open Climatology Data: TRAINING DATA
train_inputs = open_data_file(input_trainfn)
climatology = train_inputs['y']
print(f"UDL climatology shape {climatology.shape}")

# Compare SHASH predictions to climatology histogram
# p = calc_climatology.deriveclimatology(output, climatology, number_of_samples=50, config=config, climate_data = False)

# # ----------------------------- CRPS --------------------------------------------
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
# CRPS.CRPScompare(CRPS_network, CRPS_climatology, config)

# # # ----------------------------- ENSO -------------------------------------------

# Calculate ENSO Indices from Monthly ENSO Data (Po-Lun): 
monthlyENSO = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/ENSO_ne30pg2_HighRes/nino.member0201.nc')
# monthlyENSO = xr.open_dataset(str(config["perlmutter_data_dir"]) + 'presaved/ENSO_ne30pg2_HighRes/nino.member0201.nc')
Nino34 = monthlyENSO.nino34
# select a slice of only certain years
Nino34 = Nino34.sel(time=slice ( str(config["databuilder"]["input_years"][0]) + '-01-01', str(config["databuilder"]["input_years"][1]) + '-12-31'))
Nino34 = Nino34.values

# Calculate enso indices for daily data based on EVALUATION DAY
enso_indices_daily = ENSO_indices_calculator.identify_nino_phases(Nino34, config, threshold=0.4, window=6, lagtime = lagtime, smoothing_length = smoothing_length)

# # Separate CRPS scores by ENSO phases 
elnino_dates, lanina_dates, neutral_dates, CRPS_elnino, CRPS_lanina, CRPS_neutral = analysis.ENSO_indices_calculator.ENSO_CRPS(
    enso_indices_daily, CRPS_network, climatology, x, output, target, config)
#elnino, lanina, neutral, elnino_dates,

# # Compare Distributions? 
# p = calc_climatology.deriveclimatology(output, climatology, number_of_samples=30, config=config, climate_data = False)

# Calculate precipitation anomalies during each ENSO Phase + Plot -----------------

# # Open raw TESTING target data
# nc_file = xr.open_dataset('/Users/C830793391/BIG_DATA/E3SM_Data/ens3/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc')
# nc_file = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc')
# prect_global = nc_file.PRECT.sel(time = slice(str(config["databuilder"]["input_years"][0]) + '-01-01', str(config["databuilder"]["input_years"][1])))

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
    
# DISCARD PLOTS: --------------------------------------------------------------------------------------------
# Discard plot of CRPS vs IQR Percentile for INCREASING CONFIDENCE, CRPS vs Anomalies & true precip
# sample_index_increasingconf_raw = analysis_metrics.IQRdiscard_plot(output, target_raw, CRPS_network, CRPS_climatology, config, target_type = 'raw', keyword = 'All Samples', analyze_months = False, most_confident= True)
sample_index_increasingconf_anoms = analysis_metrics.IQRdiscard_plot(
    output, target, CRPS_network, CRPS_climatology, target.time, config, target_type = 'anomalous', keyword = 'All Samples', analyze_months = False, most_confident= True)
print(f"shape of sample index : {sample_index_increasingconf_anoms.shape}")
# # Discard plot of CRPS vs IQR Percentile for DECREASING CONFIDENCE
sample_index_decreasingconf_anoms = analysis_metrics.IQRdiscard_plot(
    output, target, CRPS_network, CRPS_climatology, target.time, config, target_type = 'anomalous', keyword = 'All Samples', analyze_months = False, most_confident= False)

# if config["arch"]["type"] == "basicnn":
#     crps_SNN = CRPS_network
#     CNN_expname = "exp070" # CHOOSE
#     crps_CNN = open_data_file(str(config["perlmutter_output_dir"]) + str(CNN_expname) + '/' + CNN_expname + '_CRPS_network_values.pkl')
#     CNN_inputs = open_data_file(str(config["perlmutter_inputs_dir"]) + str(CNN_expname) + "_trimmed_" + "test_dat.nc")
#     CNN_target = CNN_inputs['y']
#     SNN_target = target
# else:
#     crps_CNN = CRPS_network
#     SNN_expname = "exp036" # CHOOSE
#     crps_SNN = open_data_file(str(config["perlmutter_output_dir"]) + str(SNN_expname) + '/' + SNN_expname + '_CRPS_network_values.pkl')
#     SNN_inputs = open_data_file(str(config["perlmutter_inputs_dir"]) + str(SNN_expname) + "_trimmed_" + "test_dat.nc")
#     SNN_target = SNN_inputs['y']
#     CNN_target = target

# Discard plot of CRPS vs Target Magnitude; CNN, Simple NN, Climo
# analysis_metrics.target_discardplot(CNN_target, SNN_target, crps_CNN, crps_SNN, CRPS_climatology, config, target_type = 'anomalous', keyword = 'All Samples')
# analysis_metrics.target_discardplot(target_raw, SNN_target, crps_SNN, crps_SNN, CRPS_climatology, config, target_type = 'raw', keyword = 'All Samples')

# ANALYSIS OF BEST PREDICTIONS --------------------------------------------------------------------------------
## Analysis of Comparatively Better CRPS Predictions  

# Isolate the MOST confident of these low CRPS predictions from 'sample_index_anoms'
percent = 30 
percentile = 100 - percent # 30% most confident predictions 
percentile_index = int(sample_index_increasingconf_anoms.shape[1] - ((percentile/100) * sample_index_increasingconf_anoms.shape[1]))
IQR_subset_highconf = sample_index_increasingconf_anoms[:, percentile_index]

# Isolate the LEAST confident of low CRPS predictions from 'sample_index_anoms'
percentile_index = int(sample_index_decreasingconf_anoms.shape[1] - ((percentile/100) * sample_index_decreasingconf_anoms.shape[1]))
IQR_subset_lowconf = sample_index_decreasingconf_anoms[:, percentile_index]

# ## Composite Maps - Anomalies by ENSO Phase for LOWEST Most Confident CRPS Predictions
input_test_maps = open_data_file(input_testfn)
input_test_maps = input_test_maps['x']

# # Look at predictions with CRPS that are just comparatively lower than climatological CRPS on a sample-by-sample basis
comparatively_low_CRPS = np.where(CRPS_network < CRPS_climatology)[0]
lowCRPS_highconfident = np.intersect1d(comparatively_low_CRPS, IQR_subset_highconf).astype(int)
print(f"Number of Comparatively low CRPS High Conf Samples: {lowCRPS_highconfident.shape}")

lowCRPS_lowconfident = np.intersect1d(comparatively_low_CRPS, IQR_subset_lowconf).astype(int)
print(f"Number of Comparatively low CRPS Low Conf Samples: {lowCRPS_lowconfident.shape}")

sub_elnino_dates, sub_lanina_dates, sub_neutral_dates = analysis_metrics.subsetanalysis_SHASH_ENSO(lowCRPS_highconfident, output, climatology, target, target_raw, config, x, subset_keyword = 'Comparatively Low CRPS') 

## Composite Maps - Anomalies by ENSO Phase for Comparatively Low, Confident CRPS Predictions
analysis_metrics.compositemapping(sub_elnino_dates, input_test_maps, config, keyword= "Comparatively Low CRPS High Conf El Nino Norm")
analysis_metrics.compositemapping(sub_lanina_dates, input_test_maps, config, keyword= "Comparatively Low CRPS High Conf La Nina Norm")

# Composite Maps - All anomalies (El Nino, La Nina, Neutral) for Comparatively Low, Confident CRPS Predictions
analysis_metrics.compositemapping(lowCRPS_highconfident, input_test_maps, config, keyword= "Comparatively Low CRPS High Conf Norm All")

# Low CRPS, High Confidence SHASH Predictions:
lowCRPS_highconf_params = output[lowCRPS_highconfident, ...]
analysis_metrics.plotSHASH(lowCRPS_highconf_params, climatology, config, keyword = "ComparativelyLowCRPS_HighConf")

# Low CRPS, Low Confidence SHASH Predictions:
lowCRPS_lowconf_params = output[lowCRPS_lowconfident, ...]
analysis_metrics.plotSHASH(lowCRPS_lowconf_params, climatology, config, keyword = "ComparativelyLowCRPS_LowConf")

# Composite Maps - Conditioned on ENSO Phase Low and High Confidence Samples -----------------------------------


# EL NINO - High Confidence
sample_index_increasingconf_anoms_EN = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, elnino_dates,
                                                                        config, target_type = 'anomalous', keyword = 'El_Nino', analyze_months = False, most_confident= True)
percentile = 30 # 30% most/least confident predictions
percentile_index_EN = int(sample_index_increasingconf_anoms_EN.shape[1] - ((percentile/100) * sample_index_increasingconf_anoms_EN.shape[1]))
IQR_subset_highconf_EN = sample_index_increasingconf_anoms_EN[..., percentile_index_EN]

# # EL NINO - Low Confidence
sample_index_decreasingconf_anoms_EN = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, elnino_dates, 
                                                                        config, target_type = 'anomalous', keyword = 'El_Nino', analyze_months = False, most_confident= False)
percentile_index_EN = int(sample_index_decreasingconf_anoms_EN.shape[1] - ((percentile/100) * sample_index_decreasingconf_anoms_EN.shape[1]))
IQR_subset_lowconf_EN = sample_index_decreasingconf_anoms_EN[..., percentile_index_EN]

analysis_metrics.differenceplot(IQR_subset_highconf_EN, IQR_subset_lowconf_EN, input_test_maps, CRPS_network, config, normalized = True, keyword= "El Nino High-Low Confidence Norm")
analysis_metrics.differenceplot(IQR_subset_highconf_EN, IQR_subset_lowconf_EN, input_test_maps, CRPS_network, config, normalized = False, keyword= "El Nino High-Low Confidence")

# # LA NINA - High Confidence
sample_index_increasingconf_anoms_LN = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, lanina_dates, 
                                                                        config, target_type = 'anomalous', keyword = 'La_Nina', analyze_months = False, most_confident= True)
percentile_index_LN = int(sample_index_increasingconf_anoms_LN.shape[1] - ((percentile/100) * sample_index_increasingconf_anoms_LN.shape[1]))
IQR_subset_highconf_LN = sample_index_increasingconf_anoms_LN[..., percentile_index_LN]

# # LA NINA - Low Confidence
sample_index_decreasingconf_anoms_LN = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, lanina_dates,
                                                                        config, target_type = 'anomalous', keyword = 'La_Nina', analyze_months = False, most_confident= False)
percentile_index_LN = int(sample_index_decreasingconf_anoms_LN.shape[1] - ((percentile/100) * sample_index_decreasingconf_anoms_LN.shape[1]))
IQR_subset_lowconf_LN = sample_index_decreasingconf_anoms_LN[..., percentile_index_LN]

analysis_metrics.differenceplot(IQR_subset_highconf_LN, IQR_subset_lowconf_LN, input_test_maps, CRPS_network, config, normalized = True, keyword= "La Nina High-Low Confidence Norm")
analysis_metrics.differenceplot(IQR_subset_highconf_LN, IQR_subset_lowconf_LN, input_test_maps, CRPS_network, config, normalized = False, keyword= "La Nina High-Low Confidence")

# # Neutral - High Confidence
sample_index_increasingconf_anoms_NE = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, neutral_dates, config, target_type = 'anomalous', 
                                                                        keyword = 'Neutral', analyze_months = False, most_confident= True)
percentile_index_NE = int(sample_index_increasingconf_anoms_NE.shape[1] - ((percentile/100) * sample_index_increasingconf_anoms_NE.shape[1]))
IQR_subset_highconf_NE = sample_index_increasingconf_anoms_NE[..., percentile_index_NE]

# # Neutral - Low Confidence
sample_index_decreasingconf_anoms_NE = analysis_metrics.IQRdiscard_plot(output, target, CRPS_network, CRPS_climatology, neutral_dates, config, target_type = 'anomalous', 
                                                                        keyword = 'Neutral', analyze_months = False, most_confident= False)
percentile_index_NE = int(sample_index_decreasingconf_anoms_NE.shape[1] - ((percentile/100) * sample_index_decreasingconf_anoms_NE.shape[1]))
IQR_subset_lowconf_NE = sample_index_decreasingconf_anoms_NE[..., percentile_index_NE]

analysis_metrics.differenceplot(IQR_subset_highconf_NE, IQR_subset_lowconf_NE, input_test_maps, CRPS_network, config, normalized = True, keyword= "Neutral High-Low Confidence Norm")
analysis_metrics.differenceplot(IQR_subset_highconf_NE, IQR_subset_lowconf_NE, input_test_maps, CRPS_network, config, normalized = False, keyword= "Neutral High-Low Confidence")

print(f"IQR_subset_highconf_EN: {IQR_subset_highconf_EN[:10]}")
print(f"IQR_subset_lowconf_EN: {IQR_subset_lowconf_EN[:10]}")
print(f"IQR_subset_highconf_LN: {IQR_subset_highconf_LN[:10]}")
print(f"IQR_subset_lowconf_LN: {IQR_subset_lowconf_LN[:10]}")
print(f"IQR_subset_highconf_NE: {IQR_subset_highconf_NE[:10]}")
print(f"IQR_subset_lowconf_NE: {IQR_subset_lowconf_NE[:10]}")


# Composite Maps with Z500 Geopotential Height: 
Z500_test_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/Z500_trimmed_processed_anomalies.v2.LR.historical_0201.eam.h1.1850-2014.nc')
Z500_test_data = Z500_test_data['x']
analysis_metrics.compositemapping(lowCRPS_highconfident, Z500_test_data, config, keyword= "Comparatively Low CRPS High Conf Z500 All")
analysis_metrics.compositemapping(lowCRPS_lowconfident, Z500_test_data, config, keyword= "Comparatively Low CRPS Low Conf Z500 All")

Z500_complowCRPS_highconf = Z500_test_data.sel(time = Z500_test_data.time[lowCRPS_highconfident])
Z500_complowCRPS_lowconf = Z500_test_data.sel(time = Z500_test_data.time[lowCRPS_lowconfident])
analysis_metrics.differenceplot(lowCRPS_highconfident, lowCRPS_lowconfident, Z500_test_data, CRPS_network, config, normalized = True, keyword= "Comparatively Low CRPS High-Low Confidence Z500 Norm")
analysis_metrics.differenceplot(lowCRPS_highconfident, lowCRPS_lowconfident, Z500_test_data, CRPS_network, config, normalized = False, keyword= "Comparatively Low CRPS High-Low Confidence Z500")

# MJO Phase Analysis: BOOSTRAPPING ----------------------------------------------------------------------------
phase_subsets = {
    1: [2, 3, 4, 5], 
    2: [6, 7, 8, 1]}

MJO_subset = analysis_metrics.mjo_subsetindices(
    phase_subsets, input_test_maps, target, elnino_dates, lanina_dates, neutral_dates, CRPS_network, config, keyword = '2-5_6-1')

phase_subsets = {
    1: [3, 4, 5, 6], 
    2: [7, 8, 1, 2]}

MJO_subset = analysis_metrics.mjo_subsetindices(
    phase_subsets, input_test_maps, target, elnino_dates, lanina_dates, neutral_dates, CRPS_network, config, keyword = '3-6_7-2')

phase_subsets = {
    1: [4, 5, 6, 7], 
    2: [8, 1, 2, 3]}

MJO_subset = analysis_metrics.mjo_subsetindices(
    phase_subsets, input_test_maps, target, elnino_dates, lanina_dates, neutral_dates, CRPS_network, config, keyword = '4-7_8-3')

phase_subsets = {
    1: [5, 6, 7, 8], 
    2: [1, 2, 3, 4]}

MJO_subset = analysis_metrics.mjo_subsetindices(
    phase_subsets, input_test_maps, target, elnino_dates, lanina_dates, neutral_dates, CRPS_network, config, keyword = '5-8_1-4')


## PRECIP EXCEEDANCE THRESHOLD: ------------------------------------------------------------------------------
precip_thresh = 95

# Isolate the time stamps of samples for which the target precip is at or above the 95th percentile 
extreme_precip_dates = analysis_metrics.precip_exceedance_threshold(
    target, output, precip_thresh, CRPS_network, CRPS_climatology, config, keyword = ' ')

index_increasingconf_exceedanceprecip_success = analysis_metrics.IQRdiscard_plot(
    output, target, CRPS_network, CRPS_climatology, config, target_type = 'anomalous', keyword = 'All Samples', analyze_months = False, most_confident= True)


# ENSO / MJO Tile Plots ! -------------------------------------------------------------------------------------

# MJO_subset = analysis_metrics.mjo_subsetindices(
#     phase_subsets, input_test_maps, target, elnino, lanina, neutral, CRPS_network, config, keyword = 'All_samples')





## XAI - CAPTUM  ----------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------
# output_column = 1 # SIGMA IS SHASH PARAMETER OF INTEREST FOR XAI 

# # Open input testset and ensure that it is in tensor form
# input_test_maps = open_data_file(input_testfn)
# input_test_maps = input_test_maps['x'].values
# print(f"input_test_maps shape : {input_test_maps.shape}")

# input_test_maps_P = torch.tensor(input_test_maps, dtype=torch.float32).to(device)

# # # Compute the average attributions
# # print("Computing Average Attributions - IG") #TODO: fix sub_elnino_dates for the following functions
# # avg_attributions_IG = average_attributions(model, input_test_maps, sub_elnino, device, output_column, config, method="integrated_gradients")
# # print("Computing Average Attributions - DL")
# # avg_attributions_DL = average_attributions(model, input_test_maps, sub_elnino, device, output_column, config, method="deeplift")
# # print("Computing Average Attributions - S")
# # avg_attributions_S = average_attributions(model, input_test_maps, sub_elnino, device, output_column, config, method="saliency")
# # print(f"Average Attributions Shape: {avg_attributions_IG.shape}")

# # Load the average attributions
# avg_attributions_IG = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/average_attributions_integrated_gradients.pkl')
# avg_attributions_DL = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/average_attributions_deeplift.pkl')
# avg_attributions_S = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/average_attributions_saliency.pkl')

# # Split the attributions by channel
# avg_attributions_P_IG = avg_attributions_IG[..., 0]  # Precipitation channel
# avg_attributions_TS_IG = avg_attributions_IG[..., 1]  # Temperature channel

# avg_attributions_P_DL = avg_attributions_DL[..., 0]  # Precipitation channel
# avg_attributions_TS_DL = avg_attributions_DL[..., 1]  # Temperature channel

# avg_attributions_P_S = avg_attributions_S[..., 0]  # Precipitation channel
# avg_attributions_TS_S = avg_attributions_S[..., 1]  # Temperature channel

# # Visualize the average attributions for each channel
# # EL NINO --------
# # average input map for given indices:    #TODO: fix sub_elnino_dates for the following functions
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








# Calculate precip regime for raw target input data
# Open raw INPUT target data
# nc_file = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc')
# precip_regime(nc_file, config)

# GARBAGE LAND: ------------------------------------------------------------------------------------------------------------------

# # Spread-Skill Ratio
# # spread_skill_plot = analysis_metrics.spread_skill(output, target, config)