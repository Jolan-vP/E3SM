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

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"pytorch version = {torch.__version__}")

# https://github.com/victoresque/pytorch-template/tree/master

# ----CONFIG AND CLASS SETUP----------------------------------------------
config = utils.get_config("exp028")
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

# ----PROCESS E3SM DATA-------------------------------------------------------

d_train, d_val, d_test = data.fetch_data()


# ---- FOR SIMPLE INPUTS ONLY : ----------------------------------------------

# print(d_train['y'].shape)
# target_savename1 = str(config["perlmutter_data_dir"]) + str(config["expname"]) + "_d_train_TARGET_1850-2014.pkl"
# with gzip.open(target_savename1, "wb") as fp:
#     pickle.dump(d_train, fp)

# target_savename2 = str(config["perlmutter_data_dir"]) + str(config["expname"]) + "_d_val_TARGET_1850-2014.pkl"
# with gzip.open(target_savename2, "wb") as fp:
#     pickle.dump(d_val, fp)

# target_savename3 = str(config["perlmutter_data_dir"]) + str(config["expname"]) + "_d_test_TARGET_1850-2014.pkl"
# with gzip.open(target_savename3, "wb") as fp:
#     pickle.dump(d_test, fp)

# s_dict_train, s_dict_val, s_dict_test = multi_input_data_organizer(config, target_savename1, target_savename2, target_savename3, MJO = True, ENSO = True, other = False)

# # confirm metadata is stored for both input and target
# print(f" s_dict_train INPUT time {s_dict_train['x'].time}")
# print(f" s_dict_train TARGET time {s_dict_train['y'].time}")

# # confirm input structure: 
# print(f"input shape: {s_dict_train['x'].shape}")

# --- SAVE FULL INPUT DATA------------------------------------------------------

s_dict_savename1 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_d_train.pkl"
s_dict_savename2 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_d_val.pkl"
s_dict_savename3 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + "_d_test.pkl"

# with gzip.open(s_dict_savename1, "wb") as fp:
#     pickle.dump(d_train, fp)

# with gzip.open(s_dict_savename2, "wb") as fp:
#     pickle.dump(d_val, fp)

# with gzip.open(s_dict_savename3, "wb") as fp:
#     pickle.dump(d_test, fp)

# # ------- TRIM INPUT DATA ----------------------------------
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

# ---OPEN DATA---------------------------------------------

train_dat = open_data_file(trimmed_trainfn)
val_dat = open_data_file(trimmed_valfn)
test_dat = open_data_file(trimmed_testfn)

# Confirm data looks correct: 
print(f"Train_dat inputs: {train_dat['x'][-24:].values}")
print(f"val_dat target: {val_dat['y'][-24:].values}")

# # --- Setup the Data for Training ---------------------------------------------
lagtime = config["databuilder"]["lagtime"] 
smoothing_length = config["databuilder"]["averaging_length"]

trainset = data_loader.CustomData(trimmed_trainfn, config, is_train = True)
valset = data_loader.CustomData(trimmed_valfn, config, is_train = False)
testset = data_loader.CustomData(trimmed_testfn, config, is_train = False)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=config["data_loader"]["batch_size"],
    shuffle=True,
    drop_last=False,
)

val_loader = torch.utils.data.DataLoader(
    valset,
    batch_size=config["data_loader"]["batch_size"],
    shuffle=False,
    drop_last=False,
)

# # # --- Setup the Model ----------------------------------------------------

model = TorchModel(
    config=config["arch"],
    target_mean=trainset.target.mean(axis=0),
    target_std=trainset.target.std(axis=0),
)
std_mean = {"trainset_target_mean": trainset.target.mean(axis=0), "trainset_target_std": trainset.target.std(axis=0)}

model.freeze_layers(freeze_id="tau")
optimizer = getattr(torch.optim, config["optimizer"]["type"])(
    model.parameters(), **config["optimizer"]["args"]
)
criterion = getattr(module_loss, config["criterion"])()
metric_funcs = [getattr(module_metric, met) for met in config["metrics"]]

# Build the trainer
device = utils.prepare_device(config["device"])
trainer = Trainer(
    model,
    criterion,
    metric_funcs,
    optimizer,
    max_epochs=config["trainer"]["max_epochs"],
    data_loader=train_loader,
    validation_data_loader=val_loader,
    device=device,
    config=config,
)

# # Visualize the model
torchinfo.summary(
    model,
    [   trainset.input[: config["data_loader"]["batch_size"]].shape ],
    verbose=1,
    col_names=("input_size", "output_size", "num_params"),
)

# TRAIN THE MODEL
model.to(device)
trainer.fit()

# Save the Model
path = str(config["perlmutter_model_dir"]) + str(config["expname"]) + ".pth"
torch.save({
            "model_state_dict" : model.state_dict(),
            "training_std_mean" : std_mean,
             }, path)


# Load the Model
path = str(config["perlmutter_model_dir"]) + str(config["expname"]) + '.pth'

load_model_dict = torch.load(path)

state_dict = load_model_dict["model_state_dict"]
training_std_mean = load_model_dict["training_std_mean"]

model = TorchModel(
    config=config["arch"],
    target_mean=training_std_mean["trainset_target_mean"],
    target_std=training_std_mean["trainset_target_std"],
)

model.load_state_dict(state_dict)
model.eval()

# Evaluate Training Metrics
print(trainer.log.history.keys())

print(trainer.log.history.keys())

plt.figure(figsize=(20, 4))
for i, m in enumerate(("loss", *config["metrics"])):
    plt.subplot(1, 4, i + 1)
    plt.plot(trainer.log.history["epoch"], trainer.log.history[m], label=m)
    plt.plot(
        trainer.log.history["epoch"], trainer.log.history["val_" + m], label="val_" + m
    )
    plt.axvline(
       x=trainer.early_stopper.best_epoch, linestyle="--", color="k", linewidth=0.75
    )
    plt.title(m)
    plt.legend()
plt.tight_layout()
plt.savefig(config["perlmutter_figure_dir"] + str(config["expname"]) + "/" + str(config["expname"]) + "training_metrics.png", format = 'png', dpi = 200) 

# # ------------------------------ Model Inference ----------------------------------

with torch.inference_mode():
    print(device)
    output = model.predict(dataset=testset, batch_size=128, device=device) # The output is the batched SHASH distribution parameters

# Save Model Outputs
model_output = str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + '_network_SHASH_parameters.pkl'
analysis_metrics.save_pickle(output, model_output)
print(output[:20]) # look at a small sample of the output data

# ------------------------------ Evaluate Network Predictions ----------------------------------

# lagtime = config["databuilder"]["lagtime"] 
# smoothing_length = config["databuilder"]["averaging_length"]  
# selected_months = config["databuilder"]["target_months"]
# front_cutoff = config["databuilder"]["front_cutoff"] 
# back_cutoff = config["databuilder"]["back_cutoff"] 

# # -------------------------------------------------------------------

# # Open Model Outputs
# model_output = str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + '_network_SHASH_parameters.pkl'
# output = analysis_metrics.load_pickle(model_output)
# print(f"output shape: {output.shape}")


# # Open Target Data
# test_inputs = open_data_file(trimmed_testfn)
# target = test_inputs['y']
# print(f"UDL target shape: {target.shape}")

# # Open Climatology Data: TRAINING DATA
# train_inputs = open_data_file(trimmed_trainfn)
# climatology = train_inputs['y']
# print(f"UDL climatology shape {climatology.shape}")

# # Compare SHASH predictions to climatology histogram

# p = calc_climatology.deriveclimatology(output, climatology, number_of_samples=50, config=config, climate_data = False)

# # # ----------------------------- CRPS ----------------------------------
# x = np.linspace(-10, 12, 1000)
# x_wide = np.arange(-25, 25, 0.01)

# # # Comput CRPS for climatology
# # CRPS_climatology = CRPS.calculateCRPS(output, target, x_wide, config, climatology)

# # # Compute CRPS for all predictions 
# # CRPS_network = CRPS.calculateCRPS(output, target, x_wide, config, climatology = None)

# # analysis_metrics.save_pickle(CRPS_climatology, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_CRPS_climatology_values.pkl")
# # analysis_metrics.save_pickle(CRPS_network, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_CRPS_network_values.pkl")

# CRPS_climatology = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_CRPS_climatology_values.pkl")
# CRPS_network = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_CRPS_network_values.pkl")

# # Compare CRPS scores for climatology vs predictions (Is network better than climatology on average?)
# CRPS.CRPScompare(CRPS_network, CRPS_climatology, config)

# # ----------------------------- ENSO -------------------------------------------

# # Calculate ENSO Indices from Monthly ENSO Data (Po-Lun): 
# # monthlyENSO = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/ENSO_ne30pg2_HighRes/nino.member0201.nc')
# monthlyENSO = xr.open_dataset(str(config["perlmutter_data_dir"]) + 'presaved/ENSO_ne30pg2_HighRes/nino.member0201.nc')
# Nino34 = monthlyENSO.nino34
# # select a slice of only certain years
# Nino34 = Nino34.sel(time=slice ( str(config["databuilder"]["input_years"][0]) + '-01-01', str(config["databuilder"]["input_years"][1]) + '-12-31'))
# Nino34 = Nino34.values

# enso_indices_daily = ENSO_indices_calculator.identify_nino_phases(Nino34, config, threshold=0.4, window=6, lagtime = lagtime, smoothing_length = smoothing_length)

# # # Separate CRPS scores by ENSO phases 
# elnino, lanina, neutral, CRPS_elnino, CRPS_lanina, CRPS_neutral = analysis.ENSO_indices_calculator.ENSO_CRPS(enso_indices_daily, CRPS_network, climatology, x, output, config)

# # Compare Distributions? 
# p = calc_climatology.deriveclimatology(output, climatology, number_of_samples=30, config=config, climate_data = False)

# # Calculate precipitation anomalies during each ENSO Phase + Plot -----------------

# # Open raw TESTING target data
# nc_file = xr.open_dataset('/Users/C830793391/BIG_DATA/E3SM_Data/ens3/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc')
# # nc_file = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc')
# prect_global = nc_file.PRECT.sel(time = slice(str(config["databuilder"]["input_years"][0]) + '-01-01', str(config["databuilder"]["input_years"][1])))

# min_lat, max_lat = config["databuilder"]["target_region"][:2]
# min_lon, max_lon = config["databuilder"]["target_region"][2:]

# if isinstance(prect_global, xr.DataArray):
#     mask_lon = (prect_global.lon >= min_lon) & (prect_global.lon <= max_lon)
#     mask_lat = (prect_global.lat >= min_lat) & (prect_global.lat <= max_lat)
#     prect_regional = prect_global.where(mask_lon & mask_lat, drop=True)

# # average around seattle region 
# prect_regional = prect_regional.mean(dim=['lat', 'lon'])

# target_raw = universaldataloader(prect_regional, config, target_only = True, repackage = False)

# target_raw = target_raw * 86400 * 1000  # Convert to mm/day

# analysis_metrics.save_pickle(target_raw, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_target_raw.pkl")

# target_raw = analysis_metrics.load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_target_raw.pkl")
# # print(f"mean raw target: {np.mean(target_raw)}")
# # print(f"median raw target: {np.median(target_raw)}")
# # print(f"std raw target: {np.std(target_raw)}")
    
# # Discard plot of CRPS vs IQR Percentile, CRPS vs Anomalies & true precip
# sample_index_raw = analysis_metrics.discard_plot(output, target_raw, CRPS_network, CRPS_climatology, config, target_type = 'raw', analyze_months = False, most_confident= True)
# sample_index_anoms = analysis_metrics.discard_plot(output, target, CRPS_network, CRPS_climatology, config, target_type = 'anomalous', analyze_months = False, most_confident= True)


# anomalies_by_ENSO_phase = analysis_metrics.anomalies_by_ENSO_phase(elnino, lanina, neutral, target, target_raw, sample_index_raw, config, keyword = 'MJJAS')
# # anomalies_by_ENSO_phase_confident = analysis_metrics.anomalies_by_ENSO_phase(elnino, lanina, neutral, target, target_raw, dry_month_indices, config, keyword = 'dry_months')

# # DRY MONTH ANALYSIS: 
# top_percentage = 20
# analysis_metrics.subsetanalysis_SHASH_ENSO(sample_index_anoms, output, climatology, target, target_raw, config, x, percentage= top_percentage, subset_keyword = 'MJJAS Inputs')

# # open testing input data: 
# testing_input = open_data_file(trimmed_testfn)
# testing_input = testing_input['x']


# analysis_metrics.compositemapping(idry_5confident, idry_5unconfident, testing_input, config, keyword1 = '5% Most Confident', keyword2 = '5% Least Confident')

# Calculate precip regime for raw target input data
# Open raw INPUT target data
# nc_file = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc')
# precip_regime(nc_file, config)

# # Spread-Skill Ratio
# # spread_skill_plot = analysis_metrics.spread_skill(output, target, config)



# GARBAGE LAND: ------------------------------------------------------------------------------------------------------------------

# This seems unecessary: 
# # monthlyENSO = xr.open_dataset(str(config["perlmutter_data_dir"]) + 'presaved/ENSO_ne30pg2_HighRes/nino.member0201.nc')
# # Nino34 = monthlyENSO.nino34
# # # select a slice of only certain years
# # Nino34 = Nino34.sel(time=slice ( str(config["databuilder"]["input_years"][0]) + '-01-01', str(config["databuilder"]["input_years"][1]) + '-12-31'))
# # Nino34 = Nino34.values