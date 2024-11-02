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
import databuilder.data_loader as data_loader
import databuilder.data_generator as data_generator
from databuilder.data_generator import ClimateData
import model.loss as module_loss
import model.metric as module_metric
from databuilder.data_generator import multi_input_data_organizer
import databuilder.data_loader as data_loader
from trainer.trainer import Trainer
from model.build_model import TorchModel
from base.base_model import BaseModel
from utils import utils
from shash.shash_torch import Shash
import analysis.analysis_metrics as analysis_metrics


print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"pytorch version = {torch.__version__}")

# https://github.com/victoresque/pytorch-template/tree/master
 
# ------------------------------------------------------------------

config = utils.get_config("exp007")
seed = config["seed_list"][0]

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True

# ---------------- Data Processing ----------------------------------

# imp.reload(utils)
# imp.reload(filemethods)
# imp.reload(data_generator)

# Instantiate Climate Data class for Global Map input data processing
data = ClimateData(
    config["databuilder"], 
    expname = config["expname"],
    seed=seed,
    data_dir = config["data_dir"], 
    figure_dir=config["figure_dir"],
    target_only = False, 
    fetch=False,
    verbose=False
)
# print("Instantiated ClimateData Class")

# # Fetch training, validation, and testing data
# d_train, d_val, d_test = data.fetch_data()
# print("Fetched data")

# # convert data to xarray form from SampleClass object: 
# d_train_dict = dict(d_train) 
# d_train_xr = xr.Dataset(d_train_dict)

# d_val_dict = dict(d_val) 
# d_val_xr = xr.Dataset(d_val_dict)

# d_test_dict = dict(d_test) 
# d_test_xr = xr.Dataset(d_test_dict)

# # Saving training data as NetCDF
# savename1 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp007_d_train_1850-1900.nc"
# d_train_xr.to_netcdf(savename1)
# print("Saved Training Data as NetCDF")

# # Saving validation data as NetCDF
# savename2 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp007_d_val_1850-1900.nc"
# d_val_xr.to_netcdf(savename2)
# print("Saved Validation Data as NetCDF")

# # Saving testing data as NetCDF
# savename3 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp007_d_test_1850-1900.nc"
# d_test_xr.to_netcdf(savename3)
# print("Saved Testing Data as NetCDF")

# s_dict_savename1 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + '_d_train_1850-1900.nc'
# s_dict_savename2 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + '_d_val_1850-1900.nc'
# s_dict_savename3 = str(config["perlmutter_inputs_dir"]) + str(config["expname"]) + '_d_test_1850-1900.nc'

s_dict_savename1 = str(config["local_inputs_dir"]) + "/" + str(config["expname"]) + '_d_train_1850-1900.nc'
s_dict_savename2 = str(config["local_inputs_dir"]) + "/" + str(config["expname"]) + '_d_val_1850-1900.nc'
s_dict_savename3 = str(config["local_inputs_dir"]) + "/" + str(config["expname"]) + '_d_test_1850-1900.nc'

# Open processed data filess
train_dat = xr.open_dataset(s_dict_savename1)
val_dat = xr.open_dataset(s_dict_savename2)
test_dat = xr.open_dataset(s_dict_savename3)

# print(f"training data shape: {train_dat['x'].shape}")
# print(f"val data shape: {val_dat['x'].shape}")
# print(f"test data shape: {test_dat['x'].shape} \n")

# print(f"training data shape: {train_dat['y'].shape}")
# print(f"val data shape: {val_dat['y'].shape}")
# print(f"test data shape: {test_dat['y'].shape} \n")

# # ----------- Model Training ----------------------------------

# Setup the Data
trainset = data_loader.CustomData(config["data_loader"]["data_dir"] + "/Network Inputs/" + str(config["expname"]) + "_d_train_1850-1900.nc", config["databuilder"]["lagtime"], config["databuilder"]["averaging_length"])
valset = data_loader.CustomData(config["data_loader"]["data_dir"] + "/Network Inputs/" + str(config["expname"]) + "_d_val_1850-1900.nc", config["databuilder"]["lagtime"], config["databuilder"]["averaging_length"])
testset = data_loader.CustomData(config["data_loader"]["data_dir"] + "/Network Inputs/" + str(config["expname"]) + "_d_test_1850-1900.nc", config["databuilder"]["lagtime"], config["databuilder"]["averaging_length"])

# trainset = data_loader.CustomData(config["perlmutter_inputs_dir"] + str(config["expname"]) + "_d_train_1850-1900.nc", front_cutoff, back_cutoff)
# valset = data_loader.CustomData(config["perlmutter_inputs_dir"] + str(config["expname"]) + "_d_val_1850-1900.nc", front_cutoff, back_cutoff)
# testset = data_loader.CustomData(config["perlmutter_inputs_dir"] + str(config["expname"]) + "_d_test_1850-1900.nc", front_cutoff, back_cutoff)

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

# Setup the Model
model = TorchModel(
    config=config["arch"],
    target_mean=trainset.target.values.mean(axis=0),
    target_std=trainset.target.values.std(axis=0),
)
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
# torchinfo.summary(
#     model,
#     [   trainset.input[: config["data_loader"]["batch_size"]].shape ],
#     verbose=0,
#     col_names=("input_size", "output_size", "num_params"),
# )

# # Train the Model
# print("training model")
# model.to(device)
# trainer.fit()

# # Save the Model
# path = '/Users/C830793391/Documents/Research/E3SM/saved/models/exp007_v0.pth'
# torch.save(model.state_dict(), path)

# Load the Model
path = '/Users/C830793391/Documents/Research/E3SM/saved/models/exp007_v0.pth'
model = TorchModel(config=config["arch"])
model.load_state_dict(torch.load(path))
model.eval()

print(trainer.log.history.keys())

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
# plt.savefig('/Users/C830793391/Documents/Research/E3SM/visuals/' + str(config["expname"]) + '_training_metrics.png', format='png', bbox_inches ='tight', dpi = 300)

# # ---------------- Model Evaluation ----------------------------------

with torch.inference_mode():
    print(device)
    output = model.predict(dataset=testset, batch_size=128, device=device) # The output is the batched SHASH distribution parameters

print(output[:20]) # look at a small sample of the output data

# # Save Model Outputs
model_output = str(config["output_dir"]) + 'exp007_output_testset.pkl'
analysis_metrics.save_pickle(output, model_output)












# EXTRA -------------------------------------------------------------------------

# # Save processed data files
# savename1 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp007_d_train_1850-1900.pkl"
# # #target_savename1 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp007_d_train.pkl"
# analysis_metrics.save_pickle(d_train, savename1)
# print("Saved Training Data")

# savename2 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp007_d_val_1850-1900.pkl"
# # #target_savename2 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp007_d_val.pkl"
# analysis_metrics.save_pickle(d_val, savename2)
# print("Saved Validation Data")

# savename3 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp007_d_test_1850-1900.pkl"
# # #target_savename3 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp007_d_test.pkl"
# analysis_metrics.save_pickle(d_test, savename3)
# print("Saved Testing Data")
# print(f"Opening data: \n")

# train_dat = analysis_metrics.load_pickle(s_dict_savename1)
# val_dat = analysis_metrics.load_pickle(s_dict_savename2)
# test_dat = analysis_metrics.load_pickle(s_dict_savename3)