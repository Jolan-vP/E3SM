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

# # Save processed data files
# savename1 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp007_d_train.pkl"
# # #target_savename1 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp007_d_train.pkl"
# analysis_metrics.save_pickle(d_train, savename1)
# print("Saved Training Data")

# savename2 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp007_d_val.pkl"
# # #target_savename2 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp007_d_val.pkl"
# analysis_metrics.save_pickle(d_val, savename2)
# print("Saved Validation Data")

# savename3 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp007_d_test.pkl"
# # #target_savename3 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp007_d_test.pkl"
# analysis_metrics.save_pickle(d_test, savename3)
# print("Saved Testing Data")

s_dict_savename1 = str(config["inputs_dir"]) + str(config["expname"]) + '_d_train.pkl'
s_dict_savename2 = str(config["inputs_dir"]) + str(config["expname"]) + '_d_val.pkl'
s_dict_savename3 = str(config["inputs_dir"]) + str(config["expname"]) + '_d_test.pkl'

# Open processed data filess
train_dat = analysis_metrics.load_pickle(s_dict_savename1)
val_dat = analysis_metrics.load_pickle(s_dict_savename2)
test_dat = analysis_metrics.load_pickle(s_dict_savename3)

print(test_dat["y"][500:530]) # test a small sample of the target data

# Confirm there are no nans in the data / adjust front and back cutoff values if needed
print(np.isnan(train_dat["x"]).any())
print(np.isnan(val_dat["x"]).any())
print(np.isnan(test_dat["x"]).any())

print(np.isnan(train_dat["y"]).any())
print(np.isnan(val_dat["y"]).any())
print(np.isnan(test_dat["y"]).any())

# ----------- Model Training ----------------------------------

# Setup the Data
front_cutoff = config["databuilder"]["front_cutoff"] # remove front nans : 74 ENSO - two front nans before daily interpolation = 60 days, daily interpolation takes 1/2 the original time step = 15 days TOTAL = ~75
back_cutoff = config["databuilder"]["back_cutoff"]  # remove back nans : 32 ~ 1 month of nans

trainset = data_loader.CustomData(config["data_loader"]["data_dir"] + "/Network Inputs/" + str(config["expname"]) + "_d_train.pkl", front_cutoff, back_cutoff)
valset = data_loader.CustomData(config["data_loader"]["data_dir"] + "/Network Inputs/" + str(config["expname"]) + "_d_val.pkl", front_cutoff, back_cutoff)
testset = data_loader.CustomData(config["data_loader"]["data_dir"] + "/Network Inputs/" + str(config["expname"]) + "_d_test.pkl", front_cutoff, back_cutoff)

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
    target_mean=trainset.target.mean(axis=0),
    target_std=trainset.target.std(axis=0),
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
torchinfo.summary(
    model,
    [   trainset.input[: config["data_loader"]["batch_size"]].shape ],
    verbose=1,
    col_names=("input_size", "output_size", "num_params"),
)

# Train the Model
model.to(device)
trainer.fit()

# Save the Model
# path = '/Users/C830793391/Documents/Research/E3SM/saved/models/exp007_v0.pth'
# torch.save(model.state_dict(), path)

# Load the Model
# path = '/Users/C830793391/Documents/Research/E3SM/saved/models/exp007_v0.pth'
# model = TorchModel(config=config["arch"])
# model.load_state_dict(torch.load(path))
# model.eval()

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
plt.show()

# ---------------- Model Evaluation ----------------------------------

with torch.inference_mode():
    print(device)
    output = model.predict(dataset=testset, batch_size=128, device=device) # The output is the batched SHASH distribution parameters

print(output[:20]) # look at a small sample of the output data

# Save Model Outputs
model_output = str(config["output_dir"]) + 'exp007_output_testset.pkl'
analysis_metrics.save_pickle(output, model_output)