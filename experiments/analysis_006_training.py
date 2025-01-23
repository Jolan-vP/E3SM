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
import analysis.calc_climatology as calc_climatology
from analysis import analysis_metrics

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"pytorch version = {torch.__version__}")

# https://github.com/victoresque/pytorch-template/tree/master

# ----CONFIG AND CLASS SETUP----------------------------------------------
config = utils.get_config("exp006")
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
    figure_dir=config["perlmutter_output_dir"],
    target_only = True, 
    fetch=False,
    verbose=False
)

# ----PROCESS E3SM DATA----------------------------------------------

# d_train, d_val, d_test = data.fetch_data()

# target_PRECT_savename1 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp006_d_train_PRECT_1850-2014_unlagged.pkl"
# # target_PRECT_savename1 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp006_d_train_SeattleRegional_PRECT_1850-2014_unlagged.pkl"
# with gzip.open(target_PRECT_savename1, "wb") as fp:
#     pickle.dump(d_train, fp)

# target_PRECT_savename2 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp006_d_val_PRECT_1850-2014_unlagged.pkl"
# # target_PRECT_savename2 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp006_d_val_SeattleRegional_PRECT_1850-2014_unlagged.pkl"
# with gzip.open(target_PRECT_savename2, "wb") as fp:
#     pickle.dump(d_val, fp)

# target_PRECT_savename3 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp006_d_test_PRECT_1850-2014_unlagged.pkl"
# # target_PRECT_savename3 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp006_d_test_SeattleRegional_PRECT_1850-2014_unlagged.pkl"
# with gzip.open(target_PRECT_savename3, "wb") as fp:
#     pickle.dump(d_test, fp)

# s_dict_train, s_dict_val, s_dict_test = multi_input_data_organizer(config, MJO = True, ENSO = True, other = False)

# ---OPEN DATA---------------------------------------------

s_dict_savename1 = str(config["perlmutter_data_dir"]) + 'presaved/exp006_train_unlagged_28.pkl'
# s_dict_savename1 = '/Users/C830793391/BIG_DATA/E3SM_Data/presaved/Network Inputs/exp006_train_unlagged.pkl'
# with gzip.open(s_dict_savename1, "wb") as fp:
#     pickle.dump(s_dict_train, fp)

s_dict_savename2 = str(config["perlmutter_data_dir"]) + 'presaved/exp006_val_unlagged_28.pkl'
# s_dict_savename2 = '/Users/C830793391/BIG_DATA/E3SM_Data/presaved/Network Inputs/exp006_val_unlagged.pkl'
# with gzip.open(s_dict_savename2, "wb") as fp:
#     pickle.dump(s_dict_val, fp)

s_dict_savename3 = str(config["perlmutter_data_dir"]) + 'presaved/exp006_test_unlagged_28.pkl'
# s_dict_savename3 = '/Users/C830793391/BIG_DATA/E3SM_Data/presaved/Network Inputs/exp006_test_unlagged.pkl'
# with gzip.open(s_dict_savename3, "wb") as fp:
#     pickle.dump(s_dict_test, fp)

with gzip.open(s_dict_savename1, "rb") as obj1:
    train_dat = pickle.load(obj1)
obj1.close()

with gzip.open(s_dict_savename2, "rb") as obj2:
    val_dat = pickle.load(obj2)
obj2.close()

with gzip.open(s_dict_savename3, "rb") as obj3:
    test_dat = pickle.load(obj3)
obj3.close()

# --- Setup the Data for Training ---------------------------------------------
lagtime = config["databuilder"]["lagtime"] 
smoothing_length = config["databuilder"]["averaging_length"]

trainset = data_loader.CustomData(config["data_loader"]["perlmutter_data_dir"] + "exp006_train_unlagged_28.pkl", lagtime, smoothing_length)
valset = data_loader.CustomData(config["data_loader"]["perlmutter_data_dir"] + "exp006_val_unlagged_28.pkl", lagtime, smoothing_length)
testset = data_loader.CustomData(config["data_loader"]["perlmutter_data_dir"] + "exp006_test_unlagged_28.pkl", lagtime, smoothing_length)

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

# --- Setup the Model ----------------------------------------------------
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

# # # Visualize the model
# torchinfo.summary(
#     model,
#     [   trainset.input[: config["data_loader"]["batch_size"]].shape ],
#     verbose=1,
#     col_names=("input_size", "output_size", "num_params"),
# )

# TRAIN THE MODEL
model.to(device)
trainer.fit()

# Save the Model
# path = '/Users/C830793391/Documents/Research/E3SM/saved/models/exp006_RERUN_Nov2024.pth'
path = str(config["perlmutter_model_dir"]) + 'exp006_RERUN_28.pth'
torch.save(model.state_dict(), path)

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
plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["exp_name"]) + '/training_metrics.png', format = 'png', dpi = 300)
plt.show()

# -----------------------------------------------------------------------
# PLOT PREDICTIONS AGAINST CLMIATOLOGY:

with torch.inference_mode():
    print(device)
    output = model.predict(dataset=testset, batch_size=128, device=device) # The output is the batched SHASH distribution parameters
output[:20]

# Save Model Outputs
model_output_pred = str(config["perlmutter_output_dir"]) + 'exp006_output_pred_testset_RERUN_28.pkl'
with gzip.open(model_output_pred, "wb") as fp:
    pickle.dump(output, fp)

    # Open Model Outputs
model_output_pred = str(config["perlmutter_output_dir"]) + 'exp006_output_pred_testset_RERUN_28.pkl'
with gzip.open(model_output_pred, "rb") as obj1:
    output = pickle.load(obj1)

# -------------------------------------------------------------------------

exp006_original_output = filemethods.open_data_file('/pscratch/sd/p/plutzner/E3SM/saved/output/exp006_output_pred_testset.pkl')

lagtime = config["databuilder"]["lagtime"] 
smoothing_length = config["databuilder"]["averaging_length"]  

# Open Target Data
# target = xr.open_dataset('/Users/C830793391/BIG_DATA/E3SM_Data/presaved/Network Inputs/' + str(config["expname"]) + '_d_test_1850-1900.nc')
target = filemethods.open_data_file(config["perlmutter_data_dir"] + '/presaved/exp006_test_unlagged_28.pkl')
target = target["y"][lagtime:]
target = target[smoothing_length:]

# Open Climatology Data: TRAINING DATA
# climatology_filename = '/Users/C830793391/BIG_DATA/E3SM_Data/presaved/Network Inputs/' + str(config["expname"]) + '_d_train_1850-1900.nc'
climatology_filename = str(config["perlmutter_data_dir"]) + '/presaved/exp006_train_unlagged_28.pkl'
climatology_da = analysis_metrics.load_pickle(climatology_filename)
climatology = climatology_da["y"][lagtime:]
climatology = climatology[smoothing_length:]

# Compare SHASH predictions to climatology histogram
x = np.arange(-15, 15, 0.01)

p = calc_climatology.deriveclimatology(exp006_original_output, climatology, x, number_of_samples=50, config=config, climate_data = climatology_filename)

p = calc_climatology.deriveclimatology(output, climatology, x, number_of_samples=50, config=config, climate_data = climatology_filename)