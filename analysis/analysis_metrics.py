"""
Metrics for Analyzing Model Outputs: 

Functions: ------------------

save_pickle(file, filename)

load_pickle(variable, filename)

climatology(output, target)

discard_plot(networkoutput, target, crps_scores, filepath)

anomalies_by_ENSO_phase(elnino, lanina, neutral, target, target_raw, sample_index, config)

spread_skill(output, target, config)

"""

import xarray as xr
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nc_time_axis
import cftime
from scipy import stats
from scipy import integrate
import scipy as scipy
from sklearn import datasets, model_selection
import importlib as imp
from glob import glob
import random
import shash.shash_torch
from shash.shash_torch import Shash
import pickle 
import gzip
from model.metric import iqr_basic
from shash.shash_torch import Shash
import torch
import xarray as xr

def save_pickle(variable, filename):
    with gzip.open(filename, "wb") as fp:
        pickle.dump(variable, fp)
    print("File saved as: ", filename)


def load_pickle(filename):
    with gzip.open(filename, "rb") as obj1:
        data = pickle.load(obj1)
    return data


def climatologyCDF(target, x, climatology_var = None):
    """
    test_target: 2D array of target values
    x: 1D array of bin edges
    """
    if climatology_var is not None: 
        # Create pdf distribution of climatology, calculate CDF, and repeat same CDF across all samples
        climatology = climatology_var
        # data = xr.open_dataset(data)
        # # with gzip.open(data, "rb") as obj1:
        # #     data = pickle.load(obj1)
        # climatology = data["y"] # pulling all target values from processed data
        print(f"Climatological Mean = {np.mean(climatology)}")
        print(f"Climatological Variance = {np.var(climatology)}")

    else: 
        climatology = target

    pdf, __ = np.histogram(climatology, bins = x, density=True)
    pdf = pdf / (np.sum(pdf) * np.diff(x)[0])
    cdf_base = np.cumsum(pdf) / np.sum(pdf)
    cdf_base = cdf_base[:len(x-1)]

    # Echo cdf_base across the depth of all samples: 
    climatology_array = np.tile(cdf_base, (len(target), 1))
    climatology_pdf = np.tile(pdf, (len(target)))
    # Flip climatology array to match the shape of the cdf_array
    climatology_array = np.transpose(climatology_array)
    print(climatology_array.shape)

    return climatology_array, climatology_pdf


def discard_plot(networkoutput, target, crps_scores, crps_climatology_scores, config, target_type = 'anomalous'):
    # iqr capture relies on SHASH output parameters (mu, sigma, tau, gamma) and the SHASH class
    iqr = iqr_basic(networkoutput)

    percentiles = np.linspace(100, 0, 101)

    avg_crps = []
    avg_target = []
    sample_index = np.zeros((len(target), len(percentiles)))
    for ip, p in enumerate(percentiles):
        avg_crps.append(np.mean(crps_scores[iqr < np.percentile(iqr, p)]))
        avg_target.append(np.mean(target[iqr < np.percentile(iqr, p)]))
        # capture the index (out of total) for all the samples in each bin
        indices = np.where(iqr < np.percentile(iqr, p))[0]
        sample_index[:len(indices), ip] = indices

    color = 'tab:blue'
    fig, ax1 = plt.subplots()
    plt.gca().invert_xaxis()
    ax1.set_ylabel('Average CRPS')
    ax1.set_xlabel('IQR Percentile (% Data Remaining)', color=color)
    ax1.plot(percentiles, avg_crps, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=crps_climatology_scores.mean(), color='grey', linestyle='--', label='CRPS Mean Climatology')
    # ax1.set_ylim([1.05, 1.205])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # legend
    ax1.legend(loc = 'lower left')

    if target_type == 'anomalous':
        color = 'tab:olive'
        ax2.set_ylabel('Average Target Anomalies (mm/day)', color=color)
        ax2.plot(percentiles, avg_target, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/CRPS_IQR_DiscardPlot_anomalies.png', format='png', bbox_inches ='tight', dpi = 300)

    elif target_type == 'raw':
        color = 'tab:olive'
        ax2.set_ylabel('Raw Target Values (mm/day)', color=color)
        ax2.plot(percentiles, avg_target, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/CRPS_IQR_DiscardPlot_true_precip.png', format='png', bbox_inches ='tight', dpi = 300)

    return sample_index

def anomalies_by_ENSO_phase(elnino, lanina, neutral, target, target_raw, sample_index, config):
    # Scatter Compare: 
    plt.figure(figsize=(13, 4))
    plt.scatter(sample_index[neutral,0], target[neutral], alpha = 0.8, label = 'Neutral target Anomalies (mm/day)', s=0.1, color = '#b0b0b0')
    plt.scatter(sample_index[elnino,0], target[elnino], alpha = 0.8, label = 'El Nino target Anomalies (mm/day)', s=0.1, color = '#648FFF')
    plt.scatter(sample_index[lanina,0], target[lanina], alpha = 0.8, label = 'La Nina target Anomalies (mm/day)', s=0.1, color = '#FFB000')
    plt.xlabel('Time \n (Daily Samples in Chronological Order)')
    plt.ylabel('Precipitation Anomalies (mm/day)')
    plt.legend(markerscale = 22, loc = 'upper right')
    plt.ylim(-7, 22)

    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/scatter_ENSO_phases.png', format='png', bbox_inches ='tight', dpi = 300)

    print(f"Mean Anomaly during El Nino: {np.round(target[elnino].mean(), 4)}")
    print(f"Mean Anomaly during La Nina: {np.round(target[lanina].mean(), 4)}")
    print(f"Mean Anomaly during Neutral: {np.round(target[neutral].mean(), 4)}")

    print(f"Mean True Amount during El Nino: {np.round(target_raw[elnino].mean(), 4)}")
    print(f"Mean True Amount during La Nina: {np.round(target_raw[lanina].mean(), 4)}")
    print(f"Mean True Amount during Neutral: {np.round(target_raw[neutral].mean(), 4)}")

    print(f"Percent of La Nina events out of total: {np.round((lanina.shape[0]/target.shape[0])*100, 2)}%")
    print(f"Percent of El Nino events out of total: {np.round((elnino.shape[0]/target.shape[0])*100, 2)}%")
    print(f"Percent of Neutral events out of total: {np.round((neutral.shape[0]/target.shape[0])*100, 2)}%")



def spread_skill(output, target, config):

    # Prepare SHASH output for analysis ----------------------------------
    tensor_list = []
    for i in range(output.shape[1]):
        tensor_list.append(torch.from_numpy(output[:, i]))

    output_tensor = torch.stack(tensor_list, dim=1)

    output_shash_instance = Shash(output)
    network_std_tensor = output_shash_instance.std()
    network_mean_tensor = output_shash_instance.mean()

    # Convert back to numpy: 
    network_std = network_std_tensor.numpy()
    network_mean = network_mean_tensor.numpy()

    network_rmse = np.sqrt( ((target - network_mean) **2) / len(target) )

    # X-Axis: Standard Deviation of model's predicted distribution
    # Y-Axis: RMSE of model's predicted distribution
    # Compute spread skill ratio and compare to climatology's Spread-Skill Ratio

    ## Climatology: 
    climatology_mean = target.mean()
    climatology_std = np.std(target.values)
    # print(f"climatology_std : {climatology_std}")

    climatology_std = np.repeat(climatology_std, len(target), axis = 0)
    climatology_rmse = np.sqrt( ((target - climatology_mean) **2) / len(target) )

    num_bins = 12
    percentile_bins = np.percentile(network_std, np.linspace(0, 100, num_bins + 1))
    # print(f"percentile_bins: {percentile_bins}")
    network_bin_indices = np.digitize(network_std, percentile_bins)
    # print(f"network_bin_indices: {network_bin_indices}")

    # Calculate the mean values for each bin
    network_std_binned = []
    network_rmse_binned = []

    for i in range(1, num_bins + 1):
        bin_mask = network_bin_indices == i
        # print(f"bin_mask: {bin_mask}")
        if np.any(bin_mask):
            network_std_binned.append(network_std[bin_mask].mean())
            network_rmse_binned.append(network_rmse[bin_mask].mean())

    # Repeat for climatology data
    climatology_bin_indices = np.digitize(climatology_std, percentile_bins)

    climatology_std_binned = []
    climatology_rmse_binned = []

    for i in range(1, num_bins + 1):
        bin_mask = climatology_bin_indices == i
        if np.any(bin_mask):
            climatology_std_binned.append(climatology_std[bin_mask].mean())
            climatology_rmse_binned.append(climatology_rmse[bin_mask].mean())

    # 1:1 reference line
    max_val = max(max(network_std_binned), max(network_rmse_binned), max(climatology_std_binned), max(climatology_rmse_binned))

    plt.figure()
    plt.plot([0, max_val], [0, max_val], 'k--', color = 'gray', label='1:1 Reference Line')
    plt.plot(network_std_binned, network_rmse_binned, 'o-', markersize=5, label='Network')
    plt.plot(climatology_std_binned, climatology_rmse_binned, 'o-', markersize=5, label='Climatology')
    plt.xlabel('Spread (Standard Deviation)')
    plt.ylabel('Skill (RMSE)')
    plt.legend()
    plt.show()

    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/SpreadSkillRatio_network_vs_climatology.png', format='png', bbox_inches ='tight', dpi = 300)