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
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import utils.filemethods as filemethods
import math
import datetime


def save_pickle(variable, filename):
    with gzip.open(filename, "wb") as fp:
        pickle.dump(variable, fp)
    print("File saved as: ", filename)


def load_pickle(filename):
    try:
        # Try to open as a gzip-compressed file
        with gzip.open(filename, "rb") as f:
            data = pickle.load(f)
    except gzip.BadGzipFile:
        # If not gzip-compressed, open as a regular file
        with open(filename, "rb") as f:
            data = pickle.load(f)
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


def IQRdiscard_plot(networkoutput, target, crps_scores, crps_climatology_scores, dates, config, target_type = 'anomalous', keyword = None, analyze_months = True, most_confident = True):

    if keyword != "All Samples":
        selected_target = target.sel(time = dates)

        all_timestamps = target.time.values
        selected_timestamps = dates.values

        time_indices = np.nonzero(np.isin(all_timestamps, selected_timestamps))[0]

        print(f"time indices: {time_indices}")
        selected_networkoutput = networkoutput[time_indices]
        crps_network = crps_scores[time_indices]
        crps_climo = crps_climatology_scores[time_indices]
        print(f"dates: {dates[:5]}")
        print(f"dates indices: {time_indices[:5]}")
    else: 
        selected_networkoutput = networkoutput
        selected_target = target
        crps_network = crps_scores
        crps_climo = crps_climatology_scores

    # iqr capture relies on SHASH output parameters (mu, sigma, tau, gamma) and the SHASH class
    iqr = iqr_basic(selected_networkoutput)
    percentiles = np.linspace(100, 0, 21)

    avg_crps = []
    avg_target = []
    avg_iqr = []
    sample_index = np.zeros((len(selected_target), len(percentiles)))
    
    for ip, p in enumerate(percentiles):
        if most_confident == True:
            confidence_label = " increasing_confidence"
            avg_crps.append(np.mean(crps_network[iqr < np.percentile(iqr, p)]))
            avg_target.append(np.mean(selected_target[iqr < np.percentile(iqr, p)]))
            # capture the index (out of total) for all the samples in each bin
            indices = np.where(iqr < np.percentile(iqr, p))[0]
            avg_iqr.append(np.mean(iqr[iqr < np.percentile(iqr, p)]))

        elif most_confident == False:
            confidence_label = " decreasing_confidence"
            avg_crps.append(np.mean(crps_network[iqr >= np.percentile(iqr, p)]))
            avg_target.append(np.mean(selected_target[iqr >= np.percentile(iqr, p)]))
            # capture the index (out of total) for all the samples in each bin
            indices = np.where(iqr >= np.percentile(iqr, p))[0]
            avg_iqr.append(np.mean(iqr[iqr >= np.percentile(iqr, p)]))
            # print(f"Low-confidence indices for percentile {p}: {indices}")

        sample_index[:len(indices), ip] = indices

    plt.figure()
    plt.plot(percentiles, avg_iqr, color='tab:orange', label='IQR')
    plt.xlabel('IQR Percentile (% Data Remaining)')
    plt.ylabel('IQR')
    plt.title(f'IQR Discard Plot \n {keyword}')
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + str(confidence_label) + '_IQR_value_check.png', format='png', bbox_inches ='tight', dpi = 300)
    plt.close()

    color = 'tab:blue'
    fig, ax1 = plt.subplots()
    plt.gca().invert_xaxis()
    ax1.set_ylabel('Average CRPS')
    ax1.set_xlabel('IQR Percentile (% Data Remaining)', color=color)
    ax1.plot(percentiles, avg_crps, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=crps_climo.mean(), color='grey', linestyle='--', label='CRPS Mean Climatology')
    min = np.nanmin(avg_crps) - .08
    max = np.nanmax(avg_crps) + .1
    ax1.set_ylim([min, max])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    if target_type == 'anomalous':
        color = 'tab:olive'
        ax2.set_ylabel('Average Target Anomalies (mm/day)', color=color)
        ax2.plot(percentiles, avg_target, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        if most_confident == True:
            plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_CRPS_narrowIQR_DiscardPlot_anomalies.png', format='png', bbox_inches ='tight', dpi = 300)
            print(f'path: {str(config["perlmutter_figure_dir"]) + str(config["expname"]) + "/" + str(keyword) + "_CRPS_narrowIQR_DiscardPlot_anomalies.png"}')
            plt.close()
        else: 
            plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_CRPS_wideIQR_DiscardPlot_anomalies.png', format='png', bbox_inches ='tight', dpi = 300)
            print(f'path : {str(config["perlmutter_figure_dir"]) + str(config["expname"]) + "/" + str(keyword) + "_CRPS_wideIQR_DiscardPlot_anomalies.png"}')
            plt.close()

    elif target_type == 'raw':
        color = 'tab:olive'
        ax2.set_ylabel('Raw Target Values (mm/day)', color=color)
        ax2.plot(percentiles, avg_target, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        if most_confident == True:
            plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_CRPS_narrowIQR_DiscardPlot_raw.png', format='png', bbox_inches ='tight', dpi = 300)
            print(f'path: {str(config["perlmutter_figure_dir"]) + str(config["expname"]) + "/" + str(keyword) + "_CRPS_narrowIQR_DiscardPlot_raw.png"}')
            plt.close()
        else: 
            plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + "/" + str(keyword) + '_CRPS_wideIQR_DiscardPlot_raw.png', format='png', bbox_inches ='tight', dpi = 300)
            plt.close()
        
    return sample_index


def target_discardplot(targetCNN, targetSNN, CNNcrps_scores, NNcrps_scores, crps_climatology_scores, config, target_type = 'anomalous', keyword = None):
    
    percentiles = np.linspace(100, 0, 21)

    avg_crpsCNN = []
    avg_crpsNN = []
    avg_crpsClimo = []
    avg_target = []
    sample_indexCNN = np.zeros((len(targetCNN), len(percentiles)))
    sample_indexSNN = np.zeros((len(targetSNN), len(percentiles)))

    for ip, p in enumerate(percentiles):
        # for each percentile, calculate average CRPS score in the bin, binned by corresponding target value
        avg_crpsCNN.append(np.mean(CNNcrps_scores[targetCNN < np.percentile(targetCNN, p)]))
        avg_crpsNN.append(np.mean(NNcrps_scores[targetSNN < np.percentile(targetSNN, p)]))
        if config["arch"]["type"] == "cnn":
            avg_crpsClimo.append(np.mean(crps_climatology_scores[targetCNN < np.percentile(targetCNN, p)]))
            avg_target.append(np.mean(targetCNN[targetCNN < np.percentile(targetCNN, p)]))
            indices = np.where(targetCNN < np.percentile(targetCNN, p))[0]
            sample_indexCNN[:len(indices), ip] = indices
        else: 
            avg_crpsClimo.append(np.mean(crps_climatology_scores[targetSNN < np.percentile(targetSNN, p)]))
            avg_target.append(np.mean(targetSNN[targetSNN < np.percentile(targetSNN, p)]))
            # capture the index (out of total) for all the samples in each bin
            indices = np.where(targetSNN < np.percentile(targetSNN, p))[0]
            sample_indexSNN[:len(indices), ip] = indices

    # save avg_crpsCNN, avg_crpsNN, avg_crpsClimo, avg_target in one pickle file
    # save_pickle(avg_crpsCNN, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/avg_crpsCNN_percentiles.pkl')
    # save_pickle(avg_crpsNN, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/avg_crpsNN_percentiles.pkl')
    # save_pickle(avg_crpsClimo, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/avg_crpsClimo_percentiles.pkl')
    # save_pickle(avg_target, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/avg_target_percentiles.pkl')

    colors = ['#fca50a', '#bc3754', '#6a176e', '#420a68']
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Average CRPS')
    ax1.set_xlabel('Percentile of Target Value \n (% Data Remaining)')
    ax1.plot(percentiles, avg_crpsCNN, color=colors[0], linewidth = 1.3, label = 'CNN')
    ax1.plot(percentiles, avg_crpsNN,  color=colors[1], linewidth = 1.3, label = 'Simple NN') 
    ax1.plot(percentiles, avg_crpsClimo, color=colors[2], linewidth = 1.3, label = 'Climatology')
    ax1.set_title(f'Discard Plot \n CRPS vs. Decreasing Target Value ({config["expname"]})')

    ax1.tick_params(axis='y')
    ax1.axhline(y=crps_climatology_scores.mean(), color='grey', linestyle='--', label='CRPS Mean Climatology')
    min = np.nanmin(avg_crpsCNN) - .08
    max = np.nanmax(avg_crpsClimo) + .1
    # ax1.set_ylim([min, 3])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    if target_type == 'anomalous':
        color = 'tab:olive'
        ax2.set_ylabel('Average Target Anomalies \n (mm/day)', color=colors[3])
        ax2.plot(percentiles, avg_target, color=colors[3], linestyle = 'dotted', label = 'Target Anomalies')
        ax2.tick_params(axis='y', labelcolor=colors[3])
    
    elif target_type == 'raw':
        color = 'tab:olive'
        ax2.set_ylabel('Raw Target Values (mm/day)', color=colors[3])
        ax2.plot(percentiles, avg_target, color=colors[3], linestyle = 'dotted', label = 'True Target Values')
        ax2.tick_params(axis='y', labelcolor=colors[3])

    # Get handles and labels
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    # Combine handles and labels
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    # Create the combined legend
    ax1.legend(all_handles, all_labels, loc='upper right', bbox_to_anchor = (1.6, 1.015))

    plt.gca().invert_xaxis()
    

    if target_type == 'anomalous':
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_CRPS_TargMag_DiscardPlot_anomalies_' + str(config["expname"]) + '.png', format='png', bbox_inches ='tight', dpi = 300)
        plt.close()
    else: 
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_CRPS_TargMag_DiscardPlot_raw_' + str(config["expname"]) + '.png', format='png', bbox_inches ='tight', dpi = 300)
        plt.close()

def anomalies_by_ENSO_phase(elnino, lanina, neutral, target, target_raw, sample_index, config, keyword = None):
    # Scatter Compare: 
    plt.figure(figsize=(13, 4))
    plt.scatter(sample_index[neutral,0], target[neutral], alpha = 0.8, label = 'Neutral target Anomalies (mm/day)', s=0.1, color = '#b0b0b0')
    plt.scatter(sample_index[elnino,0], target[elnino], alpha = 0.8, label = 'El Nino target Anomalies (mm/day)', s=0.1, color = '#648FFF')
    plt.scatter(sample_index[lanina,0], target[lanina], alpha = 0.8, label = 'La Nina target Anomalies (mm/day)', s=0.1, color = '#FFB000')
    plt.xlabel('Time \n (Daily Samples in Chronological Order)')
    plt.ylabel('Precipitation Anomalies (mm/day)')
    plt.legend(markerscale = 22, loc = 'upper right')
    plt.ylim(-7, 22)

    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/scatter_ENSO_phases_' + str(keyword) + '.png', format='png', bbox_inches ='tight', dpi = 300)
    plt.close()

    print(f"Mean Anomaly during El Nino: {np.round(target[elnino].mean().item(), 4)}")
    print(f"Mean Anomaly during La Nina: {np.round(target[lanina].mean().item(), 4)}")
    print(f"Mean Anomaly during Neutral: {np.round(target[neutral].mean().item(), 4)}")

    print(f"Mean True Amount during El Nino: {np.round(target_raw[elnino].mean().item(), 4)}")
    print(f"Mean True Amount during La Nina: {np.round(target_raw[lanina].mean().item(), 4)}")
    print(f"Mean True Amount during Neutral: {np.round(target_raw[neutral].mean().item(), 4)}")

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
    plt.close()

def subsetanalysis_SHASH_ENSO(sample_index, shash_params, climatology, target, target_raw, config, x_values, subset_keyword = None): 
    # open saved ENSO phase indices and CRPS scores for all samples
    ENSO_CRPS_dict = load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "ENSO_indices_CRPS.pkl")

    elnino = np.array(ENSO_CRPS_dict["elnino"], dtype = int)
    lanina = np.array(ENSO_CRPS_dict["lanina"], dtype = int)
    neutral = np.array(ENSO_CRPS_dict["neutral"], dtype = int)
    crps_scores = load_pickle(str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "_CRPS_network_values.pkl")

    # # Select subset of samples based on percentage
    # subset_indices = sample_index[:, 100 - percentage]
    # # remove zeros from subset_indices
    # subset_indices = subset_indices[subset_indices != 0].astype(int)

    subset_indices = sample_index
    
    # Calculate relative ratio of ENSO phases relative to all samples
    elnino_ratio = elnino.shape[1] / crps_scores.shape[0]
    lanina_ratio = lanina.shape[1] / crps_scores.shape[0]
    neutral_ratio = neutral.shape[1] / crps_scores.shape[0]

    # identify enso phases of each sample in the subset
    sub_elnino = elnino[np.isin(elnino, subset_indices)]
    sub_lanina = lanina[np.isin(lanina, subset_indices)]
    sub_neutral = neutral[np.isin(neutral, subset_indices)]

    sub_elnino_dates = target.time.isel(time = sub_elnino)
    sub_lanina_dates = target.time.isel(time = sub_lanina)
    sub_neutral_dates = target.time.isel(time = sub_neutral)

    sub_elnino_ratio = len(sub_elnino) / len(subset_indices)
    sub_lanina_ratio = len(sub_lanina) / len(subset_indices)
    sub_neutral_ratio = len(sub_neutral) / len(subset_indices)
    max_ratio = max(sub_elnino_ratio, sub_lanina_ratio, sub_neutral_ratio)

    plt.figure(figsize = (5, 6))
    plt.bar(['El Nino', 'La Nina', 'Neutral'], [sub_elnino_ratio, sub_lanina_ratio, sub_neutral_ratio], color = ['#23888e', '#23888e', '#23888e'])
    plt.axhline(y=elnino_ratio, color='k',  xmin = 0.05, xmax = 0.30 , linestyle='--', linewidth=1.5, label='Ratio of ENSO Phase in All Samples')
    plt.axhline(y=lanina_ratio, color='k',  xmin = 0.37, xmax = .631, linestyle='--', linewidth=1.5)
    plt.axhline(y=neutral_ratio, color='k', xmin = .70, xmax = 0.95,  linestyle='--', linewidth=1.5)
    plt.xlabel('ENSO Phase')
    plt.ylabel('Frequency')
    plt.ylim([0, max_ratio + 0.075])
    plt.title(f'ENSO Phase Distribution for {str(subset_keyword)} Samples')
    plt.legend()
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + str(subset_keyword) + '_ENSO_phase_distribution.png', format='png', bbox_inches ='tight', dpi = 300)
    plt.close()

    # Print average target value per ENSO phase
    print(f"Average Target Anomaly during El Nino: {np.round(target[sub_elnino].mean().item(), 4)}")
    print(f"Average Target Anomaly during La Nina: {np.round(target[sub_lanina].mean().item(), 4)}")
    print(f"Average Target Anomaly during Neutral: {np.round(target[sub_neutral].mean().item(), 4)}")

    print(f"Average True Amount during El Nino: {np.round(target_raw[sub_elnino].mean().item(), 4)}")
    print(f"Average True Amount during La Nina: {np.round(target_raw[sub_lanina].mean().item(), 4)}")
    print(f"Average True Amount during Neutral: {np.round(target_raw[sub_neutral].mean().item(), 4)}")

    # identify CRPS scores of each sample in the subset
    sub_CRPS = crps_scores[subset_indices]

    plt.figure()
    plt.hist(sub_CRPS, bins = 50, density = False)
    plt.xlabel('CRPS')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of CRPS for {str(subset_keyword)} Samples')
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + str(subset_keyword) + '_CRPS_distribution.png', format='png', bbox_inches ='tight', dpi = 300)
    plt.close()

    # identify shash params of each sample in the subset
    sub_params = shash_params[subset_indices]

    # Plot SHASH curves for all samples in the subset
    dists = Shash(sub_params)
    pdf = dists.prob(x_values).numpy()

    plt.figure(figsize=(8, 4), dpi=200)
    plt.hist(
        climatology, x_values, density=True, color="silver", alpha=0.75, label="climatology"
    )

    plt.plot(x_values, pdf, linewidth = 0.5 ) #label = samples
    plt.xlabel("precipitation anomaly (mm/day)")
    plt.ylabel("probability density")
    plt.title(str(subset_keyword) + ' SHASH Curves -' + str(config["expname"]) )
    # plt.axvline(valset[:len(output)], color='r', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + str(subset_keyword) + '_SHASHs_w_climatology.png', format='png', bbox_inches ='tight', dpi = 300)
    plt.xlim([-10, 12])
    plt.close()

    return sub_elnino_dates, sub_lanina_dates, sub_neutral_dates

def compositemapping(indices, mapinputs, config, keyword = None): 
    """
    Take in two sets of indices, and create two composite maps based the two indices sets
    mapinputs is the input map data for all samples 
    """

    if len(mapinputs.shape) == 3: # Time, Lat, Lon
        # Indices/dates already correspond with VERIFICATION DAY so we will leave them as they are
        
        if isinstance(indices, xr.DataArray):
            # For each gridpoint in the icomposite maps, calculate the standard deviation at each gridpoint before they are time averaged
            icomposite_z500_std = mapinputs.sel(time=indices).std(dim='time')
            icomposite_z500 = mapinputs.sel(time=indices).mean(dim='time')  
            icomposite_z500_norm = icomposite_z500 / icomposite_z500_std
        else:
            # For each gridpoint in the icomposite maps, calculate the standard deviation at each gridpoint before they are time averaged
            icomposite_z500_std = mapinputs.isel(time=indices).std(dim='time')
            icomposite_z500 = mapinputs.isel(time=indices).mean(dim='time')  
            icomposite_z500_norm = icomposite_z500 / icomposite_z500_std

        # lats = np.linspace(-89.5, 89.5, 180) 
        # lats = mapinputs.lat
        # lons = mapinputs.lon
        # extent = [lons.min(), lons.max(), lats.min(), lats.max()]

        # if icomposite_z500_norm.max().item() > np.abs(icomposite_z500_norm.min().item()):
        #     vmaxz = icomposite_z500_norm.max().item() 
        #     vminz = (-1 * vmaxz) 
        # else: 
        #     vminz = icomposite_z500_norm.min().item() 
        #     vmaxz = (-1 * vminz) 

        plt.figure()
        fig, ax = plt.subplots(1, 1, figsize=(14, 8),  subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

        cf1 = ax.pcolormesh(icomposite_z500.lon, icomposite_z500.lat, icomposite_z500, cmap='PuOr_r', transform=ccrs.PlateCarree()) #, vmin = vminz, vmax = vmaxz )
        ax.set_title(f'Normalized Z500 Composite Map \n {keyword} Predictions')
        ax.coastlines()
        ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        # add colorbar that is same for both plots
        cbar1 = fig.colorbar(cf1, cmap='PuOr_r', ax=ax, orientation='vertical', fraction=0.01, pad=0.03)
        cbar1.set_label('Normalized Z500 Anomalies (sigma)')
        #cbar1.set_label('Precipitation Anomalies \n (mm/day)')

        # fig.tight_layout()
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_composite_maps.png', format='png', bbox_inches ='tight', dpi = 300)
        plt.close()

    elif len(mapinputs.shape) == 4: # Time, Lat, Lon, Variables
        # Indices/dates already correspond with VERIFICATION DAY so we will leave them as they are
        # if it's an xarry object, do __
        if isinstance(indices, xr.DataArray):
             # For each gridpoint in the icomposite maps, calculate the standard deviation at each gridpoint before they are time averaged
            icomposite_prect_std = mapinputs[..., 0].sel(time=indices).std(dim='time')
            icomposite_ts_std = mapinputs[..., 1].sel(time=indices).std(dim='time')

            icomposite_prect = mapinputs[..., 0].sel(time=indices).mean(dim='time')  
            icomposite_ts = mapinputs[..., 1].sel(time=indices).mean(dim='time')  

            icomposite_prect_norm = icomposite_prect / icomposite_prect_std
            icomposite_ts_norm = icomposite_ts / icomposite_ts_std

        else: 
            # if indices are integers : 
            # For each gridpoint in the icomposite maps, calculate the standard deviation at each gridpoint before they are time averaged
            icomposite_prect_std = mapinputs[..., 0].isel(time=indices).std(dim='time')
            icomposite_ts_std = mapinputs[..., 1].isel(time=indices).std(dim='time')

            icomposite_prect = mapinputs[..., 0].isel(time=indices).mean(dim='time')  
            icomposite_ts = mapinputs[..., 1].isel(time=indices).mean(dim='time')  

            icomposite_prect_norm = icomposite_prect / icomposite_prect_std
            icomposite_ts_norm = icomposite_ts / icomposite_ts_std

        lats = np.linspace(-89.5, 89.5, 180) 
        lons = mapinputs.lon

        if icomposite_prect_norm.max().item() > np.abs(icomposite_prect_norm.min().item()):
            vmaxp = icomposite_prect_norm.max().item()
            vminp = -1 * vmaxp
        else: 
            vminp = icomposite_prect_norm.min().item()
            vmaxp = -1 * vminp

        if icomposite_ts_norm.max().item() > np.abs(icomposite_ts_norm.min().item()):
            vmaxt = icomposite_ts_norm.max().item()
            vmint = -1 * vmaxt
        else:
            vmint = icomposite_ts_norm.min().item()
            vmaxt = -1 * vmint

        # print(f"Composite Data: {icomposite_prect}")
        # print(f"composite mean: {np.mean(icomposite_prect).item()}")
        # print(f"composite min: {np.min(icomposite_ts).item()}")
        # print(f"composite max: {np.max(icomposite_ts).item()}")

        plt.figure()
        fig, ax = plt.subplots(2, 1, figsize=(14, 8),  subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

        cf1 = ax[0].pcolormesh(lons, lats, icomposite_prect_norm, cmap='BrBG', transform=ccrs.PlateCarree(), vmin = vminp, vmax = vmaxp )
        ax[0].set_title(f'Normalized Precipitation Composite Map \n {keyword} Predictions')
        ax[0].coastlines()
        ax[0].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax[0].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        cf2 = ax[1].pcolormesh(lons, lats, icomposite_ts_norm, cmap='RdBu_r', transform=ccrs.PlateCarree(), vmin = vmint, vmax = vmaxt )
        ax[1].set_title(f'Normalized Temperature Composite Map \n {keyword} Predictions')
        ax[1].coastlines()
        ax[1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax[1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        # add colorbar that is same for both plots
        cbar1 = fig.colorbar(cf1, cmap='BrBG', ax=ax[0], orientation='vertical', fraction=0.01, pad=0.03)
        cbar1.set_label('Normalized Precipitation Anomalies (sigma)')
        #cbar1.set_label('Precipitation Anomalies \n (mm/day)')

        cbar2 = fig.colorbar(cf2, cmap='BrBG', ax=ax[1], orientation='vertical', fraction=0.01, pad=0.03)
        cbar2.set_label('Normalized Temperature Anomalies (sigma)')
        #cbar2.set_label('Temperature Anomalies \n (deg C)')

        # fig.tight_layout()
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_composite_maps.png', format='png', bbox_inches ='tight', dpi = 300)
        plt.close()

def differenceplot(indices1, indices2, mapinputs, crps_network, config, normalized = False, keyword = None):
    """
    Take in two sets of indices, and create a difference map based on the two indices sets
    mapinputs is the input map data for all samples 
    """
    if len(mapinputs.shape) == 3: # Time, Lat, Lon 
        # Ensure indices are arrays of integers
        indices1array = np.array(indices1, dtype=int)
        indices2array = np.array(indices2, dtype=int)

        if normalized == True:
            # Normalize maps by std of each grid point before time averaging
            icomposite1_std = mapinputs.isel(time=indices1array).std(dim='time')
            icomposite2_std = mapinputs.isel(time=indices2array).std(dim='time')

            icomposite1 = mapinputs.isel(time=indices1array).mean(dim='time')  
            icomposite2 = mapinputs.isel(time=indices2array).mean(dim='time')  

            # Normalize by dividing by std at each gridpoint
            icomposite1 = icomposite1 / icomposite1_std
            icomposite2 = icomposite2 / icomposite2_std

            diff = icomposite1 - icomposite2

            title_label = 'Normalized Anomalies (sigma)'
        else: 
            icomposite1 = mapinputs.isel(time=indices1array).mean(dim='time')  
            icomposite2 = mapinputs.isel(time=indices2array).mean(dim='time')  

            diff = icomposite1 - icomposite2

            title_label = 'Anomalies'

        # Calculate CRPS per grouping: 
        icomp1_crps = np.mean(crps_network[indices1array])
        icomp2_crps = np.mean(crps_network[indices2array])

        plt.figure()
        plt.hist(crps_network[indices1array], bins = 20, density = False, alpha = 0.5, label = f'High Confidence[{len(indices1array)}]')
        plt.hist(crps_network[indices2array], bins = 20, density = False, alpha = 0.5, label = f'Low Confidence[{len(indices2array)}]')
        plt.xlabel('CRPS')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of CRPS for {keyword} Samples')
        plt.legend()
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_CRPS_distribution.png', format='png', bbox_inches ='tight', dpi = 300)
        plt.close()

        if diff.max().item() > np.abs(diff.min().item()):
            vmaxz = diff.max().item()
            vminz = (-1 * vmaxz)
        else: 
            vminz = diff.min().item()
            vmaxz = (-1 * vminz)

        plt.figure()
        fig, ax = plt.subplots(1, 3, figsize=(18, 8),  subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

        cf0 = ax[0].pcolormesh(icomposite1.lon, icomposite1.lat, icomposite1, cmap='PuOr_r', transform=ccrs.PlateCarree(), vmin = vminz, vmax = vmaxz)
        cf1 = ax[1].pcolormesh(icomposite2.lon, icomposite2.lat, icomposite2, cmap='PuOr_r', transform=ccrs.PlateCarree(), vmin = vminz, vmax = vmaxz)
        cf2 = ax[2].pcolormesh(diff.lon, diff.lat, diff, cmap='PuOr_r', transform=ccrs.PlateCarree(), vmin = vminz, vmax = vmaxz )

        for i in range(3): 
            ax[i].coastlines()
            ax[i].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
            ax[i].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
            ax[1].legend(loc='lower left', markerscale = 22)

        ax[0].set_title(f'Low CRPS High Confidence Predictions')
        ax[1].set_title(f'Low CRPS Low Confidence Predictions')
        ax[2].set_title(f'Difference Map Predictions')

        ax[0].text(
            0.02, 0.02, f'Mean CRPS: {icomp1_crps:.3f}',
            transform=ax[0].transAxes, ha='left', va='bottom', fontsize=12, color='black'
        )
        ax[1].text(
            0.02, 0.02, f'Mean CRPS: {icomp2_crps:.3f}',
            transform=ax[1].transAxes, ha='left', va='bottom', fontsize=12, color='black'
        )

        # add colorbar that is same for both plots
        cbar1 = fig.colorbar(cf1, cmap='PuOr_r', ax=ax, orientation='vertical', fraction=0.01, pad=0.03)
        cbar1.set_label(str(title_label))
        #cbar1.set_label('Precipitation Anomalies \n (mm/day)')

        # plt.tight_layout()
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_difference_map.png', format='png', bbox_inches ='tight', dpi = 250)
        plt.close(fig)  # Close the figure to avoid memory issues

    elif len(mapinputs.shape) == 4: # Time, Lat, Lon, Variables 
        # Ensure indices are arrays of integers
        indices1 = np.array(indices1, dtype=int)
        indices2 = np.array(indices2, dtype=int)

        if normalized == True:
            # Normalize maps by std of each grid point before time averaging
            icomposite_prect1_std = mapinputs[..., 0].isel(time=indices1).std(dim='time')
            icomposite_ts1_std = mapinputs[..., 1].isel(time=indices1).std(dim='time')

            icomposite_prect2_std = mapinputs[..., 0].isel(time=indices2).std(dim='time')
            icomposite_ts2_std = mapinputs[..., 1].isel(time=indices2).std(dim='time')

            icomposite_prect1 = mapinputs[..., 0].isel(time=indices1).mean(dim='time')  
            icomposite_ts1 = mapinputs[..., 1].isel(time=indices1).mean(dim='time')  

            icomposite_prect2 = mapinputs[..., 0].isel(time=indices2).mean(dim='time')  
            icomposite_ts2 = mapinputs[..., 1].isel(time=indices2).mean(dim='time')  

            # Normalize by dividing by std at each gridpoint
            icomposite_prect1 = icomposite_prect1 / icomposite_prect1_std
            icomposite_ts1 = icomposite_ts1 / icomposite_ts1_std
            icomposite_prect2 = icomposite_prect2 / icomposite_prect2_std
            icomposite_ts2 = icomposite_ts2 / icomposite_ts2_std

            prect_diff = icomposite_prect1 - icomposite_prect2
            ts_diff = icomposite_ts1 - icomposite_ts2
            
            title_label = 'Normalized'
        else: 
            icomposite_prect1 = mapinputs[..., 0].isel(time=indices1).mean(dim='time')  
            icomposite_ts1 = mapinputs[..., 1].isel(time=indices1).mean(dim='time')  

            icomposite_prect2 = mapinputs[..., 0].isel(time=indices2).mean(dim='time')  
            icomposite_ts2 = mapinputs[..., 1].isel(time=indices2).mean(dim='time')  

            prect_diff = icomposite_prect1 - icomposite_prect2
            ts_diff = icomposite_ts1 - icomposite_ts2

            title_label = ''
        # Isolate the first 7 characters of the keyword
        keyword_phase = keyword[:7]
        
        plots_array_prect = [icomposite_prect1, icomposite_prect2, prect_diff]
        plots_array_skintemp = [icomposite_ts1, icomposite_ts2, ts_diff]
        plots_dict = {0: plots_array_prect, 1: plots_array_skintemp}
        label_dict = {"Precipitation": ("High Confidence", "Low Confidence", "(High-Low) Difference"), 
                  "Skin Temperature": ("High Confidence", "Low Confidence", "(High-Low) Difference")}

        # Calculate vmin and vmax for the first column in each row (centered around zero)
        row_0_first_col = plots_dict[0][0]  # [0, 0]
        row_1_first_col = plots_dict[1][0]  # [1, 0]
        row_0_first_col_max = row_0_first_col.max().item()
        row_0_first_col_min = -row_0_first_col_max  # Centered at zero
        row_1_first_col_max = row_1_first_col.max().item()
        row_1_first_col_min = -row_1_first_col_max  # Centered at zero

        # Calculate vmin and vmax for the second and third columns in each row
        row_0_data = [plots_dict[0][1], plots_dict[0][2]]  # [0, 1] and [0, 2]
        row_1_data = [plots_dict[1][1], plots_dict[1][2]]  # [1, 1] and [1, 2]
        row_0_max = max(data.max().item() for data in row_0_data)
        row_0_min = -row_0_max  # Centered at zero
        row_1_max = max(data.max().item() for data in row_1_data)
        row_1_min = -row_1_max  # Centered at zero

        # Apply a scaling factor to make the colors more intense for the difference plots
        scaling_factor = 1  # Adjust this value to control intensity
        row_0_max *= scaling_factor
        row_0_min *= scaling_factor
        row_1_max *= scaling_factor
        row_1_min *= scaling_factor

        # Calculate CRPS per grouping: 
        icomp1_crps = np.mean(crps_network[indices1]) # high confidence
        icomp2_crps = np.mean(crps_network[indices2]) # low confidence

        plt.figure()
        plt.hist(crps_network[indices1], bins = 20, density = False, alpha = 0.5, label = f'High Confidence [{len(indices1)}]')
        plt.hist(crps_network[indices2], bins = 20, density = False, alpha = 0.5, label = f'Low Confidence [{len(indices1)}]')
        plt.xlabel('CRPS')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of CRPS for {keyword} Samples')
        plt.legend()
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_CRPS_distribution.png', format='png', bbox_inches ='tight', dpi = 300)
        plt.close()

        crps_dict = {0: icomp1_crps, 1: icomp2_crps} # high confidence indices, low confidence indices

        plt.figure()
        fig, ax = plt.subplots(2, 3, figsize=(19, 8),  subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

        for key, value in plots_dict.items(): # rows
            for j in range(3): # columns
                plot_data = value[j]
                cmap_spec = 'BrBG' if key == 0 else 'RdBu_r'
                # Set vmin and vmax based on the plot position
                # Set vmin and vmax based on the plot position
                if j == 0:  # First column
                    vmin = row_0_first_col_min if key == 0 else row_1_first_col_min
                    vmax = row_0_first_col_max if key == 0 else row_1_first_col_max
                else:  # Second and third columns
                    vmin = row_0_min if key == 0 else row_1_min
                    vmax = row_0_max if key == 0 else row_1_max

                cf1 = ax[key, j].pcolormesh(plot_data.lon, plot_data.lat, plot_data, 
                                            cmap=cmap_spec, transform=ccrs.PlateCarree(), vmin = vmin, vmax = vmax)
                ax[key, j].coastlines()
                ax[key, j].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
                ax[key, j].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
                ax[key, j].set_title(str(title_label) + f'{list(label_dict.keys())[key]} Anomalies \n {label_dict[list(label_dict.keys())[key]][j]} - {keyword_phase}')
                cbar1 = fig.colorbar(cf1, cmap=cmap_spec, ax=ax[key, j], orientation='vertical', fraction=0.02, pad=0.04, aspect=20, shrink=0.8)
                cbar1.set_label(str(title_label) + ' Anomalies')

                # Add mean CRPS as a label on the first two plots in each column
                if j < 2:  # Only add CRPS for the first two plots
                    crps_label = crps_dict[j]
                    ax[key, j].text(
                        0.02, 0.02, f'Mean CRPS: {crps_label:.3f}',
                        transform=ax[key, j].transAxes,
                        ha='left', va='bottom', fontsize=12, color='black'
                    )

        plt.subplots_adjust(wspace=0.05, hspace=0.1)  
        plt.tight_layout()
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_difference_maps.png', format='png', bbox_inches ='tight', dpi = 250)
        plt.close(fig)  # Close the figure to avoid memory issues

def maximum_difference(shash_parameters, required_samples = 50, tau_frozen = True):
    # Calculate the which samples are most different from one another based on shash parameters
    # Return the number of samples based on the required_samples input
    # shash_parameters is a 2D array of shash parameters for each sample
    # required_samples is the number of samples to return

    # Return the required_samples/3 samples with highest and lowest mu (shash_parameters[:, 0])
    mu_indices = np.argsort(shash_parameters[:, 0])
    mu_indices = np.concatenate((mu_indices[:required_samples//3], mu_indices[-required_samples//3:]))

    # Return the required_samples/3 samples with highest and lowest sigma (shash_parameters[:, 1])
    sigma_indices = np.argsort(shash_parameters[:, 1])
    sigma_indices = np.concatenate((sigma_indices[:required_samples//3], sigma_indices[-required_samples//3:]))

    # Return the required_samples/3 samples with highest and lowest gamma (shash_parameters[:, 2])
    gamma_indices = np.argsort(shash_parameters[:, 2])
    gamma_indices = np.concatenate((gamma_indices[:required_samples//3], gamma_indices[-required_samples//3:]))
            
    if not tau_frozen:
        # Return the required_samples/3 samples with highest and lowest tau (shash_parameters[:, 2])
        tau_indices = np.argsort(shash_parameters[:, 3])
        tau_indices = np.concatenate((tau_indices[:required_samples//3], tau_indices[-required_samples//3:]))

        all_indices = np.concatenate((mu_indices, sigma_indices, gamma_indices, tau_indices))
    else:
        all_indices = np.concatenate((mu_indices, sigma_indices, gamma_indices))
    
    return shash_parameters[all_indices]


def plotSHASH(shash_parameters, climatology, config, keyword = None): 
    """
    Input: Filename for climate data, SHASH parameters for sample
    Output: probability density distribution for given data and shash curves
    """

    # print(f"Sample Size of SHASH Parameters: {len(shash_parameters)}")
    imp.reload(shash.shash_torch)

    dist = Shash(shash_parameters)

    x_values = np.linspace(np.min(climatology) - 2, np.max(climatology), 1000)

    p = dist.prob(x_values).numpy()

    plt.figure(figsize=(8, 5), dpi=200)
    plt.hist(
        climatology, x_values, density=True, color="silver", alpha=0.75, label="climatology"
    )

    plt.plot(x_values, p, linewidth = 0.5) #label = samples
    plt.xlabel("precipitation anomaly (mm/day)")
    plt.ylabel("probability density")
    plt.title("Network Shash Prediction -\n" + str(config["expname"]) + f"{keyword}")
    plt.legend(loc = 'upper right')
    plt.ylim([0, 0.375])
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + '_' + str(keyword) + '_SHASH_w_climatology.png', format='png', bbox_inches ='tight', dpi = 200)
    plt.close()
    return p

def tiled_phase_analysis(mapinputs, target, ninoindices1, ninoindices2, ninoindices3, network_crps, config, keyword = None): 
    """
    Inputs: 
    - Realtime Mulitvariate MJO Indices (time series of RMM1, RMM2, ... RMMn)
    - Time Series Data (PRECT, TS)

    Outputs: 
    - Composite graphs of groupd phases for each variable (PRECT, TS..) based on EVALUATION DATE
    
    """
    # Load MJO Indices
    MJOfilename = '/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_historical_0201_1850-2014.pkl'

    # print(f"target time for mjo phases: {target.time}")

    # Identify time stamps corresponding to each MJO phase

    with open(MJOfilename, 'rb') as MJO_file:
        MJOda = np.load(MJO_file, allow_pickle=True)
        MJOda = np.asarray(MJOda)
        # print(MJOda)

    # Isolate MJO timestamps based on the -1, -2, and -3 columns of the dataarray
    start_year = int(MJOda[0, -3])
    start_month = int(MJOda[0, -2])
    start_day = int(MJOda[0, -1])
    start = datetime.datetime(start_year, start_month, start_day)
    end_year = int(MJOda[-1, -3])
    end_month = int(MJOda[-1, -2])
    end_day = int(MJOda[-1, -1])
    end = datetime.datetime(end_year, end_month, end_day)
    time_array = pd.date_range(start = start, end = end, freq = 'D')

    # Create phase number output array
    phases = np.zeros(len(MJOda))
    phaseqty = 9

    # Identify which phase of MJO each data point is in
    for samplecoord in range(0, len(MJOda[:, 2])):
        RMM1 = MJOda[samplecoord, 2]
        RMM2 = MJOda[samplecoord, 3]

        if not math.isnan(RMM1):
            dY = RMM2
            dX = RMM1

            angle_deg = np.rad2deg(np.arctan2(dY, dX))
            if angle_deg < 0:
                angle_deg = 360 - np.abs(angle_deg)

            amplitude = np.sqrt(RMM1**2 + RMM2**2)
            assert amplitude >= 0

            if amplitude <= 1:
                phases[samplecoord] = 0
            elif angle_deg >= 0 and angle_deg < 45:
                phases[samplecoord] = 5
            elif angle_deg >= 45 and angle_deg < 90:
                phases[samplecoord] = 6
            elif angle_deg >= 90 and angle_deg < 135:
                phases[samplecoord] = 7
            elif angle_deg >= 135 and angle_deg < 180:
                phases[samplecoord] = 8
            elif angle_deg >= 180 and angle_deg < 225:
                phases[samplecoord] = 1
            elif angle_deg >= 225 and angle_deg < 270:
                phases[samplecoord] = 2
            elif angle_deg >= 270 and angle_deg < 315:
                phases[samplecoord] = 3
            elif angle_deg >= 315 and angle_deg <= 360:
                phases[samplecoord] = 4
            else:
                print(f"angle: {angle_deg}, amplitude: {amplitude}")
                raise ValueError("Sample does not fit into a phase (?)")

    # Collect timestamps for each phase
    phase_timestamps = {}
    for phase in range(phaseqty):
        collected_phase_indices = np.where(phases == phase)[0]
        phase_timestamps[phase] = time_array[collected_phase_indices]  # Map indices to timestamps
    
    print(f"Phase Timestamps: {phase_timestamps}")

    # Convert the list to a NumPy array
    phases_concat = np.array(phases_concat, dtype=object)  # Use dtype=object for arrays of varying lengths
    print(f"phases concat showing all phases? {phases_concat}")
    
    # Align time stamps of target data with those of the group MJO phases to identify the corresponding indices
    # Isolate map inputs conditioned first on El Nino, La Nina, and Neutral: 
    target_time_coord = target.time
    EN_dates = target_time_coord.isel(time = ninoindices1)
    LN_dates = target_time_coord.isel(time = ninoindices2)
    N_dates = target_time_coord.isel(time = ninoindices3)

    # Convert CRPS scores to xarray.DataArray with time coordinates
    crps_with_target_time = xr.DataArray(
                data=network_crps,  
                coords={"time": target_time_coord}, 
                dims=["time"])

    # EN_crps_scores = crps_with_target_time.sel(time = EN_dates)
    # LN_crps_scores = crps_with_target_time.sel(time = LN_dates)
    # N_crps_scores = crps_with_target_time.sel(time = N_dates)

    enso_inputsdict = {
        'El Nino': (mapinputs.sel(time = EN_dates)),
        'La Nina': (mapinputs.sel(time = LN_dates)), 
        'Neutral': (mapinputs.sel(time = N_dates))
    }
    enso_to_index = {'El Nino': 0, 'La Nina': 1, 'Neutral': 2}

    # Apply the indices to the map inputs to create a composite map
    ## PRECT ##
    data_dict_PRECT = {}

    fig, ax = plt.subplots(9, len(enso_inputsdict), figsize = (22, 14), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    for enso_phase, (maps, crps_scores) in enso_inputsdict.items():
        for grouping, dates in enumerate(phases_concat):
            # Find overlapping dates between dates and conditioned maps: 
            # Convert numpy.datetime64 to cftime.DatetimeNoLeap
            dates_converted = []
            for date in dates:
                timestamp = pd.Timestamp(date)
                try:
                    # Attempt to create a valid cftime.DatetimeNoLeap object
                    dates_converted.append(cftime.DatetimeNoLeap(timestamp.year, timestamp.month, timestamp.day))
                except ValueError:
                    # Skip invalid dates (e.g., February 29 in non-leap years)
                    # print(f"Skipping invalid date: {timestamp}")
                    pass

            overlapping_dates = np.intersect1d(maps.time, dates_converted)

            # Select the corresponding map inputs for the current phase
            phase_mapinputs = maps[..., 0].sel(time=overlapping_dates)

            # if not np.array_equal(crps_scores.time.values, maps.time.values):
            #     print("CRPS scores and maps have different time coordinates.")
            # else:
            #     print("CRPS scores and maps have the same time coordinates.")
            
            # Filter CRPS scores for the overlapping dates
            crps_filtered = crps_scores.sel(time = overlapping_dates)
            mean_crps = crps_filtered.mean()

            # Calculate the mean of the selected map inputs
            mean_mapinputs = phase_mapinputs.mean(dim='time')
            std_mapinputs = phase_mapinputs.std(dim='time')

            norm_mapinputs = mean_mapinputs / std_mapinputs

            # store normalized map inputs in the dictionary: 
            data_dict_PRECT[(grouping, enso_phase)] = norm_mapinputs

            enso_column = enso_to_index[enso_phase]
            # Plot the mean map inputs
            cf = ax[grouping, enso_column].pcolormesh(norm_mapinputs.lon, norm_mapinputs.lat, norm_mapinputs, 
                                                      cmap='BrBG', transform=ccrs.PlateCarree(), vmin = -1, vmax = 1)
            ax[grouping, enso_column].set_title(f'{enso_phase} Mean Map Inputs \n MJO Phases {grouped_phases[grouping + 1]}')
            ax[grouping, enso_column].coastlines()
            ax[grouping, enso_column].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
            ax[grouping, enso_column].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree()) 

             # Add mean CRPS as a label on the subplot
            ax[grouping, enso_column].text(
                0.02, 0.02, f'Mean CRPS: {mean_crps:.3f}',
                transform=ax[grouping, enso_column].transAxes,
                ha='left', va='bottom', fontsize=12, color='black'
            )

            # Add colorbar
            cbar = fig.colorbar(cf, ax=ax[grouping, enso_column], orientation='vertical', fraction=0.02, pad=0.04, aspect=20, shrink=0.8)
            cbar.set_label('Normalized Precipitation \n Anomalies (sigma)')

            plt.tight_layout()

    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + 
                '_ENSO_Conditioned_MJO_composite_maps_PRECT.png', format='png', bbox_inches ='tight', dpi = 300)

    ## DIFFERENCE MAPS: 
    fig_diff, ax_diff = plt.subplots(1, 3, figsize=(22, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    for enso_phase in enso_inputsdict.keys():
        top_row = data_dict_PRECT[(0, enso_phase)]  # Top row (grouping=0)
        bottom_row = data_dict_PRECT[(1, enso_phase)]  # Bottom row (grouping=1)

        # Compute the difference
        diff = top_row - bottom_row

        # Plot the difference
        enso_column = enso_to_index[enso_phase]
        cf_diff = ax_diff[enso_column].pcolormesh(
            diff.lon, diff.lat, diff,
            cmap='BrBG', transform=ccrs.PlateCarree(), vmin=-0.45, vmax=0.45
        )
        ax_diff[enso_column].set_title(f'{enso_phase} Difference (Top - Bottom)')
        ax_diff[enso_column].coastlines()
        ax_diff[enso_column].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax_diff[enso_column].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        # Add colorbar
        cbar_diff = fig_diff.colorbar(cf_diff, ax=ax_diff[enso_column], orientation='vertical', fraction=0.02, pad=0.04, aspect=20, shrink=0.8)
        cbar_diff.set_label('Difference in Normalized Precipitation \n Anomalies (sigma)')

    plt.tight_layout()

    # Save the difference plot figure
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) +
                '_ENSO_Conditioned_MJO_difference_maps_PRECT.png', format='png', bbox_inches='tight', dpi=300)

    data_dict_TEMP = {}

    fig, ax = plt.subplots(2, 3, figsize = (22, 14), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    for enso_phase, (maps, crps_scores) in enso_inputsdict.items():
        for grouping, dates in enumerate(phases_concat):
            # Find overlapping dates between dates and conditioned maps: 
            # Convert numpy.datetime64 to cftime.DatetimeNoLeap
            dates_converted = []
            for date in dates:
                timestamp = pd.Timestamp(date)
                try:
                    # Attempt to create a valid cftime.DatetimeNoLeap object
                    dates_converted.append(cftime.DatetimeNoLeap(timestamp.year, timestamp.month, timestamp.day))
                except ValueError:
                    # Skip invalid dates (e.g., February 29 in non-leap years)
                    # print(f"Skipping invalid date: {timestamp}")
                    pass

            overlapping_dates = np.intersect1d(maps.time, dates_converted)

            # Select the corresponding map inputs for the current phase
            phase_mapinputs = maps[..., 1].sel(time=overlapping_dates)

            # Filter CRPS scores for the overlapping dates
            crps_filtered = crps_scores[np.isin(maps.time, overlapping_dates)]
            mean_crps = crps_filtered.mean()

            # Calculate the mean of the selected map inputs
            mean_mapinputs = phase_mapinputs.mean(dim='time')
            std_mapinputs = phase_mapinputs.std(dim='time')

            norm_mapinputs = mean_mapinputs / std_mapinputs

            # save normalized map inputs in the dictionary:
            data_dict_TEMP[(grouping, enso_phase)] = norm_mapinputs

            enso_column = enso_to_index[enso_phase]
            # Plot the mean map inputs
            cf = ax[grouping, enso_column].pcolormesh(norm_mapinputs.lon, norm_mapinputs.lat, norm_mapinputs, 
                                                      cmap='RdBu_r', transform=ccrs.PlateCarree(), vmin = -2.5, vmax = 2.5)
            ax[grouping, enso_column].set_title(f'{enso_phase} Mean Map Inputs \n MJO Phases {grouped_phases[grouping + 1]}')
            ax[grouping, enso_column].coastlines()
            ax[grouping, enso_column].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
            ax[grouping, enso_column].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree()) 

            # Add mean CRPS as a label on the subplot
            ax[grouping, enso_column].text(
                0.02, 0.02, f'Mean CRPS: {mean_crps:.3f}',
                transform=ax[grouping, enso_column].transAxes,
                ha='left', va='bottom', fontsize=12, color='black'
            )
            # Add colorbar
            cbar = fig.colorbar(cf, ax=ax[grouping, enso_column], orientation='vertical', fraction=0.02, pad=0.04, aspect=20, shrink=0.8)
            cbar.set_label('Normalized Temperature \n Anomalies (sigma)')
            
            plt.tight_layout()

    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) +
                 '_ENSO_Conditioned_MJO_composite_maps_SkinTemp.png', format='png', bbox_inches ='tight', dpi = 300)


     ## DIFFERENCE MAPS: 
    fig_diff, ax_diff = plt.subplots(1, 3, figsize=(22, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    for enso_phase in enso_inputsdict.keys():
        top_row = data_dict_TEMP[(0, enso_phase)]  # Top row (grouping=0)
        bottom_row = data_dict_TEMP[(1, enso_phase)]  # Bottom row (grouping=1)

        # Compute the difference
        diff = top_row - bottom_row

        # Plot the difference
        enso_column = enso_to_index[enso_phase]
        cf_diff = ax_diff[enso_column].pcolormesh(
            diff.lon, diff.lat, diff,
            cmap='RdBu_r', transform=ccrs.PlateCarree(), vmin=-0.45, vmax=0.45
        )
        ax_diff[enso_column].set_title(f'{enso_phase} Difference (Top - Bottom)')
        ax_diff[enso_column].coastlines()
        ax_diff[enso_column].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
        ax_diff[enso_column].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        # Add colorbar
        cbar_diff = fig_diff.colorbar(cf_diff, ax=ax_diff[enso_column], orientation='vertical', fraction=0.02, pad=0.04, aspect=20, shrink=0.8)
        cbar_diff.set_label('Difference in Normalized Skin Temperature \n Anomalies (sigma)')

    plt.tight_layout()

    # Save the difference plot figure
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) +
                '_ENSO_Conditioned_MJO_difference_maps_TEMP.png', format='png', bbox_inches='tight', dpi=300)


def precip_exceedance_threshold(target, output, precip_thresh, CRPS_network, CRPS_climatology, config, keyword = None):
    """
    Input: 
    - target: xarray DataArray containing the target precipitation data
    - precip_thresh: Precipitation threshold for exceedance
    - CRPS_network: CRPS scores for the network predictions
    - CRPS_climatology: CRPS scores for the climatology predictions
    - config: Configuration dictionary containing paths and other settings
    - keyword: Optional keyword for naming the output files

    """
    
    # Calculate the dates corresponding to target values over the 95th percentile for precip 
    itarget_percentile = np.where(target >= np.percentile(target, precip_thresh)) 
    exceeding_target_vals = target[itarget_percentile]

    # CRPS values for extreme precip events:
    exceed_networkCRPS = CRPS_network[itarget_percentile]
    exceed_climoCRPS = CRPS_climatology[itarget_percentile]
    print(f"Mean Network CRPS for Exceedance Events: {exceed_networkCRPS.mean().item()}")
    print(f"Mean Climatology CRPS for Exceedance Events: {exceed_climoCRPS.mean().item()}")

    # Proportion of samples whose network CRPS is > climatology CRPS for the exceedance events: 
    networkCRPS_successratio = np.sum(exceed_networkCRPS < exceed_climoCRPS) / len(exceed_networkCRPS)
    print(f"The proportion of network predictions for exceedance events that are better than climatology: {networkCRPS_successratio:.3f}")

    # Plot shash curves for these high precip events: 
    exceedance_params = output[itarget_percentile]

    plotSHASH(exceedance_params, target, config, keyword = keyword + f' {precip_thresh}% Precip Anoms Exceedance Threshold')

    # How confident (what is IQR) for these shash curves relative to all network predictions? 
    # Calculate the IQR for the exceedance parameters
    iqr_exceedance = iqr_basic(exceedance_params)
    print(f"IQR for exceedance parameters: {iqr_exceedance}")

    iqr_all = iqr_basic(output)

    plt.figure()
    plt.scatter(iqr_all, CRPS_network, s = 0.8, label = f'All Network Predictions [{len(CRPS_network)}]', color = 'grey')
    plt.scatter(iqr_exceedance, exceed_networkCRPS, s = 0.8, label = f'Exceedance Events [{len(exceed_networkCRPS)}]', color = 'red')
    plt.xlabel('IQR of SHASH Parameters')
    plt.ylabel('CRPS')
    plt.title(f'CRPS vs IQR of SHASH Parameters \n {precip_thresh}% Precip Exceedance Predictions')
    plt.legend()
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + 
                f'Exceedance {precip_thresh}_CRPS_vs_IQR_exceedance.png', format='png', bbox_inches ='tight', dpi = 300)
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + 
                f'Exceedance {precip_thresh}_CRPS_vs_IQR_exceedance.png', format='png', bbox_inches ='tight', dpi = 300)

    plt.figure()
    plt.scatter(target, CRPS_network, s = 0.8, label = f'All Network Predictions [{len(CRPS_network)}]', color = 'grey')
    plt.scatter(exceeding_target_vals, exceed_networkCRPS, s = 0.8, label = f'Exceedance Events [{len(exceed_networkCRPS)}]', color = 'blue')
    plt.xlabel('Anomaly Target Value (mm/day)')
    plt.ylabel('CRPS')
    plt.title(f'CRPS vs Target Value \n {precip_thresh}% Precip Exceedance Predictions')
    plt.legend()
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + 
                f'Exceedance {precip_thresh}_CRPS_vs_Target_exceedance.png', format='png', bbox_inches ='tight', dpi = 300)



def mjo_subsetindices(grouped_phases, mapinputs, target, ninodates1, ninodates2, ninodates3, network_crps, config, keyword = None): 
    """
    Inputs: 
    - Realtime Mulitvariate MJO Indices (time series of RMM1, RMM2, ... RMMn)
    - Time Series Data (PRECT, TS)

    Outputs: 
    - Composite graphs of groupd phases for each variable (PRECT, TS..) based on EVALUATION DATE
    
    """
    # Load MJO Indices
    MJOfilename = '/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_historical_0201_1850-2014.pkl'

    # print(f"target time for mjo phases: {target.time}")

    # Identify time stamps corresponding to each MJO phase

    with open(MJOfilename, 'rb') as MJO_file:
        MJOda = np.load(MJO_file, allow_pickle=True)
        MJOda = np.asarray(MJOda)
        # print(MJOda)

    # Isolate MJO timestamps based on the -1, -2, and -3 columns of the dataarray
    start_year = int(MJOda[0, -3])
    start_month = int(MJOda[0, -2])
    start_day = int(MJOda[0, -1])
    start = datetime.datetime(start_year, start_month, start_day)
    end_year = int(MJOda[-1, -3])
    end_month = int(MJOda[-1, -2])
    end_day = int(MJOda[-1, -1])
    end = datetime.datetime(end_year, end_month, end_day)
    time_array = pd.date_range(start = start, end = end, freq = 'D')

    # Create phase number output array
    phases = np.zeros(len(MJOda))
    phaseqty = 9

    # Identify which phase of MJO each data point is in
    for samplecoord in range(0, len(MJOda[:, 2])):
        RMM1 = MJOda[samplecoord, 2]
        RMM2 = MJOda[samplecoord, 3]

        if not math.isnan(RMM1):
            dY = RMM2
            dX = RMM1

            angle_deg = np.rad2deg(np.arctan2(dY, dX))
            if angle_deg < 0:
                angle_deg = 360 - np.abs(angle_deg)

            amplitude = np.sqrt(RMM1**2 + RMM2**2)
            assert amplitude >= 0

            if amplitude <= 1:
                phases[samplecoord] = 0
            elif angle_deg >= 0 and angle_deg < 45:
                phases[samplecoord] = 5
            elif angle_deg >= 45 and angle_deg < 90:
                phases[samplecoord] = 6
            elif angle_deg >= 90 and angle_deg < 135:
                phases[samplecoord] = 7
            elif angle_deg >= 135 and angle_deg < 180:
                phases[samplecoord] = 8
            elif angle_deg >= 180 and angle_deg < 225:
                phases[samplecoord] = 1
            elif angle_deg >= 225 and angle_deg < 270:
                phases[samplecoord] = 2
            elif angle_deg >= 270 and angle_deg < 315:
                phases[samplecoord] = 3
            elif angle_deg >= 315 and angle_deg <= 360:
                phases[samplecoord] = 4
            else:
                print(f"angle: {angle_deg}, amplitude: {amplitude}")
                raise ValueError("Sample does not fit into a phase (?)")

    # Collect timestamps for each phase
    phase_timestamps = {}
    for phase in range(phaseqty):
        collected_phase_indices = np.where(phases == phase)[0]
        phase_timestamps[phase] = time_array[collected_phase_indices]  # Map indices to timestamps
    
    # print(f"Phase Timestamps: {phase_timestamps}")

    phases_concat = []  # Use a list to dynamically store concatenated timestamps
    for key, value in grouped_phases.items():
        concatenated_timestamps = np.concatenate([phase_timestamps[phase] for phase in value])
        phases_concat.append(concatenated_timestamps)

    # Convert the list to a NumPy array
    phases_concat = np.array(phases_concat, dtype=object)  # Use dtype=object for arrays of varying lengths

    # Align time stamps of target data with those of the group MJO phases to identify the corresponding indices
    # Isolate map inputs conditioned first on El Nino, La Nina, and Neutral: 
    target_time_coord = target.time
    # Convert CRPS scores to xarray.DataArray with time coordinates
    crps_with_target_time = xr.DataArray(
                data=network_crps,  
                coords={"time": target_time_coord}, 
                dims=["time"])

    EN_crps_scores = crps_with_target_time.sel(time = ninodates1)
    LN_crps_scores = crps_with_target_time.sel(time = ninodates2)
    N_crps_scores = crps_with_target_time.sel(time = ninodates3)

    enso_inputsdict = {
        'El Nino': (mapinputs.sel(time = ninodates1), EN_crps_scores),
        'La Nina': (mapinputs.sel(time = ninodates2), LN_crps_scores), 
        'Neutral': (mapinputs.sel(time = ninodates3), N_crps_scores)
    }
    enso_to_index = {'El Nino': 0, 'La Nina': 1, 'Neutral': 2}

    # Apply the indices to the map inputs to create a composite map
    ## PRECT ##
    data_dict_PRECT = {}

    var_dict = {0: 'Precipitation', 1: 'SkinTemp'}
    color_dict = {0: 'BrBG', 1: 'RdBu_r'}

    for key, var in var_dict.items():
        for normalize in [False, True]:
            fig, ax = plt.subplots(2, 3, figsize = (22, 14), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            for enso_phase, (maps, crps_scores) in enso_inputsdict.items():
                for grouping, dates in enumerate(phases_concat):
                    # Find overlapping dates between dates and conditioned maps: 
                    # Convert numpy.datetime64 to cftime.DatetimeNoLeap
                    dates_converted = []
                    for date in dates:
                        timestamp = pd.Timestamp(date)
                        try:
                            # Attempt to create a valid cftime.DatetimeNoLeap object
                            dates_converted.append(cftime.DatetimeNoLeap(timestamp.year, timestamp.month, timestamp.day))
                        except ValueError:
                            # Skip invalid dates (e.g., February 29 in non-leap years)
                            # print(f"Skipping invalid date: {timestamp}")
                            pass

                    overlapping_dates = np.intersect1d(maps.time, dates_converted)

                    # Select the corresponding map inputs for the current phase
                    phase_mapinputs = maps[..., key].sel(time=overlapping_dates)

                    # Filter CRPS scores for the overlapping dates
                    crps_filtered = crps_scores.sel(time = overlapping_dates)
                    mean_crps = crps_filtered.mean()

                    if normalize:
                        # Normalize the map inputs
                        mean_mapinputs = phase_mapinputs.mean(dim='time')
                        std_mapinputs = phase_mapinputs.std(dim='time')

                        plot_data = mean_mapinputs / std_mapinputs
                        norm_label = "normalized"
                    else:
                        plot_data = phase_mapinputs.mean(dim='time')
                        norm_label = "anomalies"

                    # # Calculate the mean of the selected map inputs
                    # mean_mapinputs = phase_mapinputs.mean(dim='time')
                    # std_mapinputs = phase_mapinputs.std(dim='time')

                    # norm_mapinputs = mean_mapinputs / std_mapinputs

                    # store normalized map inputs in the dictionary: 
                    data_dict_PRECT[(grouping, enso_phase)] = plot_data

                    enso_column = enso_to_index[enso_phase]

                    vlim = max(np.abs(plot_data.min()), np.abs(plot_data.max()))

                    # Plot the mean map inputs
                    cf = ax[grouping, enso_column].pcolormesh(plot_data.lon, plot_data.lat, plot_data, 
                                                            cmap=color_dict[key], transform=ccrs.PlateCarree(), vmin = -vlim, vmax = vlim)
                    ax[grouping, enso_column].set_title(f'{enso_phase} Mean Map Inputs \n MJO Phases {grouped_phases[grouping + 1]}')
                    ax[grouping, enso_column].coastlines()
                    ax[grouping, enso_column].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
                    ax[grouping, enso_column].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree()) 

                    # Add mean CRPS as a label on the subplot
                    ax[grouping, enso_column].text(
                        0.02, 0.02, f'Mean CRPS: {mean_crps:.3f}',
                        transform=ax[grouping, enso_column].transAxes,
                        ha='left', va='bottom', fontsize=12, color='black'
                    )

                    # Add colorbar
                    cbar = fig.colorbar(cf, ax=ax[grouping, enso_column], orientation='vertical', fraction=0.02, pad=0.04, aspect=20, shrink=0.8)
                    cbar.set_label(f'Normalized {var} \n Anomalies (sigma)')

                    plt.tight_layout()

            plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + 
                '_ENSO_Conditioned_MJO_composite_maps_' + str(var) + str(norm_label) +'.png', format='png', bbox_inches ='tight', dpi = 300)
            plt.close(fig)
