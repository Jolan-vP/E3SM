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


def IQRdiscard_plot(networkoutput, target, crps_scores, crps_climatology_scores, config, target_type = 'anomalous', analyze_months = True, most_confident = True):
    if most_confident == True:
        keyword = 'narrow'
    else: 
        keyword = 'wide'

    # iqr capture relies on SHASH output parameters (mu, sigma, tau, gamma) and the SHASH class
    iqr = iqr_basic(networkoutput)
    percentiles = np.linspace(100, 0, 21)

    avg_crps = []
    avg_target = []
    months = np.arange(1, 13)
    sample_index = np.zeros((len(target), len(percentiles)))
    month_per_percentilebin = np.zeros((len(target), len(percentiles)))
    dry_month_percentilebin = np.zeros_like(month_per_percentilebin)
    dry_month_indices = np.zeros_like(month_per_percentilebin)
    dry_months = [4, 5, 6, 7, 8, 9]

    for ip, p in enumerate(percentiles):
        if most_confident == True:
            avg_crps.append(np.mean(crps_scores[iqr < np.percentile(iqr, p)]))
            avg_target.append(np.mean(target[iqr < np.percentile(iqr, p)]))
            # capture the index (out of total) for all the samples in each bin
            indices = np.where(iqr < np.percentile(iqr, p))[0]
        elif most_confident == False:
            avg_crps.append(np.mean(crps_scores[iqr >= np.percentile(iqr, p)]))
            avg_target.append(np.mean(target[iqr >= np.percentile(iqr, p)]))
            # capture the index (out of total) for all the samples in each bin
            indices = np.where(iqr > np.percentile(iqr, p))[0]

        sample_index[:len(indices), ip] = indices

        if analyze_months == True: 
            # identify the month of the year for each sample based on sample index and target date.time
            month_per_percentilebin[:len(indices), ip] = target.time.dt.month[indices]
            # keep only the values that are in dry months 
            mask = np.isin(month_per_percentilebin[:len(indices), ip], dry_months)
            dry_month_percentilebin[:len(indices), ip][mask] = month_per_percentilebin[:len(indices), ip][mask]
            # Save the indices of the dry months
            dry_month_indices[:len(indices), ip][mask] = indices[mask]

    color = 'tab:blue'
    fig, ax1 = plt.subplots()
    plt.gca().invert_xaxis()
    ax1.set_ylabel('Average CRPS')
    ax1.set_xlabel('IQR Percentile (% Data Remaining)', color=color)
    ax1.plot(percentiles, avg_crps, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=crps_climatology_scores.mean(), color='grey', linestyle='--', label='CRPS Mean Climatology')
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
            plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/CRPS_narrowIQR_DiscardPlot_anomalies.png', format='png', bbox_inches ='tight', dpi = 300)
        else: 
            plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/CRPS_wideIQR_DiscardPlot_anomalies.png', format='png', bbox_inches ='tight', dpi = 300)

    elif target_type == 'raw':
        color = 'tab:olive'
        ax2.set_ylabel('Raw Target Values (mm/day)', color=color)
        ax2.plot(percentiles, avg_target, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        if most_confident == True:
            plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/CRPS_narrowIQR_DiscardPlot_raw.png', format='png', bbox_inches ='tight', dpi = 300)
        else: 
            plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/CRPS_wideIQR_DiscardPlot_true_precip.png', format='png', bbox_inches ='tight', dpi = 300)
    
    if analyze_months == True:
        save_pickle(dry_month_percentilebin, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/DRYmonth_per_percentilebin_' + str(keyword) + '.pkl')
        save_pickle(dry_month_indices, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/DRYmonth_indices_' + str(keyword) + '.pkl')
        # Filter and plot each column separately

        # Establish Month of Year Axis 
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.2))
        ax3.set(ylim=(0.1, 12.9))
        ax3.set_ylabel('Month of the Year', color='black')
        colors = ['#5ca1e1','#f49b62', '#f49b62', '#5ca1e1']
        cmap = mpl.colors.ListedColormap(colors)

        for ip in [99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]:
            last_index = np.where(dry_month_percentilebin[:, ip] == 0)[0][0]
            tiled_percentiles = np.tile(percentiles, (len(dry_month_percentilebin), 1))
            # ax3.scatter(tiled_percentiles[:last_index, ip], month_per_percentilebin[:last_index, ip], c=month_per_percentilebin[:last_index, ip], cmap=cmap, s = 0.4)
            ax3.scatter(tiled_percentiles[:, ip], dry_month_percentilebin[:, ip], c='grey', s = 0.4)
    
        ax1.legend(loc = 'upper right')

        return sample_index, dry_month_indices

    else: 
        
        return sample_index


def target_discardplot(targetCNN, targetSNN, CNNcrps_scores, NNcrps_scores, crps_climatology_scores, config, target_type = 'anomalous'):
    
    percentiles = np.linspace(100, 0, 21)

    avg_crpsCNN = []
    avg_crpsNN = []
    avg_crpsClimo = []
    avg_target = []
    sample_indexCNN = np.zeros((len(targetCNN), len(percentiles)))
    sample_indexSNN = np.zeros((len(targetSNN), len(percentiles)))

    for ip, p in enumerate(percentiles):
        # for each percentile, calculate average CRPS score in the bin, binned by corresponding target value
        avg_crpsCNN.append(np.mean(CNNcrps_scores[np.abs(targetCNN) < np.percentile(np.abs(targetCNN), p)]))
        avg_crpsNN.append(np.mean(NNcrps_scores[np.abs(targetSNN) < np.percentile(np.abs(targetSNN), p)]))
        if config["arch"]["type"] == "cnn":
            avg_crpsClimo.append(np.mean(crps_climatology_scores[np.abs(targetCNN) < np.percentile(np.abs(targetCNN), p)]))
            avg_target.append(np.mean(np.abs(targetCNN[np.abs(targetCNN) < np.percentile(np.abs(targetCNN), p)])))
            indices = np.where(targetCNN < np.percentile(np.abs(targetCNN), p))[0]
            sample_indexCNN[:len(indices), ip] = indices
        else: 
            avg_crpsClimo.append(np.mean(crps_climatology_scores[np.abs(targetSNN) < np.percentile(np.abs(targetSNN), p)]))
            avg_target.append(np.mean(np.abs(targetSNN[np.abs(targetSNN) < np.percentile(np.abs(targetSNN), p)])))
            # capture the index (out of total) for all the samples in each bin
            indices = np.where(targetSNN < np.percentile(np.abs(targetSNN), p))[0]
            sample_indexSNN[:len(indices), ip] = indices

    # save avg_crpsCNN, avg_crpsNN, avg_crpsClimo, avg_target in one pickle file
    # save_pickle(avg_crpsCNN, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/avg_crpsCNN_percentiles.pkl')
    # save_pickle(avg_crpsNN, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/avg_crpsNN_percentiles.pkl')
    # save_pickle(avg_crpsClimo, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/avg_crpsClimo_percentiles.pkl')
    # save_pickle(avg_target, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/avg_target_percentiles.pkl')

    colors = ['#fca50a', '#bc3754', '#6a176e', '#420a68']
    fig, ax1 = plt.subplots()
    ax1.set_ylabel('Average CRPS')
    ax1.set_xlabel('Percentile of Target Magnitude \n (% Data Remaining)')
    ax1.plot(percentiles, avg_crpsCNN, color=colors[0], linewidth = 1.3, label = 'CNN')
    ax1.plot(percentiles, avg_crpsNN,  color=colors[1], linewidth = 1.3, label = 'Simple NN') 
    ax1.plot(percentiles, avg_crpsClimo, color=colors[2], linewidth = 1.3, label = 'Climatology')
    ax1.set_title(f'Discard Plot \n CRPS vs. Decreasing Target Magnitude ({config["expname"]})')

    ax1.tick_params(axis='y')
    ax1.axhline(y=crps_climatology_scores.mean(), color='grey', linestyle='--', label='CRPS Mean Climatology')
    min = np.nanmin(avg_crpsCNN) - .08
    max = np.nanmax(avg_crpsClimo) + .1
    # ax1.set_ylim([min, 3])
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    if target_type == 'anomalous':
        color = 'tab:olive'
        ax2.set_ylabel('Magnitude of Average Target Anomalies \n (mm/day)', color=colors[3])
        ax2.plot(percentiles, avg_target, color=colors[3], linestyle = 'dotted', label = 'Target Anomalies')
        ax2.tick_params(axis='y', labelcolor=colors[3])
    
    elif target_type == 'raw':
        color = 'tab:olive'
        ax2.set_ylabel('Raw Target Values (mm/day)', color=colors[3])
        ax2.plot(percentiles, avg_target, color=colors[3], linestyle = 'dotted', label = 'True Target Magnitudes')
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
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/CRPS_TargMag_DiscardPlot_anomalies_' + str(config["expname"]) + '.png', format='png', bbox_inches ='tight', dpi = 300)
    else: 
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/CRPS_TargMag_DiscardPlot_raw_' + str(config["expname"]) + '.png', format='png', bbox_inches ='tight', dpi = 300)


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
    print(type(subset_indices))
    print(subset_indices.shape)
    
    # Calculate relative ratio of ENSO phases relative to all samples
    elnino_ratio = elnino.shape[1] / crps_scores.shape[0]
    lanina_ratio = lanina.shape[1] / crps_scores.shape[0]
    neutral_ratio = neutral.shape[1] / crps_scores.shape[0]

    # identify enso phases of each sample in the subset
    sub_elnino = elnino[np.isin(elnino, subset_indices)]
    sub_lanina = lanina[np.isin(lanina, subset_indices)]
    sub_neutral = neutral[np.isin(neutral, subset_indices)]

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

    return sub_elnino, sub_lanina, sub_neutral

def compositemapping(indices, mapinputs, config, keyword = None): 
    """
    Take in two sets of indices, and create two composite maps based the two indices sets
    mapinputs is the input map data for all samples 
    """

    # adjust indices so that every value is config["databuilder"]["lagtime"] values EARLIER. 
    # This is because the indices are based on the target values, which are lagtime values ahead of the input values
    # We want maps that represent the state of the world at the time the prediction is made ("Prediction Day") rather than Verification Day
    indices = indices - config["databuilder"]["lagtime"]
    print(f"Number of indices: {len(indices)}")

    icomposite_prect = mapinputs[..., 0].isel(time=indices).mean(dim='time')  
    icomposite_ts = mapinputs[..., 1].isel(time=indices).mean(dim='time')  

    lats = np.linspace(-89.5, 89.5, 180) 
    lons = mapinputs.lon

    # print(f"Composite Data: {icomposite_prect}")
    # print(f"composite mean: {np.mean(icomposite_prect).item()}")
    # print(f"composite min: {np.min(icomposite_ts).item()}")
    # print(f"composite max: {np.max(icomposite_ts).item()}")

    plt.figure()
    fig, ax = plt.subplots(2, 1, figsize=(16, 9),  subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})

    cf1 = ax[0].pcolormesh(lons, lats, icomposite_prect, cmap='BrBG', transform=ccrs.PlateCarree(), vmin = -1, vmax = 1 )
    ax[0].set_title(f'Precipitation Composite Map \n {keyword} Predictions')
    ax[0].coastlines()
    ax[0].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax[0].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

    cf2 = ax[1].pcolormesh(lons, lats, icomposite_ts, cmap='RdBu_r', transform=ccrs.PlateCarree(), vmin = -1, vmax = 1 )
    ax[1].set_title(f'Temperature Composite Map \n {keyword} Predictions')
    ax[1].coastlines()
    ax[1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax[1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

    # add colorbar that is same for both plots
    cbar1 = fig.colorbar(cf1, cmap='BrBG', ax=ax[0], orientation='vertical', fraction=0.01, pad=0.03)
    cbar1.set_label('Precipitation Anomalies \n (mm/day)')

    cbar2 = fig.colorbar(cf2, cmap='BrBG', ax=ax[1], orientation='vertical', fraction=0.01, pad=0.03)
    cbar2.set_label('Temperature Anomalies \n (deg C)')

    # fig.tight_layout()
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_composite_maps.png', format='png', bbox_inches ='tight', dpi = 300)



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
    imp.reload(shash.shash_torch)

    dist = Shash(shash_parameters)

    x_values = np.linspace(np.min(climatology) - 2, np.max(climatology), 1000)

    p = dist.prob(x_values).numpy()

    plt.figure(figsize=(8, 5), dpi=200)
    plt.hist(
        climatology, x_values, density=True, color="silver", alpha=0.75, label="climatology"
    )

    plt.plot(x_values, p, linewidth = 0.5 ) #label = samples
    plt.xlabel("precipitation anomaly (mm/day)")
    plt.ylabel("probability density")
    plt.title("Network Shash Prediction -" + str(config["expname"]))
    plt.legend()
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + '_' + str(keyword) + '_SHASH_w_climatology.png', format='png', bbox_inches ='tight', dpi = 300)

    return p