"""
Functions designed to take outputs from variety of experiments and meaningfully compare them. 

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
import utils
import utils.filemethods as filemethods
from utils.filemethods import open_data_file
from utils.filemethods import filter_dates
import matplotlib.cm as cm
from analysis.analysis_metrics import load_pickle
from matplotlib.colors import LinearSegmentedColormap
from XAI.captum import average_attributions, visualize_average_attributions
import torch
from model.build_model import TorchModel

import math
import datetime


def combined_success_discard(experiments, keyword = None):
    """
    Discard plot of success ratio vs IQR percentile for variety of experiments

    experiments = {
        "OBS(OBS)": [obs_exp1, obs_exp2, obs_exp3, ...], 
        "E3SM(OBS)": [e3sm_obs_exp1, e3sm_obs_exp2, e3sm_obs_exp3, ...],
        "E3SM(E3SM)": [e3sm_exp1, e3sm_exp2, e3sm_exp3, ...],
        "E3SM-short(OBS)": [e3sm_short_obs_exp1, e3sm_short_obs_exp2, e3sm_short_obs_exp3, ...],
        "E3SM-short(E3SM)": [e3sm_short_exp1, e3sm_short_exp2, e3sm_short_exp3, ...]
    }
    """

    # exps = {}

    # if "OBS(OBS)" in experiments: 
    #     obs_exp = experiments["OBS(OBS)"]
    #     exps["OBS(OBS)"] = obs_exp
    # if "E3SM(OBS)" in experiments: 
    #     e3sm_obs_exp = experiments["E3SM(OBS)"]
    #     exps["E3SM(OBS)"] = e3sm_obs_exp
    # if "E3SM-short(OBS)" in experiments:
    #     e3sm_short_obs_exp = experiments["E3SM-short(OBS)"]
    #     exps["E3SM-short(OBS)"] = e3sm_short_obs_exp
    # if "E3SM(E3SM)" in experiments: 
    #     e3sm_exp = experiments["E3SM(E3SM)"]
    #     exps["E3SM(E3SM)"] = e3sm_exp
    # elif "E3SM-short(E3SM)" in experiments:
    #     e3sm_short_exp = experiments["E3SM-short(E3SM)"]
    #     exps["E3SM-short(E3SM)"] = e3sm_short_exp

    exps = experiments

    plt.figure(figsize=(7, 5))
    plt.gca().invert_xaxis()  # high confidence = low IQR = right side of plot

    color_themes = {
        0: "#3b528b", 
        1: "#21918c", 
        2: "#5ec962",
        3: "#fde725",
        }
    # color_themes = {
    #     0:  "#9335D1", 
    #     1:  "#3451D4",
    #     2:  "#2FB4C9",
    #     3:  '#63BA31',
    #     4:  '#D4932A',
    #     5:  '#D03F6D'
    # }

    i = 0
    for experiment_type, exp_names in exps.items():

        for iexp, exp in enumerate(exp_names):

            filename = str("/pscratch/sd/p/plutzner/E3SM/saved/output/" + str(exp) + "/" + str(exp) + "_success_ratio.pkl")

            try:
                with open(filename, 'rb') as f:
                    discard_data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError, UnicodeDecodeError):
                try:
                    # If it fails, try gzip
                    with gzip.open(filename, 'rb') as f:
                        discard_data = pickle.load(f)
                except Exception as e:
                    raise RuntimeError(f"Failed to load file with both normal and gzip methods: {e}")
            
            percentiles = discard_data['percentiles']
            avg_success_ratio = discard_data['avg_success_ratio']

            if iexp == 0: 
                plt.plot(percentiles, avg_success_ratio, color=color_themes[i], alpha = 0.35, label = f"{experiment_type}", linewidth = 2.5)
            else: 
                plt.plot(percentiles, avg_success_ratio, color=color_themes[i], alpha = 0.35, linewidth = 2.5)
            # plt.fill_between(x = [0, 100], y1 = [0.5, 0.5], color = 'grey', alpha=0.03, edgecolor = None)


            plt.xlabel('IQR Percentile (% Data Remaining)')
            plt.ylabel('Proportion of Samples with Lower Network CRPS')
            plt.ylim(0.5, 0.85)
            plt.xlim(101, 4)
            plt.title('Increasing Confidence Success Ratio Discard Plot')
            plt.tight_layout()
        i += 1
            

    plt.legend()

    plt.savefig('/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/combined_SuccessRatio_DiscardPlot_' + str(keyword) + '_Z500.png', format = 'png',  dpi = 250) 






def combined_CRPS_IQR_discard(experiments, keyword = None):
    """
    Discard plot of mean binned CRPS by IQR percentile for all experiment types passed in. 
    
    """

    plt.figure(figsize=(7, 5))
    plt.gca().invert_xaxis()  # high confidence = low IQR = right side of plot

    exps = experiments 

    color_themes = {
        0: "#0d0887", 
        1: "#7e03a8", 
        2: "#cc4778",
        3: "#f89540"
    }

    i = 0
    for experiment_type, exp_names in exps.items():
        
        for iexp, exp in enumerate(exp_names):
            filename = str("/pscratch/sd/p/plutzner/E3SM/saved/output/" + str(exp) + "/" + str(exp) + "_IQR_CRPS_discard.pkl")
            climatology_crps = open_data_file('/pscratch/sd/p/plutzner/E3SM/saved/output/' + str(exp) + '/' + str(exp) + '_CRPS_climatology_values.pkl')
            mean_climo_crps = np.mean(climatology_crps)

            try:
                with open(filename, 'rb') as f:
                    discard_data = pickle.load(f)
            except (pickle.UnpicklingError, EOFError, UnicodeDecodeError):
                try:
                    # If it fails, try gzip
                    with gzip.open(filename, 'rb') as f:
                        discard_data = pickle.load(f)
                except Exception as e:
                    raise RuntimeError(f"Failed to load file with both normal and gzip methods: {e}")

            percentile_dict = discard_data['percentiles']
            crps_dict = discard_data['avg_crps']            

            if iexp == 0:
                if experiment_type == "OBS(OBS)":
                    plt.axhline(y=mean_climo_crps, color='grey', linestyle='--', label = f'OBS Baseline Mean CRPS', linewidth = 2)
                if experiment_type == "E3SM-short(E3SM)":
                    plt.axhline(y=mean_climo_crps, color='grey', linestyle='--', label = f'E3SM Baseline Mean CRPS', linewidth = 2)
                plt.plot(percentile_dict, crps_dict, label=f'{experiment_type}', alpha = 0.35, linewidth = 2.5, color = color_themes[i])
            else:
                plt.plot(percentile_dict, crps_dict, alpha = 0.35, linewidth = 2.5, color = color_themes[i])

        plt.xlabel('IQR Percentile (% Data Remaining)')
        plt.ylabel('Average CRPS')
        plt.ylim(0.32, 0.68)
        plt.xlim(101, 4)
        plt.title('Increasing Confidence CRPS Discard Plot')
        plt.tight_layout()
        plt.legend()
        i += 1

    plt.savefig('/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/CRPS_discard_plot_combined__' + str(keyword) + '_Z500.png', format = 'png',  dpi = 250)



def IQR_distributions(experiments, keyword = None):
    """
    Plot the distribution of IQR values for a variety of experiments on the same plot. 
    """
    plt.figure(figsize=(7, 5))

    exps = experiments 

    color_themes = {
        0: "#0d0887", 
        1: "#7e03a8", 
        2: "#cc4778",
        3: "#f89540"
    }

    i = 0
    for experiment_type, exp_names in exps.items():
        
        for iexp, exp in enumerate(exp_names):
            # Load the output and target data for all experiments
            output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp}/{exp}_network_SHASH_parameters.pkl')

            # Load climatologies: 
            climatology = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp153_ERA5_processed_Z500_climatology_1981-2010.nc')

            # Load climatology statistics: 
            climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')

            # # Load testing target data: 
            target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')

            max_value = ((np.max(climatology['y'].values) - climatology_stats['z'][2]) / climatology_stats['z'][3])
            min_value = ((np.min(climatology['y'].values) - climatology_stats['z'][2]) / climatology_stats['z'][3])
            x_values = np.linspace(min_value, max_value, 100)

            # Calculate IQR for each sample in both experiments
            iqr = iqr_basic(output)

            bins = 65
            min_value = min(np.min(iqr), np.min(iqr))
            max_value = max(np.max(iqr), np.max(iqr))
            bin_edges = np.linspace(min_value, max_value, bins)

            # histograms of IQR for each phase
            if iexp == 0:
                plt.hist(iqr, bins=bin_edges, alpha=0.4, label=f'{experiment_type}', color = color_themes[i], density = True, histtype = 'step')
            else:
                plt.hist(iqr, bins=bin_edges, alpha=0.4, color = color_themes[i], density = True, histtype = 'step')

        i += 1
            
        plt.xlabel('IQR')
        plt.ylabel('Density')
        plt.title('IQR Distribution Across Model Types')
        plt.legend()

    plt.savefig('/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/IQR_distribution_combined__' + str(keyword) + '_Z500.png', format = 'png',  dpi = 250)








def composite_inputmap_target(experiments, confidence_level = 50, keyword = None):
    """
    For given confidence threshold, plot a subplot with all input variables, and a distribution of the true target values for those samples. 
    """
    exps = experiments

    # convert confidence level
    confidence_threshold = confidence_level / 100.0

    colormaps = ["BrBG", "RdBu_r", "PuOr_r"]
    vars = ["Total Precip", "Skin Temp", "Z500"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude= 180)}, )
    axs_flat = axs.flatten()

    for experiment_type, exp_name in exps.items():
        
        # for iexp, exp in enumerate(exp_names):
        # Load the output and target data for all experiments
        output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')

        # input maps: 
        if experiment_type == "E3SM-short(OBS)" or experiment_type == "E3SM(OBS)" or experiment_type == "E3SM-long(OBS)":
            config = utils.get_config(exp_name)
            input_data = config["input_data"]

            input_maps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/{input_data}_trimmed_test_dat.nc')
            input_maps = input_maps['x']

        else:
            input_maps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/{exp_name}_trimmed_test_dat.nc')
            input_maps = input_maps['x']

        # Load climatologies: 
        climatology = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp153_ERA5_processed_Z500_climatology_1981-2010.nc')

        # Load climatology statistics: 
        climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')

        # # Load testing target data: 
        target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
        target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]

        max_value = ((np.max(climatology['y'].values) - climatology_stats['z'][2]) / climatology_stats['z'][3])
        min_value = ((np.min(climatology['y'].values) - climatology_stats['z'][2]) / climatology_stats['z'][3])
        x_values = np.linspace(min_value, max_value, 100)

        # Calculate IQR for each sample in both experiments
        iqr = iqr_basic(output)

        # List IQR by percentile
        percentiles = np.percentile(iqr, np.arange(0, 101, 1))
        print(percentiles[int(confidence_threshold * 100)])

        # Select samples whose IQR is smaller than the given confidence threshold (ie. "20% lowest IQR values")
        selected_indices = np.where(iqr <= percentiles[int(confidence_threshold * 100)])[0]

        print(f"Experiment: {experiment_type}, Number of samples selected at {confidence_level}% confidence: {len(selected_indices)}")
        if len(selected_indices) < 30:
            print(f"Less than 30 samples selected for {experiment_type} at {confidence_level}% confidence")

        # Extract corresponding input maps and target values
        selected_targets = target.values[selected_indices]
        
        # Plotting
        mean_target = np.mean(selected_targets)

            # Plot histogram with proper settings for density
        hist_values, bin_edges, patches = axs_flat[-1].hist(selected_targets, bins=30, alpha=0.7, color='gray', density=True)
        axs_flat[-1].set_title(f'{experiment_type} Target Distribution (n={len(selected_targets)})')
        axs_flat[-1].axvline(mean_target, color='purple', linestyle='--', linewidth=1.8, label=f'Mean = {mean_target:.2f}')
        axs_flat[-1].legend()
        axs_flat[-1].set_xlabel('Standardized Z500')
        axs_flat[-1].set_ylabel('Density')  # Changed from 'Count' to 'Density'
        
        # Set proper aspect ratio and limits for histogram
        axs_flat[-1].set_xlim(-5, 5)
        axs_flat[-1].set_xticks(np.arange(-5, 6, 1))

        # Fix: Explicitly set y-ticks for density plot
        max_density = np.max(hist_values)
        if max_density > 0:
            # Create 5-6 evenly spaced y-ticks from 0 to max density
            y_tick_step = max_density / 5
            y_ticks = np.arange(0, max_density + y_tick_step, y_tick_step)
            axs_flat[-1].set_yticks(y_ticks)
            # Format y-tick labels to show appropriate precision
            axs_flat[-1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))

        # For density plots, let matplotlib handle the y-ticks automatically
        # or set them based on the density values if needed
        axs_flat[-1].tick_params(axis='x', which='major', bottom=True, top=False, labelbottom=True)
        axs_flat[-1].tick_params(axis='y', which='major', left=True, right=False, labelleft=True)
        
        # Make histogram more rectangular
        axs_flat[-1].set_aspect('auto')

        for i in range(len(colormaps)):
            cmap = colormaps[i]
            
            mean_map = np.mean(input_maps[... , i].values[selected_indices, ...], axis=0)

            # set vmin and vmax to be symmetric around zero for diverging colormaps
            vmin = np.min(mean_map)
            vmax = np.max(mean_map)
            abs_max = max(abs(vmin), abs(vmax))
            vmin = -abs_max
            vmax = abs_max

            # Create the map plot
            axs_flat[i].coastlines()
            axs_flat[i].add_feature(cfeature.BORDERS, linestyle=':')
            im = axs_flat[i].pcolormesh(input_maps['lon'], input_maps['lat'], mean_map, transform=ccrs.PlateCarree(central_longitude= 0),cmap=cmap, vmin=vmin, vmax=vmax)
            axs_flat[i].set_title(f'{experiment_type} Composite {vars[i]} Map')
            cbar = fig.colorbar(im, ax=axs_flat[i], orientation='vertical', shrink=0.6, fraction=0.02, pad=0.02)

            # Set global extent to show full map
            axs_flat[i].set_global()
            
            # Set consistent geographic ticks for all map plots
            axs_flat[i].set_xticks([-180, -120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
            axs_flat[i].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree(central_longitude = 180))


        plt.suptitle(f'Composite Input Maps and Target Distribution at {confidence_level}% Confidence Level \n ({exp_name}) ', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/composite_inputmaps_target_{confidence_level}percent_confidence_'+ str(exp_name) + '_Z500.png', format='png', dpi=250, bbox_inches='tight')
        plt.close()


    
# def COMPARE_composite_inputmap_target(experiments, confidence_level = 50, keyword = None):
#     """
#     For given confidence threshold, plot difference maps between two experiments and compare their target distributions.
#     """
#     exps = experiments
    
#     # Ensure we have exactly 2 experiments
#     if len(exps) != 2:
#         raise ValueError("This function requires exactly 2 experiments for comparison")
    
#     exp_names = list(exps.keys())
#     exp_codes = list(exps.values())

#     # convert confidence level
#     confidence_threshold = confidence_level / 100.0

#     colormaps = ["BrBG", "RdBu_r", "PuOr_r"]
#     vars = ["Total Precip", "Skin Temp", "Z500"]
#     units = ['(mm/day)', '(K)', '(m)']

#     # Create figure with 1x3 layout: 3 difference maps
#     fig, axs = plt.subplots(1, 3, figsize=(18, 5), 
#                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    
#     # Store composite data for both experiments
#     composite_data = {}
    
#     for idx, (experiment_type, exp_name) in enumerate(exps.items()):
#         print(f'experiment type: {experiment_type}, exp name: {exp_name}')
        
#         # Load the output and target data for all experiments
#         output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')

#         # input maps: 
#         if experiment_type == "E3SM-short(OBS)" or experiment_type == "E3SM(OBS)" or experiment_type == "E3SM-long(OBS)":
#             config = utils.get_config(exp_name)
#             input_data = config["input_data"]
#             input_maps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/{input_data}_trimmed_test_dat.nc')
#             input_maps = input_maps['x']
#         else:
#             input_maps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
#             input_maps = input_maps['x']

#         # Load climatology statistics: 
#         climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')

#         # Load testing target data: 
#         target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
#         target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]

#         # Calculate IQR for each sample
#         iqr = iqr_basic(output)

#         # List IQR by percentile
#         percentiles = np.percentile(iqr, np.arange(0, 101, 1))

#         # Select samples whose IQR is smaller than the given confidence threshold
#         selected_indices = np.where(iqr <= percentiles[int(confidence_threshold * 100)])[0]

#         print(f"Experiment: {experiment_type}, Number of samples selected at {confidence_level}% confidence: {len(selected_indices)}")
#         if len(selected_indices) < 30:
#             print(f"Less than 30 samples selected for {experiment_type} at {confidence_level}% confidence")

#         # Extract corresponding target values (removed since we're not using histograms)
#         # selected_targets = target.values[selected_indices]
#         # selected_targets_all[experiment_type] = selected_targets
        
#         # Calculate composite maps for each variable
#         composite_maps = []
#         for i in range(len(colormaps)):
#             mean_map = np.mean(input_maps[... , i].values[selected_indices, ...], axis=0)
#             composite_maps.append(mean_map)
        
#         composite_data[experiment_type] = {
#             'maps': composite_maps,
#             'lon': input_maps['lon'],
#             'lat': input_maps['lat'],
#             'n_samples': len(selected_indices)
#         }
    
#     # Calculate difference maps (second experiment minus first experiment)
#     exp1_name, exp2_name = exp_names[0], exp_names[1]
    
#     for i in range(len(colormaps)):
#         cmap = colormaps[i]
        
#         # Calculate difference map
#         diff_map = composite_data[exp2_name]['maps'][i] - composite_data[exp1_name]['maps'][i]
        
#         # Set symmetric colorbar limits
#         abs_max = np.max(np.abs(diff_map))
#         vmin, vmax = -abs_max, abs_max
        
#         # Create the difference map plot
#         axs[i].coastlines()
#         axs[i].add_feature(cfeature.BORDERS, linestyle=':')
#         im = axs[i].pcolormesh(composite_data[exp1_name]['lon'], 
#                               composite_data[exp1_name]['lat'], 
#                               diff_map, 
#                               transform=ccrs.PlateCarree(central_longitude=0),
#                               cmap=cmap, vmin=vmin, vmax=vmax)
        
#         axs[i].set_title(f'{vars[i]} Difference\n({exp2_name} - {exp1_name})', fontsize=14)
#         cbar = fig.colorbar(im, ax=axs[i], orientation='vertical', shrink=0.5, fraction=0.03, pad=0.04)
#         cbar.set_label(units[i], fontsize=12)
        
#         # Set global extent to show full map
#         axs[i].set_global()
        
#         # Set consistent geographic ticks for all map plots
#         axs[i].set_xticks([-180, -120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
#         axs[i].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree(central_longitude=180))
    
#     # Create overall title
#     plt.suptitle(f'Composite Difference Maps at {confidence_level}% Confidence Level\n'
#                 f'({exp2_name} - {exp1_name})', fontsize=16)
#     plt.tight_layout()
    
#     # Save figure
#     save_name = f'composite_DIFFERENCE_{confidence_level}percent_confidence_{exp_codes[1]}_minus_{exp_codes[0]}_Z500.png'
#     plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/{save_name}', 
#                 format='png', dpi=250, bbox_inches='tight')
#     plt.close()
    
#     print(f"Difference composite plot saved for {exp2_name} - {exp1_name}")



def COMPARE_composite_inputmap_target(experiments, confidence_level_low = 20, confidence_level_high = 40, keyword = None):
    """
    For given confidence threshold range, plot difference maps between two experiment types with multiple experiments each.
    Calculate mean composite maps for each experiment type, then compute the difference.
    Now includes individual composite maps for each experiment type before showing differences.
    
    Args:
        experiments: Dict with experiment types as keys and lists of experiment names as values
                    e.g., {"OBS(OBS)": ["exp173", "exp174", ...], "E3SM-short(OBS)": ["exp189", "exp195", ...]}
        confidence_level_low: Lower bound of confidence range (percentile)
        confidence_level_high: Upper bound of confidence range (percentile)
        keyword: Optional keyword for filename
    """
    exps = experiments
    
    # Ensure we have exactly 2 experiment types
    if len(exps) != 2:
        raise ValueError("This function requires exactly 2 experiment types for comparison")
    
    exp_type_names = list(exps.keys())
    
    # convert confidence levels
    confidence_threshold_low = confidence_level_low / 100.0
    confidence_threshold_high = confidence_level_high / 100.0
    percentiles = np.linspace(100, 0, 21)

    colormaps = ["BrBG", "RdBu_r", "PuOr_r"]
    vars = ["Total Precip", "Skin Temp", "Z500"]
    units = ['(mm/day)', '(K)', '(m)']

    # Create figure with 3x3 layout: 
    # Row 1: First experiment type composites
    # Row 2: Second experiment type composites  
    # Row 3: Difference maps
    fig, axs = plt.subplots(3, 3, figsize=(18, 10), 
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # Store composite data for both experiment types
    composite_data = {}

    selected_dates_allexps = []
    
    for experiment_type, exp_list in exps.items():
        print(f'Processing experiment type: {experiment_type} with {len(exp_list)} experiments')
        
        # Store all composite maps for this experiment type
        all_composite_maps = [[] for _ in range(len(colormaps))]  # One list per variable
        total_samples = 0

        # Collect CRPS for confident and all samples
        selected_dates = []
        
        for exp_name in exp_list:
            print(f'  Processing experiment: {exp_name}')
            
            # Load the output and target data for this experiment
            output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')

            # Load crps for experiment: 
            # crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

            # compare crps to IQR CRPS information: 
            crps_iqr = open_data_file(f"/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_IQR_CRPS_discard.pkl")

            percentile_index_low = int(20 - (confidence_level_low / 5))
            percentile_index_high = int(20 - (confidence_level_high / 5))
            print(f"percentile index low {percentile_index_low}, high {percentile_index_high}")

            print(f"crps iqr 80: {crps_iqr['avg_crps'][percentile_index_low]} crps iqr 60: {crps_iqr['avg_crps'][percentile_index_high]}")

            # Load input maps based on experiment type
            if experiment_type in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)"]:
                config = utils.get_config(exp_name)
                input_data = config["input_data"]
                input_maps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/{input_data}_trimmed_test_dat.nc')
                input_maps = input_maps['x']
            else:  # OBS(OBS) type experiments
                input_maps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_maps = input_maps['x']

            # Load climatology statistics
            climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')

            # Load testing target data
            target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
            target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]

            # Calculate IQR for each sample
            iqr = iqr_basic(output)
            # print(f"iqr min: {np.min(iqr)}, iqr max: {np.max(iqr)}, iqr median: {np.median(iqr)}")

            # List IQR by percentile
            iqr_percentiles = np.percentile(iqr, percentiles)
            # print(f"iqr percentiles: {iqr_percentiles}")

            # Select samples whose IQR is within the given confidence range
            iqr_sorted = np.argsort(iqr)
            num_samps = len(iqr)

            lower_threshold = np.percentile(iqr, confidence_level_low)   
            upper_threshold = np.percentile(iqr, confidence_level_high) 

            selected_indices = np.where((iqr >= lower_threshold) & (iqr <= upper_threshold))[0]

            # keep selected dates for each experiment: 
            selected_dates_allexps.append(np.unique(input_maps['time'].values[selected_indices]))


            print(f"    Experiment: {exp_name}, Number of samples selected in {confidence_level_low}-{confidence_level_high}% confidence range: {len(selected_indices)}")
            print(f"      IQR range: {lower_threshold:.4f} to {upper_threshold:.4f}")
            if len(selected_indices) < 30:
                print(f"    WARNING: Less than 30 samples selected for {exp_name} in {confidence_level_low}-{confidence_level_high}% confidence range")

            total_samples += len(selected_indices)
            
            # Calculate composite maps for each variable for this experiment
            for i in range(len(colormaps)):
                mean_map = np.mean(input_maps[... , i].values[selected_indices, ...], axis=0)
                all_composite_maps[i].append(mean_map)

            # Calculate mean composite maps across all experiments of this type
            mean_composite_maps = []

        for i in range(len(colormaps)):
            # Average across all experiments for this variable
            mean_composite_map = np.mean(all_composite_maps[i], axis=0)
            mean_composite_maps.append(mean_composite_map)
        
        # Calculate mean CRPS across all experiments for this type
        mean_crps_confident = np.mean(crps_iqr['avg_crps'][percentile_index_high:percentile_index_low])
        print(f"type crps percentile: {type(crps_iqr['avg_crps'][percentile_index_low])}")
        print(f"crps low : {crps_iqr['avg_crps'][percentile_index_low]}, crps high: {crps_iqr['avg_crps'][percentile_index_high]}")
        print(f"multiple values: {crps_iqr['avg_crps'][percentile_index_high:percentile_index_low]}")
        mean_crps_all = np.mean(crps_iqr['avg_crps'])
        
        composite_data[experiment_type] = {
            'maps': mean_composite_maps,
            'lon': input_maps['lon'],  # Using last loaded input_maps for coordinates
            'lat': input_maps['lat'],
            'n_experiments': len(exp_list),
            'total_samples': total_samples,
            'all_crps_confident': crps_iqr['avg_crps'][percentile_index_low:percentile_index_high],
            'mean_crps_confident': mean_crps_confident,
            'mean_crps_all': mean_crps_all,
            'selected_dates': selected_dates
        }
        

    # Now plot all the maps
    exp1_type, exp2_type = exp_type_names[0], exp_type_names[1]
    
    # Find global min/max for consistent color scaling within each variable
    # Force zero to be at center by making limits symmetric
    global_abs_maxs = []
    for i in range(len(colormaps)):
        all_values = np.concatenate([
            composite_data[exp1_type]['maps'][i].flatten(),
            composite_data[exp2_type]['maps'][i].flatten()
        ])
        abs_max = np.max(np.abs(all_values))  # Find maximum absolute value
        global_abs_maxs.append(abs_max)
    
    for i in range(len(colormaps)):
        cmap = colormaps[i]
        
        # Set symmetric colorbar limits for individual composites (zero at center)
        abs_max_indiv = global_abs_maxs[i]
        vmin_indiv = -abs_max_indiv
        vmax_indiv = abs_max_indiv
        
        # Row 0: First experiment type composite
        axs[0, i].coastlines()
        axs[0, i].add_feature(cfeature.BORDERS, linestyle=':')
        im1 = axs[0, i].pcolormesh(composite_data[exp1_type]['lon'], 
                                   composite_data[exp1_type]['lat'], 
                                   composite_data[exp1_type]['maps'][i], 
                                   transform=ccrs.PlateCarree(central_longitude=0),
                                   cmap=cmap, vmin=vmin_indiv, vmax=vmax_indiv)
        
        exp1_crps_conf = composite_data[exp1_type]['mean_crps_confident']
        print(f"exp1_crps_conf: {exp1_crps_conf}")
        
        title1_text = f'{vars[i]} - {exp1_type}\n'
        title1_text += f'CRPS ({confidence_level_low}-{confidence_level_high}%): {exp1_crps_conf:.3f}'
        axs[0, i].set_title(title1_text, fontsize=10)
        
        cbar1 = fig.colorbar(im1, ax=axs[0, i], orientation='vertical', shrink=0.7, fraction=0.04, pad=0.02)
        cbar1.set_label(units[i], fontsize=10)
        axs[0, i].set_global()
        axs[0, i].set_xticks([-180, -120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
        axs[0, i].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree(central_longitude=180))
        
        # Row 1: Second experiment type composite
        axs[1, i].coastlines()
        axs[1, i].add_feature(cfeature.BORDERS, linestyle=':')
        im2 = axs[1, i].pcolormesh(composite_data[exp2_type]['lon'], 
                                   composite_data[exp2_type]['lat'], 
                                   composite_data[exp2_type]['maps'][i], 
                                   transform=ccrs.PlateCarree(central_longitude=0),
                                   cmap=cmap, vmin=vmin_indiv, vmax=vmax_indiv)
        
        exp2_crps_conf = composite_data[exp2_type]['mean_crps_confident']
        print(f"exp2_crps_conf: {exp2_crps_conf}")
        
        title2_text = f'{vars[i]} - {exp2_type}\n'
        title2_text += f'CRPS ({confidence_level_low}-{confidence_level_high}%): {exp2_crps_conf:.3f}'
        axs[1, i].set_title(title2_text, fontsize=10)
        
        cbar2 = fig.colorbar(im2, ax=axs[1, i], orientation='vertical', shrink=0.7, fraction=0.04, pad=0.02)
        cbar2.set_label(units[i], fontsize=10)
        axs[1, i].set_global()
        axs[1, i].set_xticks([-180, -120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
        axs[1, i].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree(central_longitude=180))
        
        # Row 2: Difference map (exp2_type minus exp1_type)
        diff_map = composite_data[exp2_type]['maps'][i] - composite_data[exp1_type]['maps'][i]
        
        # Set symmetric colorbar limits for difference
        abs_max = np.max(np.abs(diff_map))
        vmin_diff, vmax_diff = -abs_max, abs_max
        
        axs[2, i].coastlines()
        axs[2, i].add_feature(cfeature.BORDERS, linestyle=':')
        im3 = axs[2, i].pcolormesh(composite_data[exp1_type]['lon'], 
                                   composite_data[exp1_type]['lat'], 
                                   diff_map, 
                                   transform=ccrs.PlateCarree(central_longitude=0),
                                   cmap=cmap, vmin=vmin_diff, vmax=vmax_diff)
        
        title3_text = f'{vars[i]} Difference\n({exp2_type} - {exp1_type})'
        axs[2, i].set_title(title3_text, fontsize=10)
        
        cbar3 = fig.colorbar(im3, ax=axs[2, i], orientation='vertical', shrink=0.7, fraction=0.04, pad=0.02)
        cbar3.set_label(f'Δ{units[i]}', fontsize=10)
        axs[2, i].set_global()
        axs[2, i].set_xticks([-180, -120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
        axs[2, i].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree(central_longitude=180))
    
    # Create overall title with experiment counts and CRPS comparison
    exp1_count = composite_data[exp1_type]['n_experiments']
    exp2_count = composite_data[exp2_type]['n_experiments']
    
    suptitle_text = f'Multi-Experiment Composite Analysis for {confidence_level_low}-{confidence_level_high}% Confidence Range\n'
    suptitle_text += f'Top: {exp1_type} (n={exp1_count}) | Middle: {exp2_type} (n={exp2_count}) | Bottom: Difference\n'
    
    plt.suptitle(suptitle_text, fontsize=14, y=0.96)
    plt.tight_layout()
    
    # Create save name using experiment type names
    exp1_short = exp1_type.replace('(', '_').replace(')', '').replace('-', '_')
    exp2_short = exp2_type.replace('(', '_').replace(')', '').replace('-', '_')
    save_name = f'multi_composite_FULL_{confidence_level_low}to{confidence_level_high}percent_range_{exp1_short}_vs_{exp2_short}_n{exp1_count}_vs_n{exp2_count}_CRPS.png'
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/{save_name}', 
                format='png', dpi=250, bbox_inches='tight')
    plt.close()
    
    print(f"\nMulti-experiment full composite plot saved: {save_name}")
    print(f"Final comparison: {exp1_type} ({exp1_count} experiments) vs {exp2_type} ({exp2_count} experiments)")
    print(f"Confidence range: {confidence_level_low}% to {confidence_level_high}% (selecting samples between these IQR percentiles)")

    # # Create better CRPS comparison plots
    # plt.figure(figsize=(12, 8))

    # # Get the CRPS arrays
    # exp1_crps_confident = np.array(composite_data[exp1_type]['all_crps_confident'])
    # exp2_crps_confident = np.array(composite_data[exp2_type]['all_crps_confident'])

    # # Subplot 1: Histogram comparison
    # bins_shared = np.linspace(min(exp1_crps_confident.min(), exp2_crps_confident.min()), max(exp1_crps_confident.max(), exp2_crps_confident.max()), 80)

    # plt.subplot(2, 2, 1)
    # plt.hist(exp1_crps_confident, bins=bins_shared, alpha=0.7, label=f'{exp1_type} (confident)', color='blue', density=True)
    # plt.hist(exp2_crps_confident, bins=bins_shared, alpha=0.7, label=f'{exp2_type} (confident)', color='orange', density=True)
    # plt.xlabel('CRPS')
    # plt.ylabel('Density')
    # plt.title(f'CRPS Distribution for Confident Samples ({confidence_level_low}-{confidence_level_high}%)')
    # plt.legend()

    # # Subplot 2: Box plot comparison
    # plt.subplot(2, 2, 2)
    # plt.boxplot([exp1_crps_confident, exp2_crps_confident], 
    #             labels=[exp1_type, exp2_type])
    # plt.ylabel('CRPS')
    # plt.title('CRPS Box Plot Comparison (Confident Samples)')
    # plt.xticks(rotation=45)

    # plt.tight_layout()
    # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/CRPS_Analysis_{exp1_type}_vs_{exp2_type}_proper_aggregation.png', 
    #             format='png', dpi=250, bbox_inches='tight')
    # plt.close()  # Add this to close the figure

    return composite_data


def XAI_confidence_compositing(experiments, confidence_level_low = 20, confidence_level_high = 40, xai_method = 'integrated_gradients', keyword = None):
    """
    Calculate XAI attributions for specific confidence range for each shash parameter and each input variable based on input maps.
    
    """    
    exps = experiments
    exp_type_names = list(exps.keys())
    
    # convert confidence levels
    confidence_threshold_low = confidence_level_low / 100.0
    confidence_threshold_high = confidence_level_high / 100.0
    percentiles = np.linspace(100, 0, 21)

    colormaps = ["BrBG", "RdBu_r", "PuOr_r"]
    units = ['(mm/day)', '(K)', '(m)']
    vars = ["Total Precip", "Skin Temp", "Z500"]

    #XAI colors
    colors = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (173/255, 74/255, 1), (102/255, 10/255, 78/255)] #(220/255, 182/255, 251/255), 
    n_bins = 100
    cmap_name = "xai_purple"
    new_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Create figure with 5x3 layout: Row 1: Composite Input Maps, Row 2: XAI attribution maps
    fig, axs = plt.subplots(5, 3, figsize=(18, 10), 
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # Store composite data for both experiment types
    composite_data = {}

    selected_dates_allexps = []
        
    # Store all composite maps for this experiment type
    all_composite_maps = [[] for _ in range(len(colormaps))]  # One list per variable
    total_samples = 0

    # Collect CRPS for confident and all samples
    selected_dates = []

    for iexp_name, (exp_type, exp_name) in enumerate(exps.items()):
        print(f'  Processing experiment: {exp_name}')
        config = utils.get_config({exp_name})

        # OPEN ALL INPUTS -----------------------
        # Load the output and target data for this experiment
        output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')

        # compare crps to IQR CRPS information: 
        crps_iqr = open_data_file(f"/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_IQR_CRPS_discard.pkl")

        percentile_index_low = int(20 - (confidence_level_low / 5))
        percentile_index_high = int(20 - (confidence_level_high / 5))

        # Load input maps based on experiment type
        if exp_type in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)"]:
            config = utils.get_config(exp_name)
            input_data = config["input_data"]
            input_maps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/{input_data}_trimmed_test_dat.nc')
            input_maps = input_maps['x']
        else:  # OBS(OBS) type experiments
            input_maps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
            input_maps = input_maps['x']

        # Load climatology statistics
        climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')

        # Load testing target data
        target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
        target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]

        # Load Experiment Torch Model
        path = str(config["perlmutter_model_dir"]) + str(config["expname"]) + '.pth'
        load_model_dict = torch.load(path)
        state_dict = load_model_dict["model_state_dict"]
        std_mean = load_model_dict["training_std_mean"]
        device = utils.prepare_device(config["device"])

        model = TorchModel(
            config=config["arch"],
            target_mean=std_mean["trainset_target_mean"],
            target_std=std_mean["trainset_target_std"],
        )

        model.load_state_dict(state_dict)
        model.eval()

        # CALCULATE BASIC QUANTITIES -----------------------
        # Calculate IQR for each sample
        iqr = iqr_basic(output)

        lower_threshold = np.percentile(iqr, confidence_level_low)   
        upper_threshold = np.percentile(iqr, confidence_level_high) 

        selected_indices = np.where((iqr >= lower_threshold) & (iqr <= upper_threshold))[0]
        selected_dates = input_maps['time'].values[selected_indices]

        # keep selected dates for each experiment: 
        selected_dates_allexps.append(np.unique(input_maps['time'].values[selected_indices])) # TODO, COULD look at frequency of selected dates? Not just unique dates

        total_samples += len(selected_indices)

        # CALCULATE XAI and INPUT MAPS MEANS per EXPERIMENT -----------------------

        # Calculate XAI mean attribution map for each experiment:
        avg_attr = np.zeros((len(exps.items()), 3, 4, 180, 360))
        
        # Calculate composite maps for each variable for this experiment
        for i in range(len(colormaps)):
            mean_map = np.mean(input_maps[... , i].values[selected_indices, ...], axis=0)
            all_composite_maps[i].append(mean_map)

            for shash_param in range(3): # mu, sigma, gamma, tau
                avg_attr[iexp_name, i, shash_param, ...] = average_attributions(model, input_maps, selected_dates, device, shash_param, config, method= xai_method, keyword = f"{confidence_level_low}-{confidence_level_high}_Confidence")

    # CALCULATE XAI and INPUT MAPS MEANS ACROSS ALL EXPERIMENTS -----------------------

    # Calculate mean composite maps across all experiments of this type
    mean_composite_maps = []
    mean_composite_attr = np.zeros((3, 4, 180, 360))

    for i in range(len(colormaps)):
        # Average across all experiments for this variable
        mean_composite_map = np.mean(all_composite_maps[i], axis=0)
        mean_composite_maps.append(mean_composite_map)

        for shash_param in range(3):
            mean_composite_attr[i, shash_param, ...] = np.mean(avg_attr[..., shash_param, ...], axis=0)

    # Calculate mean CRPS across all experiments for this type
    mean_crps_confident = np.mean(crps_iqr['avg_crps'][percentile_index_high:percentile_index_low])

        # Calculate mean CRPS across all experiments for this type
    mean_crps_confident = np.mean(crps_iqr['avg_crps'][percentile_index_high:percentile_index_low])
    mean_crps_all = np.mean(crps_iqr['avg_crps'])

    # STORE ALL CALCULATED QUANTITIES -----------------------
    
    composite_data = {
        'maps': mean_composite_maps,
        'lon': input_maps['lon'],  # Using last loaded input_maps for coordinates
        'lat': input_maps['lat'],
        'n_experiments': len(exps.items()),
        'total_samples': total_samples,
        'all_crps_confident': crps_iqr['avg_crps'][percentile_index_low:percentile_index_high],
        'mean_crps_confident': mean_crps_confident,
        'mean_crps_all': mean_crps_all,
        'selected_dates': selected_dates,
        'mean_composite_attr': mean_composite_attr
    }

    for row in range(4):

        for i in range(len(colormaps)):
            cmap = colormaps[i]

            if row == 0: 
                # set vmin and vmax to be symmetric around zero for diverging colormaps
                vmin = np.min(composite_data['maps'])
                vmax = np.max(composite_data['maps'])
                abs_max = max(abs(vmin), abs(vmax))
                vmin = -abs_max
                vmax = abs_max

                # Create the map plot
                axs[row, i].coastlines()
                axs[row, i].add_feature(cfeature.BORDERS, linestyle=':')
                im = axs[row, i].pcolormesh(input_maps['lon'], input_maps['lat'], composite_data['maps'], transform=ccrs.PlateCarree(central_longitude= 0),cmap=cmap, vmin=vmin, vmax=vmax)
                axs[row, i].set_title(f'{exp_type} Composite {vars[i]} Map')
                cbar = fig.colorbar(im, ax=axs[0, i], orientation='vertical', shrink=0.6, fraction=0.02, pad=0.02)

                # Set global extent to show full map
                axs[row, i].set_global()
                
                # Set consistent geographic ticks for all map plots
                axs[row, i].set_xticks([-180, -120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
                axs[row, i].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree(central_longitude = 180))

            elif row > 0: 
                shash_index = row - 1

                axs[row, i].coastlines()
                axs[row, i].add_feature(cfeature.BORDERS, linestyle=':')
                im = axs[row, i].pcolormesh(input_maps['lon'], input_maps['lat'], composite_data['mean_composite_attr'][i, shash_index], transform=ccrs.PlateCarree(central_longitude= 0),cmap=new_cmap, vmin=vmin, vmax=vmax)
                axs[row, i].set_title(f'{exp_type} Composite {vars[i]} Map')
                cbar = fig.colorbar(im, ax=axs[0, i], orientation='vertical', shrink=0.6, fraction=0.02, pad=0.02)

                # Set global extent to show full map
                axs[row, i].set_global()

                # Set consistent geographic ticks for all map plots
                axs[row, i].set_xticks([-180, -120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
                axs[row, i].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree(central_longitude = 180))


    suptitle_text = f'Multi-Experiment XAI Composite Analysis ({confidence_level_low}-{confidence_level_high}% Confidence Range)\n'
    suptitle_text += f'{xai_method}'
    
    plt.suptitle(suptitle_text, fontsize=14, y=0.96)
    plt.tight_layout()
    
    save_name = f'{keyword}_composite_XAI_{confidence_level_low}to{confidence_level_high}_range_{xai_method}.png'
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/{save_name}', 
                format='png', dpi=250, bbox_inches='tight')
    plt.close()









#---------------------------------------------------------------------------
# future ideas: 
# def shash_bias_check(experiments):
#     """
#     Bias = The distance between the forecast and observation average values
    
#     Check the bias of the SHASH parameters for each experiment in the list of experiments against the climatological mean 

#     Plot the mean of shash means for each experiment and the climatological mean and climatology histogram

#     """
#     if "OBS(OBS)" in experiments: 
#         obs_exp = experiments["OBS(OBS)"]
#     if "E3SM(OBS)" in experiments: 
#         e3sm_obs_exp = experiments["E3SM(OBS)"]
#     if "E3SM-short(OBS)" in experiments:
#         e3sm_short_obs_exp = experiments["E3SM-short(OBS)"]
#     if "E3SM(E3SM)" in experiments: 
#         e3sm_exp = experiments["E3SM(E3SM)"]
#     elif "E3SM-short(E3SM)" in experiments:
#         e3sm_short_exp = experiments["E3SM-short(E3SM)"]
  