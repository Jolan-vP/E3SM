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
import analysis
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
from analysis.analysis_metrics import save_pickle as save_pickle
from matplotlib.colors import LinearSegmentedColormap
from XAI.captum import average_attributions, visualize_average_attributions
import torch
from model.build_model import TorchModel
from analysis import analysis_metrics
import databuilder.data_loader as data_loader

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
        1: "#019bba", 
        2: "#33c316",
        3: "#B6B309",
        }
    # color_themes = {
    #     0:  "#9335D1", 
    #     1:  "#3451D445    #     2:  "#2FB4C9",
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
                plt.plot(percentiles, avg_success_ratio, color=color_themes[i], alpha = 0.45, label = f"{experiment_type}", linewidth = 2.5)
            else: 
                plt.plot(percentiles, avg_success_ratio, color=color_themes[i], alpha = 0.45, linewidth = 2.5)
            # plt.fill_between(x = [0, 100], y1 = [0.5, 0.5], color = 'grey', alpha=0.03, edgecolor = None)


            plt.xlabel('IQR Percentile (% Data Remaining)')
            plt.ylabel('Proportion of Samples with Lower Network CRPS')
            plt.ylim(0.5, 0.85)
            plt.xlim(101, 4)
            plt.title('Increasing Confidence Success Ratio Discard Plot')
            plt.tight_layout()
        i += 1
            

    leg = plt.legend()
    for lh in leg.legendHandles: 
            lh.set_alpha(1)

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
        1: "#6e00b2", 
        2: "#d0326c",
        3: "#f97a0a"
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
                    obs_obs_color = "#0d0887"
                    plt.axhline(y=mean_climo_crps, color=obs_obs_color, linestyle='--', label = f'OBS Baseline Mean CRPS', linewidth = 2)
                if experiment_type == "E3SM-short(E3SM)": #or experiment_type == "OBS(E3SM)":
                    e3sm_color = "#cc4778"
                    plt.axhline(y=mean_climo_crps, color=e3sm_color, linestyle='--', label = f'E3SM Baseline Mean CRPS', linewidth = 2)
                plt.plot(percentile_dict, crps_dict, label=f'{experiment_type}', alpha = 0.4, linewidth = 2.5, color = color_themes[i])
            else:
                plt.plot(percentile_dict, crps_dict, alpha = 0.4, linewidth = 2.5, color = color_themes[i])

        plt.xlabel('IQR Percentile (% Data Remaining)')
        plt.ylabel('Average CRPS')
        plt.ylim(0.32, 0.68)
        plt.xlim(101, 4)
        plt.title('Increasing Confidence CRPS Discard Plot')
        plt.tight_layout()
        leg = plt.legend()
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

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
                plt.hist(iqr, bins=bin_edges, alpha=0.1, label=f'{experiment_type}', color = color_themes[i], density = True) #, histtype = 'step')
            else:
                plt.hist(iqr, bins=bin_edges, alpha=0.1, color = color_themes[i], density = True) #, histtype = 'step')

        i += 1
            
        plt.xlabel('IQR')
        plt.ylabel('Density')
        plt.title('IQR Distribution Across Model Types')
        leg = plt.legend()
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

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

    selected_dates_allexps = {}
    
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

            # compare crps to IQR CRPS information: 
            crps_iqr = open_data_file(f"/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_IQR_CRPS_discard.pkl")

            percentile_index_low = int(20 - (confidence_level_low / 5))
            percentile_index_high = int(20 - (confidence_level_high / 5))
            print(f"percentile index low {percentile_index_low}, high {percentile_index_high}")

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

            lower_threshold = np.percentile(iqr, confidence_level_low)   
            upper_threshold = np.percentile(iqr, confidence_level_high) 

            selected_indices = np.where((iqr >= lower_threshold) & (iqr <= upper_threshold))[0]
            selected_dates = input_maps['time'].values[selected_indices]

            selected_dates_allexps[exp_name] = selected_dates

            # # keep selected dates for each experiment: 
            # selected_dates_allexps.append(np.unique(input_maps['time'].values[selected_indices]))

            print(f"Experiment: {exp_name}, Number of samples selected in {confidence_level_low}-{confidence_level_high}% confidence range: {len(selected_indices)}")
            print(f"IQR range: {lower_threshold:.4f} to {upper_threshold:.4f}")
            if len(selected_indices) < 30:
                print(f"WARNING: Less than 30 samples selected for {exp_name} in {confidence_level_low}-{confidence_level_high}% confidence range")

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
            'selected_dates_per_exp': selected_dates_allexps
        }
    
        # Save composite data to file: 
        if experiment_type == "E3SM-short(OBS)": 
            name_shorthand = "E3SMshort-OBS"
        elif experiment_type == "OBS(OBS)":
            name_shorthand = "OBS-OBS"
        elif experiment_type == "E3SM(OBS)":
            name_shorthand = "E3SM-OBS"
        elif experiment_type == "E3SM-long(OBS)":
            name_shorthand = "E3SMlong-OBS"
        elif experiment_type == "E3SM-short(E3SM)":
            name_shorthand = "E3SMshort-E3SM"
        elif experiment_type == "E3SM-long(E3SM)":
            name_shorthand = "E3SMlong-E3SM"
             
        composite_savename = f"/pscratch/sd/p/plutzner/E3SM/saved/output/COMBINED/{name_shorthand}_composite_data_{confidence_level_low}-{confidence_level_high}_all_vars.pkl"
        save_pickle(composite_data[experiment_type], composite_savename)

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
    exp_types_str = ', '.join(exp_type_names)

    colormaps = ["BrBG", "RdBu_r", "PuOr_r"]
    units = ['(mm/day)', '(K)', '(m)']
    vars = ["Total Precip", "Skin Temp", "Z500"]

    #XAI colors
    colors = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (173/255, 74/255, 1), (102/255, 10/255, 78/255)] #(220/255, 182/255, 251/255), 
    n_bins = 100
    cmap_name = "xai_purple"
    new_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Create figure with 5x3 layout: Row 1: Composite Input Maps, Row 2-5: XAI attribution maps
    fig, axs = plt.subplots(5, 3, figsize=(18, 18), 
                           subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    experiment_counter = 0
    all_composite_maps = []

    total_experiments = sum(len(exp_list) for exp_list in exps.values())
    avg_attr_all_experiments = np.zeros((total_experiments, 4, 180, 360, 3)) # num experiments per experiment type, 4 shash params, lat, lon, 3 variables

    for iexp_type, (exp_type, exp_list) in enumerate(exps.items()):
        print(f'Processing experiment type: {exp_type}')
        # Load selected data for given confidence level: 
        if exp_type == "E3SM-short(OBS)": 
            name_shorthand = "E3SMshort-OBS"
        elif exp_type == "OBS(OBS)":
            name_shorthand = "OBS-OBS"
        elif exp_type == "E3SM(OBS)":
            name_shorthand = "E3SM-OBS"
        elif exp_type == "E3SM-long(OBS)":
            name_shorthand = "E3SMlong-OBS"
        elif exp_type == "E3SM-short(E3SM)":
            name_shorthand = "E3SMshort-E3SM"
        elif exp_type == "E3SM-long(E3SM)":
            name_shorthand = "E3SMlong-E3SM"

        if confidence_level_low == 20 and confidence_level_high == 40:
            selected_data_fn = f"/pscratch/sd/p/plutzner/E3SM/saved/output/COMBINED/{name_shorthand}_composite_data_{confidence_level_low}-{confidence_level_high}_all_vars.pkl"
            selected_data = open_data_file(selected_data_fn)
        else: 
            print("STOP: Must run composite comparison function with desired confidence levels first, then add to this if-statement.")

        for iexp_name, exp_name in enumerate(exp_list):  # Add this inner loop
            print(f'  Processing experiment: {exp_name}')
            config = utils.get_config(exp_name)

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
            # try opening model: 
            try:
                path = str(config["perlmutter_model_dir"]) + str(config["expname"]) + '.pth'
                load_model_dict = torch.load(path)
            except: 
                path = str(config["perlmutter_model_dir"]) + str(config["trained_model"]) + '.pth'
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

            # CALCULATE XAI and INPUT MAPS MEANS per EXPERIMENT -----------------------
            selected_dates = selected_data['selected_dates_per_exp'][exp_name]

            for shash_param in range(4):  # Loop over SHASH parameters
                avg_attr_all_experiments[experiment_counter, shash_param, ...] = average_attributions(model, input_maps, selected_dates, device, shash_param, config, method=xai_method, keyword=f"{confidence_level_low}-{confidence_level_high}_Confidence_ShashParam{shash_param}")
         
        experiment_counter += 1
    
    # CALCULATE XAI and INPUT MAPS MEANS ACROSS ALL EXPERIMENTS -----------------------
    # Calculate mean composite attribution maps across all experiments of this type
    mean_composite_attr = np.zeros((3, 4, 180, 360))

    print(f" selected mean maps: {selected_data['maps']}")
    print(f" selected mean maps: {len(selected_data['maps'])}")

    # for i in range(len(colormaps)):
    #     for shash_param in range(4):
    #         mean_composite_attr[i, shash_param, ...] = np.mean(avg_attr_all_experiments[:, shash_param, :, :, i], axis=0)

    # Normalize attribution maps so map contents sum to 1
    for i in range(len(colormaps)):
        for shash_param in range(4):
            attr_map = np.mean(avg_attr_all_experiments[:, shash_param, :, :, i], axis=0)
            
            abs_attr_map = np.abs(attr_map)  # Take absolute values for normalization
            total_attribution = np.sum(abs_attr_map)  # Sum of absolute attributions
            
            if total_attribution > 0:  # Avoid division by zero
                # Normalize while preserving the original signs
                normalized_attr_map = attr_map / total_attribution
                print(f"Variable {vars[i]}, SHASH param {shash_param}: normalized sum = {np.sum(np.abs(normalized_attr_map)):.6f}")
            else:
                normalized_attr_map = attr_map
                print(f"Variable {vars[i]}, SHASH param {shash_param}: zero attribution, no normalization")
                
            mean_composite_attr[i, shash_param, ...] = normalized_attr_map


    # Calculate mean CRPS across all experiments for this type
    mean_crps_confident = np.mean(crps_iqr['avg_crps'][percentile_index_high:percentile_index_low])
    mean_crps_all = np.mean(crps_iqr['avg_crps'])

    # STORE ALL CALCULATED QUANTITIES -----------------------
    composite_data = {
        'maps': selected_data['maps'],
        'attr_maps': mean_composite_attr, 
        'lon': input_maps['lon'],  # Using last loaded input_maps for coordinates
        'lat': input_maps['lat'],
        'n_experiments': len(exps.items()),
        'all_crps_confident': crps_iqr['avg_crps'][percentile_index_low:percentile_index_high],
        'mean_crps_confident': mean_crps_confident,
        'mean_crps_all': mean_crps_all,
        'selected_dates': selected_dates,
        'mean_composite_attr': mean_composite_attr
    }

    # Pre-calculate vmin/vmax for each variable across all SHASH parameters
    xai_scales = []
    for i in range(len(colormaps)):
        first_three_attr_values = mean_composite_attr[:, :3, :, :].flatten()

        # Get attribution values for tailweight parameter across ALL variables  
        tailweight_attr_values = mean_composite_attr[:, 3, :, :].flatten()

        # Calculate scales
        first_three_vmin = np.percentile(first_three_attr_values, 5)
        first_three_vmax = np.percentile(first_three_attr_values, 95)

        tailweight_vmin = np.percentile(tailweight_attr_values, 5)
        tailweight_vmax = np.percentile(tailweight_attr_values, 95)

        xai_scales = {
            'first_three': (first_three_vmin, first_three_vmax),
            'tailweight': (tailweight_vmin, tailweight_vmax)
        }
    
    for row in range(5):
        for i in range(len(colormaps)):
            cmap = colormaps[i]
            
            # First row: Input composite maps
            if row == 0: 
                # set vmin and vmax to be symmetric around zero for diverging colormaps
                vmin = np.min(composite_data['maps'][i])
                vmax = np.max(composite_data['maps'][i])
                abs_max = max(abs(vmin), abs(vmax))
                vmin = -abs_max
                vmax = abs_max

                # Create the map plot
                axs[row, i].coastlines()
                axs[row, i].add_feature(cfeature.BORDERS, linestyle=':')
                im = axs[row, i].pcolormesh(input_maps['lon'], input_maps['lat'], 
                                          composite_data['maps'][i], 
                                          transform=ccrs.PlateCarree(central_longitude=0),
                                          cmap=cmap, vmin=vmin, vmax=vmax)
                axs[row, i].set_title(f'Composite {vars[i]} Map {units[i]}')
                
                # Add colorbar for input maps
                cbar = fig.colorbar(im, ax=axs[row, i], orientation='vertical', 
                                  shrink=0.6, fraction=0.02, pad=0.02)
                
                # Set global extent to show full map
                axs[row, i].set_global()
                
                # Set consistent geographic ticks for all map plots
                axs[row, i].set_xticks([-180, -120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
                axs[row, i].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

            # Rows 1-4: XAI attribution maps for each SHASH parameter
            else: 
                shash_index = row - 1
                
                if shash_index < 3:  # First three parameters (μ, σ, ε) - same scale for all 9 plots
                    attr_vmin, attr_vmax = xai_scales['first_three']
                else:  # Tailweight parameter (δ) - same scale across 3 variables in row 4
                    attr_vmin, attr_vmax = xai_scales['tailweight']

                    
                attr_data = mean_composite_attr[i, shash_index]

                axs[row, i].coastlines()
                axs[row, i].add_feature(cfeature.BORDERS, linestyle=':')
                im = axs[row, i].pcolormesh(input_maps['lon'], input_maps['lat'], 
                                          attr_data,
                                          transform=ccrs.PlateCarree(central_longitude=0),
                                          cmap=new_cmap, vmin=attr_vmin, vmax=attr_vmax)
                
                # Add SHASH parameter names for better clarity
                shash_names = ['μ (location)', 'σ (scale)', 'ε (skewness)', 'δ (tailweight)']
                axs[row, i].set_title(f'XAI Attribution: {vars[i]} → {shash_names[shash_index]}')
                
                # Add colorbar for XAI attribution maps
                cbar = fig.colorbar(im, ax=axs[row, i], orientation='vertical', 
                                  shrink=0.6, fraction=0.02, pad=0.02)
                cbar.set_label('Normalized Attribution', rotation=270, labelpad=15)

                # Set global extent to show full map
                axs[row, i].set_global()

                # Set consistent geographic ticks for all map plots
                axs[row, i].set_xticks([-180, -120, -60, 0, 60, 120], crs=ccrs.PlateCarree())
                axs[row, i].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

    # Create a more descriptive title
    suptitle_text = f'Multi-Experiment XAI Composite Analysis ({confidence_level_low}-{confidence_level_high}% Confidence Range)\n'
    suptitle_text += f'Method: {xai_method} | Experiments: {exp_types_str}'
    
    plt.suptitle(suptitle_text, fontsize=12, y=0.98)
    plt.tight_layout()
    
    save_name = f'{keyword}_composite_XAI_{confidence_level_low}to{confidence_level_high}_range_{xai_method}.png'
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/{save_name}', 
                format='png', dpi=250, bbox_inches='tight')
    plt.close()
    
    
    return composite_data  # Return the data for potential further analysis



def teleconnection_bias_analysis(experiments, confidence_level_low = 20, confidence_level_high = 0, keyword = None):
    """
    For given experiments, compute outputs and plots for the following functions: 
    - monthly analysis 
    - results by ENSO phase
    - results by MJO phase
    
    """

    exps = experiments
    exp_type_names = list(exps.keys())
    exp_types_str = ', '.join(exp_type_names)
    
    for exp_type, exp_list in exps.items():
        print(f'Processing experiment type: {exp_type}')

        # preallocate storage for monthly analysis outputs across experiments of this type
        composite_iqr_by_month = [[] for _ in range(6)] # 6 months of interest: Oct-Mar

        composite_monthly_analysis_output = np.zeros((len(exp_list), 6, 4))  # experiments, months, mean_crps, mean_iqr, mean_target, count

        all_confident_dates = {}

        for exp_name in exp_list:
            print(f'  Processing experiment: {exp_name}')
            
            # Load the output and target data for this experiment
            output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')

            # compare crps to IQR CRPS information: 
            crps_iqr = open_data_file(f"/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_IQR_CRPS_discard.pkl")

            # open crps: 
            crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

            # Load testing target data
            if exp_type in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                # Load climatology statistics
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
       
            elif exp_type in ["E3SM-short(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]
            
            # target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]

            ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr = iqr_basic(output)
            percentiles = np.linspace(100, 0, 21)

            avg_crps = []
            avg_target = []
            avg_iqr = []
            sample_index = np.zeros((len(target), len(percentiles)))

            # Sort by IQR
            iqr_sorted_indices = np.argsort(iqr)
            iqr_sorted = iqr[iqr_sorted_indices]
            target_sorted = target[iqr_sorted_indices]
            crps_sorted = crps[iqr_sorted_indices]

            for ip, p in enumerate(percentiles):
                # percentage of samples to keep for each round of the loop
                num_to_keep = int(len(iqr_sorted) * p / 100)
                
                indices = iqr_sorted_indices[:num_to_keep]

                if len(indices) == 0:
                    avg_crps.append(np.nan)
                    avg_target.append(np.nan)
                    avg_iqr.append(np.nan)
                else:
                    avg_crps.append(np.mean(crps[indices]))
                    avg_target.append(np.mean(target[indices]))
                    avg_iqr.append(np.mean(iqr[indices]))
                    sample_index[:len(indices), ip] = indices
            
            percentile_index_low = int(20 - (confidence_level_low / 5))
            percentile_index_high = int(20 - (confidence_level_high / 5))
            # print(f"percentile index low {percentile_index_low}, high {percentile_index_high}")
            # print(f"confidence levels: {confidence_level_low}, {confidence_level_high}")

            # Find unique indices in the columns corresponding to correct confidence levels
            all_indices_to_low_index = sample_index[:, percentile_index_low]
            indices_above_high_index = sample_index[:, percentile_index_high + 1] if (percentile_index_high + 1) < sample_index.shape[1] else np.array([])

            # select samples that are within all_indices_to_low_index but NOT in indices_above_high_index
            indices_within_confidence = np.setdiff1d(all_indices_to_low_index, indices_above_high_index)
            dates_within_confidence = target.time[indices_within_confidence.astype(int)]

            all_confident_dates[exp_name] = dates_within_confidence

            selected_crps = crps[indices_within_confidence.astype(int)]
            # print(f"selected crps: {np.mean(selected_crps)}")
            selected_target = target.sel(time = dates_within_confidence)
            selected_iqr = iqr[indices_within_confidence.astype(int)]

            # Call monthly analysis function
            iqr_by_month, crps_by_month, target_by_month, mean_data_by_month = monthly_analysis(selected_iqr, selected_crps, selected_target) 
            months_of_interest = [10, 11, 12, 1, 2, 3]

            # Aggregate monthly iqr analysis outputs across experiments of this type
            for i, month_num in enumerate(months_of_interest):
                    composite_iqr_by_month[i].append(iqr_by_month[month_num])

            composite_monthly_analysis_output[exp_list.index(exp_name), :, :] = mean_data_by_month

        mean_composite_monthly = np.mean(composite_monthly_analysis_output, axis=0)

        # Plot IQR distributions, one per month, overlayed to see the difference in spread by month
        plt.figure(figsize=(10, 6))
        colors = ['#d73027', '#fc8d59', "#c8b47c", "#8b9a9f", "#4b9acb", "#1c58a6"]
        labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']

        # Collect data for all months
        all_month_data = []
        month_colors = []
        month_labels = []

        for i in range(len(composite_iqr_by_month)):
            if composite_iqr_by_month[i]:
                # Flatten all experiments for this month into one dataset
                flattened_month_data = [item for sublist in composite_iqr_by_month[i] for item in sublist]
                all_month_data.append(flattened_month_data)
                month_colors.append(colors[i])
                month_labels.append(labels[i])

        # Stacked Histogram of IQR distribution by month
        plt.hist(all_month_data, bins=100, alpha=0.7, color=month_colors, 
                histtype='bar', stacked=True, label=month_labels)

        plt.xlabel('IQR Values')
        plt.ylabel('Density')
        plt.title(f'IQR Distribution by Month | {exp_type} \n Confidence Range: {confidence_level_low}-{confidence_level_high}%')
        plt.xlim([0.75, 2])
        plt.legend()
        plt.show()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/IQR_monthly_analysis_{exp_type}_{confidence_level_low}to{confidence_level_high}.png', format = 'png', dpi = 250)


        # Temporal Distribution of selected samples by month using dates_within_confidence
        # def find_chunks(dates, max_gap=3):
        #     """Find consecutive chunks in dates with gaps <= max_gap days"""
        #     if len(dates) < 2:
        #         return [len(dates)] if len(dates) == 1 else []
            
        #     day_diffs = np.diff(dates.dt.dayofyear.values)
        #     chunks = []
        #     chunk_start = 0
            
        #     for i, gap in enumerate(day_diffs):
        #         if gap > max_gap:  # End current chunk
        #             chunks.append(i + 1 - chunk_start)
        #             chunk_start = i + 1
            
        #     # Add final chunk
        #     chunks.append(len(dates) - chunk_start)
        #     return chunks

        # # Setup
        # fall_months = [10, 11, 12]
        # first_exp = exp_list[0]
        # max_gap = 3
        # target_year = 2013  # Pick a specific year that has data

        # print(f"Analyzing consecutive chunks for experiment: {first_exp}")

        # # Get and filter dates
        # dates_fall = all_confident_dates[first_exp]
        # dates_fall = dates_fall[dates_fall.dt.month.isin(fall_months)]

        
        # Line Plot of mean CRPS, IQR, Target, and Sample Count by month
        plt.figure(figsize=(8, 5))
        month_positions = [0, 1, 2, 3, 4, 5]
        month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']
        plt.plot(month_positions, mean_composite_monthly[:, 0], marker='o', label='Mean CRPS', color='#3b528b')
        plt.plot(month_positions, mean_composite_monthly[:, 1], marker='o', label='Mean IQR (sigma)', color='#019bba')
        plt.plot(month_positions, mean_composite_monthly[:, 2], marker='o', label='Mean Target (m)', color='#33c316')
        # plt.bar(months, mean_composite_monthly[:, 3], alpha=0.3, label='Sample Count', color='gray')
        plt.xlabel('Month')
        plt.ylabel('Values')
        plt.title(f'Monthly Analysis of CRPS, IQR, Target, and Sample Count | {exp_type} \n Confidence Range: {confidence_level_low}-{confidence_level_high}%')
        plt.xticks(month_positions, month_labels)
        plt.ylim([-0.3, 1.8])
        plt.legend(loc = 'upper left')
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/monthly_analysis_{exp_type}_{confidence_level_low}to{confidence_level_high}.png', format = 'png', dpi = 250)

            # Call ENSO phase analysis function
            # enso_phase_analysis_output = enso_phase_analysis({exp_name: [exp_name]

def monthly_analysis(iqr, crps, target):
    """
    For given experiments, plot mean CRPS, IQR, target value, and number of samples per month on plot vs month.
    """

    # select months of interest based on months found in target data: 
    months_of_interest = [10, 11, 12, 1, 2, 3]
    month_labels = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar']

    iqr_by_month = {}
    crps_by_month = {}
    target_by_month = {}

    data_by_month = np.zeros((len(months_of_interest), 4))  # month, mean_crps, mean_iqr, mean_target, count

    for i, month in enumerate(months_of_interest):
        # select by month using xarray
        month_mask = (target.time.dt.month == month).values
        if np.sum(month_mask) == 0:
            pass

        iqr_by_month[month] = iqr[month_mask]
        crps_by_month[month] = crps[month_mask]
        target_by_month[month] = target[month_mask]

        data_by_month[i, 0] = np.mean(crps[month_mask])
        data_by_month[i, 1] = np.mean(iqr[month_mask])
        data_by_month[i, 2] = np.mean(target[month_mask])
        data_by_month[i, 3] = np.sum(month_mask)


    return iqr_by_month, crps_by_month, target_by_month, data_by_month



def anom_var_distributions(experiments, keyword = None):
    """
    (1) Variance by month across all random seeds for given experiment type
    (2) Stacked histogram of count of positive vs negative anomalies by month (October, November, December individually)    
    
    """

    exps = experiments

    for exp_type, exp_list in exps.items():
  
        all_monthly_pos_anoms = []
        all_monthly_neg_anoms = []

        for exp_name in exp_list:
            print(f'  Processing experiment: {exp_name}')
            # Load the output and target data for this experiment
            output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')

            # open crps: 
            crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

             ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr = iqr_basic(output)
            percentiles = np.linspace(100, 0, 21)
            
            # Load testing target data
            if exp_type in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                # Load climatology statistics
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
                target_label = "OBS"
       
            elif exp_type in ["E3SM-short(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]
                target_label = "E3SM"

            avg_crps = []
            avg_target = []
            avg_iqr = []
            sample_index = np.zeros((len(target), len(percentiles)))

            # Sort by IQR
            iqr_sorted_indices = np.argsort(iqr)
            iqr_sorted = iqr[iqr_sorted_indices]
            target_sorted = target[iqr_sorted_indices]
            crps_sorted = crps[iqr_sorted_indices]

            for ip, p in enumerate(percentiles):
                # percentage of samples to keep for each round of the loop
                num_to_keep = int(len(iqr_sorted) * p / 100)
                
                indices = iqr_sorted_indices[:num_to_keep]

                if len(indices) == 0:
                    avg_crps.append(np.nan)
                    avg_target.append(np.nan)
                    avg_iqr.append(np.nan)
                else:
                    avg_crps.append(np.mean(crps[indices]))
                    avg_target.append(np.mean(target[indices]))
                    avg_iqr.append(np.mean(iqr[indices]))
                    sample_index[:len(indices), ip] = indices

            # Count positive and negative anomalies by month with their corresponding IQR values
            monthly_pos_count = []
            monthly_neg_count = []
            monthly_pos_iqr = []
            monthly_neg_iqr = []
            
            for month in [1, 2, 3, 10, 11, 12]:  
                month_mask = target['time.month'] == month
                month_target = target[month_mask]
                month_iqr = iqr[month_mask]
                
                # Positive anomalies
                pos_mask = month_target > 0
                pos_count = pos_mask.sum().values
                pos_iqr_vals = month_iqr[pos_mask]
                
                # Negative anomalies  
                neg_mask = month_target < 0
                neg_count = neg_mask.sum().values
                neg_iqr_vals = month_iqr[neg_mask]
                
                monthly_pos_count.append(pos_count)
                monthly_neg_count.append(neg_count)
                monthly_pos_iqr.append(pos_iqr_vals)
                monthly_neg_iqr.append(neg_iqr_vals)
            
            all_monthly_pos_anoms.append(monthly_pos_iqr)
            all_monthly_neg_anoms.append(monthly_neg_iqr)

        plot_all_months_anomaly_histograms(all_monthly_pos_anoms, all_monthly_neg_anoms, exp_list, exp_type)

    target_ERA5 = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
    # Load climatology statistics
    climatology_stats_ERA5 = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')
    target_ERA5 = (target_ERA5['y'] - climatology_stats_ERA5['z'][2]) / climatology_stats_ERA5['z'][3]
    target_ERA5_label = "OBS"
       
    target_E3SM = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
    climatology_stats_E3SM = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
    target_E3SM = (target_E3SM['y'] - climatology_stats_E3SM['Z500'][2]) / climatology_stats_E3SM['Z500'][3]
    target_E3SM_label = "E3SM"

    # variance by month of target data:
    monthly_variance_E3SM = target_E3SM.groupby('time.month').var(dim='time')
    monthly_variance_ERA5 = target_ERA5.groupby('time.month').var(dim='time')

    # Convert xarray to numpy array and handle scalar case
    if hasattr(monthly_variance_E3SM, 'values'):
        monthly_variance_E3SM = monthly_variance_E3SM.values  # Extract numpy array from xarray

    # Handle the case where it might be a scalar after taking mean
    if np.isscalar(monthly_variance_E3SM):
        print(f"Warning: mean_monthly_var is a scalar: {monthly_variance_E3SM}")
    else:
        plt.figure(figsize=(8, 5))
        months = [0, 1, 2, 3, 4, 5]  # Positions for Oct, Nov, Dec, Jan, Feb, Mar}
        month_labels = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
        
        # Now we can safely use len() on the numpy array
        x_positions = list(range(len(monthly_variance_E3SM)))
        plt.scatter(x_positions, monthly_variance_E3SM, label='E3SM')
        plt.scatter(x_positions, monthly_variance_ERA5, label='ERA5')
        plt.xlabel('Month')
        plt.ylabel('Variance')
        plt.title(f'Mean Monthly Variance of Target Variable | {target_label}')
        plt.xticks(x_positions, month_labels[:len(monthly_variance_E3SM)])
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/monthly_variance_E3SMvsERA5.png', format='png', dpi=250)
            





def plot_all_months_anomaly_histograms(all_monthly_pos_anoms, all_monthly_neg_anoms, exp_names, exp_type):
    """
    Plot stacked histograms of positive and negative anomaly counts vs IQR for specified months
    Creates 2x3 subplot layout for 6 months
    """
    months_of_interest = [1, 2, 3, 10, 11, 12]  
    month_names = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
    
    # Create figure with 2x3 subplots for 6 months
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()  
    
    import matplotlib.cm as cm
    viridis = cm.get_cmap('Purples')
    neg_color = viridis(0.8)  # Dark purple
    pos_color = viridis(0.3)  # Light purple
    
    # Determine global IQR range for consistent binning across the 6 months of interest
    global_iqr_vals = []
    for i_month, month_idx in enumerate(months_of_interest):
        month_pos_iqr = []
        month_neg_iqr = []
        
        for exp_idx in range(len(all_monthly_pos_anoms)):
            month_pos = all_monthly_pos_anoms[exp_idx][i_month]
            month_neg = all_monthly_neg_anoms[exp_idx][i_month]

        global_iqr_vals.extend(month_pos_iqr + month_neg_iqr)
    
    # Create global bins if we have data
    if len(global_iqr_vals) > 0:
        global_bins = np.linspace(min(global_iqr_vals), max(global_iqr_vals), 25)
    else:
        global_bins = np.linspace(0.6, 1.9, 25)
    
    # Plot each of the 6 months of interest
    for plot_idx in range(6):  # 0 through 5 for the 6 months
        ax = axes[plot_idx]
    
        # Combine all experiments' data for this month
        month_pos_iqr = []
        month_neg_iqr = []
        
        for exp_idx in range(len(all_monthly_pos_anoms)):
            month_pos = all_monthly_pos_anoms[exp_idx][plot_idx]
            month_neg = all_monthly_neg_anoms[exp_idx][plot_idx]

            month_pos_iqr.extend(month_pos.values if hasattr(month_pos, 'values') else month_pos)
            month_neg_iqr.extend(month_neg.values if hasattr(month_neg, 'values') else month_neg)

            global_iqr_vals.extend(month_pos_iqr + month_neg_iqr)

        # Create stacked histogram
        if len(month_pos_iqr) > 0 or len(month_neg_iqr) > 0:
            # Always create both arrays, use empty arrays if no data
            plot_data = [
                month_neg_iqr if len(month_neg_iqr) > 0 else [],
                month_pos_iqr if len(month_pos_iqr) > 0 else []
            ]
            try:
                ax.hist(plot_data, 
                        bins=global_bins, 
                        stacked=True,
                        density=True,
                        color=[neg_color, pos_color],
                        alpha=0.8,
                        edgecolor='black',
                        linewidth=0.3)
                print(f"  -> Successfully plotted month {plot_idx}")
            except Exception as e:
                print(f"  -> Error plotting month {plot_idx}: {e}")
        else:
            print(f"  -> No data for month {plot_idx}")
               
        ax.set_title(month_names[plot_idx], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 5.15])
        
        # Only add labels to edge subplots to avoid clutter
        if plot_idx >= 3:  # Bottom row
            ax.set_xlabel('IQR', fontsize=10)
        if plot_idx % 3 == 0:  # Left column
            ax.set_ylabel('Density', fontsize=10)
    
    # Add overall title and legend
    fig.suptitle(f'Monthly Anomaly Distributions by IQR | {exp_type}', fontsize=14, y=0.95)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=neg_color, alpha=0.8, label='Negative Anomalies'),
                      Patch(facecolor=pos_color, alpha=0.8, label='Positive Anomalies')]
    fig.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88) 
    plt.savefig(f"/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/monthly_anomaly_histograms_{exp_type}.png",)

    # Print proportion of positive vs negative anomalies for each month: 
    for i_month, month_idx in enumerate(months_of_interest):
        total_pos = sum(len(all_monthly_pos_anoms[exp_idx][i_month]) for exp_idx in range(len(exp_names)))
        total_neg = sum(len(all_monthly_neg_anoms[exp_idx][i_month]) for exp_idx in range(len(exp_names)))
        total = total_pos + total_neg
        if total > 0:
            pos_pct = (total_pos / total) * 100
            neg_pct = (total_neg / total) * 100
            print(f"Month {month_names[i_month]}: Positive Anomalies: {total_pos} ({pos_pct:.1f}%), Negative Anomalies: {total_neg} ({neg_pct:.1f}%)")
        else:
            print(f"Month {month_names[i_month]}: No anomalies found.")





def m2m_sample_transfer(experiments, selection_method = 'confident_by_month', keyword = None):
    """
    Select samples from either OBS(OBS) or E3SM(E3SM) using variety of methods: 
        - select most confident sample from each month from each random seed
        - select most confident percentage of samples from total samples from each random seed
        - select dates that are input by hand
    
    Run selected samples through opposing model type (OBS->E3SM or E3SM->OBS) and compare results to original model type.
    
    for EACH sample separately, plot: 
        - First SHASH
        - Second SHASH
        - CRPS of first SHASH
        - CRPS of second SHASH
        - IQR of first SHASH
        - IQR of second SHASH

    Compute: 
        - Which month of the year do samples come from
        - ENSO phase of samples
        - MJO phase of samples
        - Target value of anomaly 

    """

    exps = experiments
    exp_type_names = list(exps.keys())
    exp_types_str = ', '.join(exp_type_names)
    
    for exp_type, exp_list in exps.items():
        print(f'Processing experiment type: {exp_type}')
        data_from_all_seeds1 = {}
        data_from_all_seeds2 = {}
        all_selected_indices = []

        for exp_name in exp_list:
            print(f'  Processing experiment: {exp_name}')
            selected_samples = {}
            config = utils.get_config(exp_name)
            
            # Load the output and target data for this experiment
            output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')

            # open crps: 
            crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

                ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr = iqr_basic(output)
            percentiles = np.linspace(100, 0, 21)
            
            # Load testing target data
            if exp_type in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                data_type = "OBS"
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
                target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
        
            elif exp_type in ["E3SM-short(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                data_type = "E3SM"
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp153_ERA5_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]
                target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]


            if selection_method == 'confident_by_month':
                # select most confident prediction from each month from each random seed
                # Identify most confident sample from each month
                selected_indices = []

                for month in [1, 2, 3, 10, 11, 12]:  
                    month_mask = target['time.month'] == month
                    if np.sum(month_mask) == 0:
                        continue

                    month_iqr = iqr[month_mask]
                    month_indices = np.where(month_mask)[0]

                    # Find index of minimum IQR in this month
                    min_iqr_idx = np.argmin(month_iqr)
                    selected_indices.append(month_indices[min_iqr_idx]) # this collects from all months for this random seed
            else: 
                print(f"choose sample selection method or code another one up")

            # identify target dates for these conf samples
            selected_target_dates = target['time'][selected_indices]
            # use lagtime to identify input dates for these conf samples
            lagtime = config['databuilder']['lagtime']
            selected_input_dates = selected_target_dates - np.timedelta64(lagtime, 'D')

            selected_samples["output1"] = output[selected_indices]
            selected_samples["target_date1"] = selected_target_dates
            selected_samples["input_date1"] = selected_input_dates
            selected_samples["iqr1"] = iqr[selected_indices]
            selected_samples["crps1"] = crps[selected_indices]

            # accumulate all selected indices from the test dataset: 
            all_selected_indices.extend(selected_indices)

            # print(f"selected indices: {selected_indices}")
            # print(f"selected target dates: {selected_target_dates.values}")
            # print(f"selected input dates: {selected_input_dates.values}")
            # print(f"selected samples: output1: {selected_samples['output1']}")
            
            # ---- TODO identify ENSO and MJO phase of selected samples ----------------
            # Open ENSO dates for E3SM vs OBS data: 
            enso_dates_pkl = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_daily_enso_timestamps.pkl')
            # check which key (category) each of the target dates falls into, and create a list with either "EN", "LN" or "N"
            enso_phase = []
            for date in selected_target_dates.values:
                if date in enso_dates_pkl['El Nino']:
                    enso_phase.append("EN")
                elif date in enso_dates_pkl['La Nina']:
                    enso_phase.append("LN")
                else:
                    enso_phase.append("N")

            selected_samples["enso_phase"] = enso_phase

            # MJO: 
            phase_timestamps = analysis_metrics.mjo_timestamps(data_type, config)
            selected_mjo_phase = phase_timestamps.sel(time=selected_target_dates)

            selected_samples["mjo_phase"] = selected_mjo_phase
            data_from_all_seeds1[str(exp_name)] = selected_samples

        # print(f"all selected indices: {all_selected_indices}, len: {len(all_selected_indices)}")
        
        # Find corresponding samples in opposing model type: 
        if exp_type in ["OBS(OBS)"]:
            opposing_exp = "E3SM-short(OBS)"
            models = ["exp189", "exp195", "exp196", "exp197", "exp198", "exp199"]
        elif exp_type in ["E3SM-short(E3SM)"]: 
            opposing_exp = "OBS(E3SM)"
            models = ["exp206", "exp207", "exp208", "exp209", "exp210", "exp211", "exp212", "exp213", "exp214", "exp215", "exp216", "exp217"]

        # For each selected sample from original model, find corresponding sample in opposing model using input date
        # Collect all target dates from 'data_from_all_seeds[exp_name]["target_date1"]'
        all_target1_dates = []
        all_output1 = []
        all_iqr1 = []
        all_crps1 = []

        for iexp, exp_name in enumerate(exp_list):
            all_target1_dates.extend(data_from_all_seeds1[exp_name]["target_date1"].values)
            all_output1.extend(data_from_all_seeds1[exp_name]["output1"])
            all_iqr1.extend(data_from_all_seeds1[exp_name]["iqr1"])
            all_crps1.extend(data_from_all_seeds1[exp_name]["crps1"])

        for ood_model in models: 
            print(f"  Processing opposing model: {ood_model}")
            selected_samples = {}
            config = utils.get_config(ood_model)
            
            # Load the output and target data for this experiment
            output_ood = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{ood_model}/{ood_model}_network_SHASH_parameters.pkl')

            # open crps: 
            crps_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{ood_model}/{ood_model}_CRPS_network_values.pkl')

                ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr_ood = iqr_basic(output_ood)
            percentiles = np.linspace(100, 0, 21)
            
            # Load testing target data
            if opposing_exp in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                target_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                climatology_stats_ood = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                target_ood = (target_ood['y'] - climatology_stats_ood['z'][2]) / climatology_stats_ood['z'][3]
        
            elif opposing_exp in ["E3SM-short(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                target_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                climatology_stats_ood = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                target_ood = (target_ood['y'] - climatology_stats_ood['Z500'][2]) / climatology_stats_ood['Z500'][3]

            # use target_dates1 to identify results from other models for these conf samples
            selected_samples["output2"] = output[all_selected_indices]
            selected_samples["target_date2"] = all_target1_dates
            selected_samples["input_date2"] = selected_input_dates
            selected_samples["iqr2"] = iqr[all_selected_indices]
            selected_samples["crps2"] = crps[all_selected_indices]
            selected_samples["enso_phase2"] = data_from_all_seeds1[exp_list[0]]["enso_phase"]
            selected_samples["mjo_phase2"] = data_from_all_seeds1[exp_list[0]]["mjo_phase"]

            data_from_all_seeds2[str(ood_model)] = selected_samples

        # ---- PLOTTING ---------------------------------------------------------
        # for each sample, plot SHASH curves, climatology, CRPS, IQR for both models

        # SUMMARY PLOT: 
        # 4 panels: (1) shash curves (2) MJO phase distributions (3) ENSO phase distribution (4) target value distribution


        # INDIVIDUAL PLOTS: 
        # Select single random seed experiment to plot individual samples from
        # 2 panels (1) shash curves (2) input map

        


        # plot summary statistics; distribution of MJO phase, ENSO phase for most confident sample by month in every random seed 
        # plot distribution of target values for most confident sample by month in every random seed
        # aggregate infromation across all random seeds (using append?)


        






# -----------------------------------------------------------------------------------------------------
# ------------------------------------------------- GARBAGE LAND ----------------------------------------

        # ============================================================================
        # MAIN PLOT: Show gaps with different emphasis for large vs small gaps
        # ============================================================================
        # fig, ax = plt.subplots(figsize=(15, 8))

        # # Colors for different seed experiments
        # seed_colors = plt.cm.viridis(np.linspace(0, 1, len(exp_list)))

        # large_gaps_count = 0
        # total_gaps_count = 0

        # for seed_idx, exp_name in enumerate(exp_list):
        #     dates_within_fall = all_confident_dates[exp_name][all_confident_dates[exp_name].dt.month.isin(fall_months)]
            
        #     if len(dates_within_fall) > 0:
        #         # Sort dates to ensure proper sequential analysis
        #         dates_sorted = dates_within_fall.sortby(dates_within_fall)
                
        #         # Group by year to handle year boundaries properly
        #         for year in np.unique(dates_sorted.dt.year.values):
        #             year_dates = dates_sorted[dates_sorted.dt.year == year]
                    
        #             if len(year_dates) > 1:
        #                 # Calculate day-of-year differences
        #                 day_diffs = np.diff(year_dates.dt.dayofyear.values)
                        
        #                 # For dates that span across months, we need to handle them carefully
        #                 for i, (date, gap) in enumerate(zip(year_dates[:-1], day_diffs)):
        #                     month = date.dt.month.item()
        #                     if month in fall_months:
        #                         total_gaps_count += 1
                                
        #                         # Determine x-position based on month
        #                         # Add small offset for each seed
        #                         seed_offset = (seed_idx - len(exp_list)/2 + 0.5) * 0.08
        #                         x_pos = month_positions[month] + seed_offset
                                
        #                         # Different visualization for large gaps vs normal gaps
        #                         if gap >= large_gap_threshold:
        #                             large_gaps_count += 1
        #                             # Large gaps: prominent red stems
        #                             markerline, stemlines, baseline = ax.stem([x_pos], [gap], 
        #                                                                     linefmt='red', markerfmt='ro', basefmt=' ')
        #                             stemlines.set_linewidth(3)
        #                             markerline.set_markersize(10)
        #                         else:
        #                             # Small gaps: subtle stems
        #                             markerline, stemlines, baseline = ax.stem([x_pos], [gap], 
        #                                                             linefmt='-', markerfmt='o', basefmt=' ')
        #                             stemlines.set_linewidth(1)
        #                             markerline.set_markersize(5)
        #                             markerline.set_alpha(0.6)
        #                             stemlines.set_alpha(0.6)

        # # Add horizontal line to show large gap threshold
        # ax.axhline(y=large_gap_threshold, color='red', linestyle='--', alpha=0.7, 
        #         label=f'Large Gap Threshold ({large_gap_threshold} days)')

        # # Customize the plot
        # ax.set_xticks([0, 1, 2])
        # ax.set_xticklabels(['Oct', 'Nov', 'Dec'])
        # ax.set_xlabel('Month', fontsize=12)
        # ax.set_ylabel('Days Between Sequential Confident Predictions', fontsize=12)
        # ax.set_title(f'Sequential Date Gaps in Fall Months | {exp_type}\n'
        #             f'Most dates are sequential (1-2 day gaps) with occasional large gaps (≥{large_gap_threshold} days)\n'
        #             f'Large gaps: {large_gaps_count}/{total_gaps_count} ({100*large_gaps_count/total_gaps_count:.1f}%)',
        #             fontsize=14)

        # ax.legend()
        # ax.grid(True, alpha=0.3)
        # ax.set_ylim(0, max(50, ax.get_ylim()[1]))  # Ensure we can see large gaps clearly

        # plt.tight_layout()
        # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/sequential_gaps_{exp_type}_{confidence_level_low}to{confidence_level_high}.png', 
        #             format='png', dpi=250)
        # plt.show()

        # ============================================================================
        # SUPPLEMENTARY PLOT: Histogram of gap sizes to show the distribution
        # ============================================================================
        # fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

        # all_gaps = []
        # large_gaps = []

        # for exp_name in exp_list:
        #     dates_within_fall = all_confident_dates[exp_name][all_confident_dates[exp_name].dt.month.isin(fall_months)]
            
        #     if len(dates_within_fall) > 0:
        #         dates_sorted = dates_within_fall.sortby(dates_within_fall)
                
        #         for year in np.unique(dates_sorted.dt.year.values):
        #             year_dates = dates_sorted[dates_sorted.dt.year == year]
                    
        #             if len(year_dates) > 1:
        #                 day_diffs = np.diff(year_dates.dt.dayofyear.values)
        #                 all_gaps.extend(day_diffs)
        #                 large_gaps.extend([gap for gap in day_diffs if gap >= large_gap_threshold])

        # # Histogram of all gaps
        # ax1.hist(all_gaps, bins=range(1, 50), alpha=0.7, color='skyblue', edgecolor='black')
        # ax1.set_yscale('log')
        # ax1.axvline(x=large_gap_threshold, color='red', linestyle='--', 
        #         label=f'Large Gap Threshold ({large_gap_threshold} days)')
        # ax1.set_xlabel('Gap Size (days)')
        # ax1.set_ylabel('Frequency')
        # ax1.set_ylim(0, 10**4)
        # ax1.set_title(f'Distribution of All Gap Sizes | {exp_type}\n Confidence Level: {confidence_level_low} to {confidence_level_high}')
        # ax1.legend()
        # ax1.grid(True, alpha=0.3)

        # plt.tight_layout()
        # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/gap_distribution_{exp_type}_{confidence_level_low}to{confidence_level_high}.png', 
        #             format='png', dpi=250)
        # plt.show()

        # ============================================================================
        # DETAILED ANALYSIS: Summary statistics
        # ============================================================================
        # print(f"\n=== GAP ANALYSIS SUMMARY for {exp_type} ===")
        # print(f"Total number of gaps analyzed: {len(all_gaps)}")
        # print(f"Number of large gaps (≥{large_gap_threshold} days): {len(large_gaps)}")
        # print(f"Percentage of large gaps: {100*len(large_gaps)/len(all_gaps):.1f}%")
        # print(f"Most common gap size: {max(set(all_gaps), key=all_gaps.count)} days")
        # print(f"Mean gap size: {np.mean(all_gaps):.2f} days")
        # print(f"Median gap size: {np.median(all_gaps):.1f} days")
        # print(f"Sequential gaps (1 day): {all_gaps.count(1)} ({100*all_gaps.count(1)/len(all_gaps):.1f}%)")
        # print(f"Near-sequential gaps (1-2 days): {sum(1 for gap in all_gaps if gap <= 2)} ({100*sum(1 for gap in all_gaps if gap <= 2)/len(all_gaps):.1f}%)")

        # if large_gaps:
        #     print(f"\nLarge gap statistics:")
        #     print(f"Mean large gap size: {np.mean(large_gaps):.1f} days")
        #     print(f"Max gap size: {max(large_gaps)} days")
        #     print(f"Large gaps by month:")
            
        #     # Analyze which months have the most large gaps
        #     large_gap_months = {month: 0 for month in fall_months}
        #     for exp_name in exp_list:
        #         dates_within_fall = all_confident_dates[exp_name][all_confident_dates[exp_name].dt.month.isin(fall_months)]
                
        #         if len(dates_within_fall) > 0:
        #             dates_sorted = dates_within_fall.sortby(dates_within_fall)
                    
        #             for year in np.unique(dates_sorted.dt.year.values):
        #                 year_dates = dates_sorted[dates_sorted.dt.year == year]
                        
        #                 if len(year_dates) > 1:
        #                     day_diffs = np.diff(year_dates.dt.dayofyear.values)
                            
        #                     for date, gap in zip(year_dates[:-1], day_diffs):
        #                         month = date.dt.month.item()
        #                         if gap >= large_gap_threshold and month in fall_months:
        #                             large_gap_months[month] += 1
            
        #     for month in fall_months:
        #         print(f"  {month_names[month]}: {large_gap_months[month]} large gaps")




        # for exp_name in exp_list:
        #     # select dates within fall months
        #     dates_within_fall = all_confident_dates[exp_name][all_confident_dates[exp_name].dt.month.isin(fall_months)]
        #     print(f"dates_within_fall: {dates_within_fall}")
    
        #     if len(dates_within_fall) > 0:
        #         # check for dates within one year at a time: 
        #         for year in np.unique(dates_within_fall.dt.year.values):
        #             print(f"year: {year}")
        #             dates_in_year = dates_within_fall[dates_within_fall.dt.year == year]
        #             print(f"fall day of year: {dates_in_year.dt.dayofyear.values}")
        #             fall_diffs = np.diff(dates_in_year.dt.dayofyear)
        #             print(f"fall_diffs: {fall_diffs}")
        #             # find dates associated with each value in fall_diffs
        #             ax.stem(dates_in_year[:-1], fall_diffs, markerfmt=' ')
                

        #     # fall_group_lengths = np.diff(np.concatenate(([0], fall_chunks + 1, [len(dates_within_fall)])))
        #     # max_fall_group_length = np.max(fall_group_lengths)
        #     # mean_fall_group_length = np.mean(fall_group_lengths)

        
        # # plot group lenghts as histogram: 
        # # plt.hist(fall_group_lengths, bins=20, color="#d75f27", alpha=0.7)
        # plt.xlabel('Consecutive Days in Fall (Oct-Dec)')
        # plt.ylabel('Count of Days between Confident Predictions')
        # plt.title(f'Temporal Distribution of Selected Samples in Fall | {exp_type} \n Confidence Range: {confidence_level_low}-{confidence_level_high}%')
        # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/temporal_distribution_fall_{exp_type}_{confidence_level_low}to{confidence_level_high}.png', format = 'png', dpi = 250)



           # figure out storage of these variables from all random seed experiments together
                # selected_samples["output1"] = output[selected_indices]
                # selected_samples["target_date1"] = selected_target_dates
                # selected_samples["input_date1"] = selected_input_dates
                # selected_samples["iqr1"] = iqr[selected_indices]
                # selected_samples["crps1"] = crps[selected_indices]


            # # Prepare selected samples to go into dataloader: ['x'], ['y'], ['time'], etc... 
            #     input_testfn = str(config["perlmutter_inputs_dir"]) + str(config["input_data"]) + "_trimmed_" + "test_dat.nc"
            #     input_test_data = open_data_file(input_testfn)
            #     input_test_data_trimmed = input_test_data.sel(time = selected_target_dates)
            #     save_fn = '/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/{exp_name}_conf_monthly_selected_input_test_data.nc'
            #     input_test_data_trimmed.to_netcdf(save_fn)

            #     # NOW RUN SELECTED SAMPLES THROUGH OPPOSING MODEL TYPE -----------------
            #     # Establish selected samples as an inference dataset usign dataloader: 
            #     selected_inf_set = data_loader.CustomData(save_fn, config, which_set = 'testing')

            #     if exp_type in ["OBS(OBS)"]:
            #         opposing_exp_type = "E3SM-short(OBS)"
            #         trained_model2_exp = "exp185" # seed 1 of E3SM-short(E3SM) models
            #     elif exp_type in ["E3SM-short(E3SM)"]:
            #         opposing_exp_type = "OBS(E3SM)"
            #         trained_model2_exp = "exp173" # seed 1 of OBS(OBS) models

            #     ood_config = utils.get_config(str(trained_model2_exp))
            #     device = utils.prepare_device(ood_config["device"])
                
            #     # Load the Model
            #     path = str(ood_config["perlmutter_model_dir"]) + str(trained_model2_exp) + '.pth'

            #     load_model_dict = torch.load(path)

            #     state_dict = load_model_dict["model_state_dict"]
            #     std_mean = load_model_dict["training_std_mean"]

            #     model = TorchModel(
            #         config=ood_config["arch"],
            #         target_mean=std_mean["trainset_target_mean"],
            #         target_std=std_mean["trainset_target_std"],
            #     )
            
            #     model.load_state_dict(state_dict)
            #     model.eval()

            #     with torch.inference_mode():
            #         print(device)
            #         ood_output = model.predict(dataset=selected_inf_set, batch_size=128, device=device)
                
            #     # Save Model Outputs
            #     ood_model_output = str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/' + str(trained_model2_exp) + 'T_' + str(config["expname"]) + '_OOD_network_SHASH_parameters_CONFIDENT_SELECTION.pkl'
            #     analysis_metrics.save_pickle(ood_output, ood_model_output)
            #     # print(ood_output[:20]) # look at a small sample of the output data