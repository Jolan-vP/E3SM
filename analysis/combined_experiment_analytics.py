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
import cmcrameri as crameri
from shash.shash_torch import Shash
import pickle 
import gzip
from model.metric import iqr_basic
from shash.shash_torch import Shash
import torch
import glob
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
    exps = experiments

    plt.figure(figsize=(7, 5))
    plt.gca().invert_xaxis()  # high confidence = low IQR = right side of plot

    color_themes = {
        0: "#3b528b", 
        1: "#019bba", 
        2: "#33c316",
        3: "#B6B309",
        }
    
    color_themes_dark = {
        0: "#2B4A9A", 
        1: "#0F9DB9", 
        2: "#2FA816",
        3: "#8A8800",
        }

    # i = 0
    # for experiment_type, exp_names in exps.items():

    #     for iexp, exp in enumerate(exp_names):

    #         output = str("/pscratch/sd/p/plutzner/E3SM/saved/output/" + str(exp) + "/" + str(exp) + "_success_ratio.pkl")

    #         try:
    #             with open(output, 'rb') as f:
    #                 discard_data = pickle.load(f)
    #         except (pickle.UnpicklingError, EOFError, UnicodeDecodeError):
    #             try:
    #                 # If it fails, try gzip
    #                 with gzip.open(output, 'rb') as f:
    #                     discard_data = pickle.load(f)
    #             except Exception as e:
    #                 raise RuntimeError(f"Failed to load file with both normal and gzip methods: {e}")
            
    #         percentiles = discard_data['percentiles']
    #         avg_success_ratio = discard_data['avg_success_ratio']

    #         if iexp == 0: 
    #             plt.plot(percentiles, avg_success_ratio, color=color_themes[i], alpha = 0.45, label = f"{experiment_type}", linewidth = 2.5)
    #         else: 
    #             plt.plot(percentiles, avg_success_ratio, color=color_themes[i], alpha = 0.45, linewidth = 2.5)
    #         # plt.fill_between(x = [0, 100], y1 = [0.5, 0.5], color = 'grey', alpha=0.03, edgecolor = None)


    #         plt.xlabel('IQR Percentile (% Data Remaining)')
    #         plt.ylabel('Proportion of Samples with Lower Network CRPS')
    #         # plt.ylim(0.5, 0.85)
    #         # plt.xlim(101, 4)
    #         plt.axhline(y=0.5, color='grey', alpha = 0.8, linestyle='--', linewidth = 0.8)
    #         plt.title('Increasing Confidence Success Ratio Discard Plot')
    #         plt.tight_layout()
    #     i += 1
            

    # leg = plt.legend()
    # for lh in leg.legendHandles: 
    #         lh.set_alpha(1)

    # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/combined_SuccessRatio_DiscardPlot_{keyword}_Z500.png', format = 'png',  dpi = 250) 


    ### SUCCESS RATIO MANUALLY CALCULATED: 
    plt.figure(figsize=(7, 5))
    plt.gca().invert_xaxis()

    for i, (experiment_type, exp_names) in enumerate(exps.items()):

        all_success_ratios = []

        for iexp, exp in enumerate(exp_names):

            # open crps 
            crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp}/{exp}_CRPS_network_values.pkl')
            # open climo crps
            climo_crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp}/{exp}_CRPS_climatology_values.pkl')

            # calcualte success ratio by percentile 
            percentiles = np.linspace(100, 0, 21)
            try:
                output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp}/{exp}_network_SHASH_parameters.pkl')
            except FileNotFoundError:
                pattern = f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp}/exp*_{exp}_OOD_INFERENCE_network_SHASH_parameters.pkl'
                matching_files = glob.glob(pattern)
                output = load_pickle(matching_files[0]) if matching_files else None

            # Calculate IQR:
            iqr = iqr_basic(output)
            # Sort by IQR
            iqr_sorted_indices = np.argsort(iqr)
            iqr_sorted = iqr[iqr_sorted_indices]

            avg_success_ratio = []
            for ip, p in enumerate(percentiles):
                # percentage of samples to keep for each round of the loop
                num_to_keep = int(len(iqr_sorted) * p / 100)

                indices = iqr_sorted_indices[:num_to_keep]

                if len(indices) == 0:
                    avg_success_ratio.append(np.nan)
                else:
                    success_ratio = np.sum(crps[indices] < climo_crps[indices]) / len(indices)
                    avg_success_ratio.append(success_ratio)

            all_success_ratios.append(avg_success_ratio)
        
            # Plot individual experiment line with low alpha
            plt.plot(percentiles, avg_success_ratio, color=color_themes[i], alpha=0.3, linewidth=1.2)

        # Calculate mean across all experiments for this type
        mean_success_ratio = np.nanmean(all_success_ratios, axis=0)
        
        # Plot mean line with high alpha and thicker linewidth
        plt.plot(percentiles, mean_success_ratio, color=color_themes[i], alpha=1, 
                label=f"{experiment_type}", linewidth=2)

        plt.xlabel('IQR Percentile (% Data Remaining)')
        plt.ylabel('Proportion of Samples with Lower Network CRPS')
        plt.axhline(y=0.5, color='grey', alpha=0.8, linestyle='--', linewidth=0.8)
        plt.title('Increasing Confidence Success Ratio Discard Plot')
        plt.tight_layout()
        # plt.ylim([0.47, 0.81])

    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/combined_SuccessRatio_DiscardPlot_{keyword}.png', format = 'png',  dpi = 250)
    plt.close()

        

def combined_CRPS_IQR_discard(experiments, keyword = None):
    """
    Discard plot of mean binned CRPS by IQR percentile for all experiment types passed in. 
    
    """

    plt.figure(figsize=(7, 5))
    plt.gca().invert_xaxis()  # high confidence = low IQR = right side of plot

    exps = experiments 

    color_themes = {
        0: "#2b25ed", 
        1: "#9911d3", 
        2: "#e02c2c",
        3: "#f9910a"
    }

    i = 0
    for experiment_type, exp_names in exps.items():

        all_avg_crps = []
        
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

            all_avg_crps.append(crps_dict)

            if iexp == 0:
                if experiment_type in ["OBS(OBS)", "OBS(OBS)sv"]:
                    obs_obs_color = "#0a03cc"
                    plt.axhline(y=mean_climo_crps, color=obs_obs_color, linestyle='--', label = f'OBS Baseline Mean CRPS', linewidth = 2)
                if experiment_type in ["E3SM(E3SM)", "E3SM(E3SM)sv"]: #or experiment_type == "OBS(E3SM)":
                    e3sm_color = "#cc4778"
                    plt.axhline(y=mean_climo_crps, color=e3sm_color, linestyle='--', label = f'E3SM Baseline Mean CRPS', linewidth = 2)
                plt.plot(percentile_dict, crps_dict, alpha = 0.3, linewidth = 1.2, color = color_themes[i])
            else:
                plt.plot(percentile_dict, crps_dict, alpha = 0.3, linewidth = 1.2, color = color_themes[i])

        # Calculate mean across all experiments for this type
        mean_avg_crps = np.mean(all_avg_crps, axis=0)
        plt.plot(percentile_dict, mean_avg_crps, label=f'{experiment_type}', alpha = 1, linewidth = 2, color = color_themes[i])

        plt.xlabel('IQR Percentile (% Data Remaining)')
        plt.ylabel('Average CRPS')
        # plt.ylim(0.32, 0.68)
        # plt.xlim(101, 4)
        plt.title('Increasing Confidence CRPS Discard Plot')
        plt.tight_layout()
        leg = plt.legend()
        for lh in leg.legendHandles: 
            lh.set_alpha(0.8)

        i += 1

    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/CRPS_discard_plot_combined_{keyword}.png', format = 'png',  dpi = 250)



def IQR_distributions_STEP_hist(experiments, keyword = None):
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
                plt.hist(iqr, bins=bin_edges, alpha=0.6, label=f'{experiment_type}', color = color_themes[i], density = True, histtype = 'step')
            else:
                plt.hist(iqr, bins=bin_edges, alpha=0.6, color = color_themes[i], density = True, histtype = 'step')

        i += 1
            
        plt.xlabel('IQR')
        plt.ylabel('Density')
        plt.title('IQR Distribution Across Model Types')
        leg = plt.legend()
        for lh in leg.legendHandles: 
            lh.set_alpha(1)

    plt.savefig('/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/IQR_distribution_combined_STEP_' + str(keyword) + '.png', format = 'png',  dpi = 250)


def IQR_distributions_STACKED_hist(experiments, keyword = None):
    """
    Plot the distribution of IQR values for a variety of experiments on the same plot. 
    """
    plt.figure(figsize=(7, 5))

    exps = experiments 

    color_themes = {
        0: "#0d0887", 
        1: "#7e03a8", 
        2: "#cc4778",
        3: "#f89540", 
        4: "#33c316",
        5: "#019bba", 
    }

    i = 0

    all_models_iqr = [] 
    for i, (experiment_type, exp_names) in enumerate(exps.items()):
        
        all_seeds_iqr = []

        N = np.empty(len(exp_names))

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

            all_seeds_iqr.append(iqr)

            bins = 65
            min_value = min(np.min(iqr), np.min(iqr))
            max_value = max(np.max(iqr), np.max(iqr))
            bin_edges = np.linspace(min_value, max_value, bins)

            N[i] = len(iqr)

        all_models_iqr.append(np.concatenate(all_seeds_iqr))
        
    # histograms of IQR for each phase
    model_types = list(exps.keys())
    colors = [color_themes[i] for i in range(len(model_types))]
    plt.hist(all_models_iqr, bins=bin_edges, alpha=0.8, label=model_types, color = colors, density = True, histtype = 'barstacked', stacked = True)
    i += 1
        
    plt.xlabel('IQR')
    plt.ylabel('Density')
    plt.xlim([0, 0.045])
    plt.title('IQR Distribution Across Model Types')
    leg = plt.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    plt.tight_layout()
    plt.savefig('/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/IQR_distribution_combined_STACKED_' + str(keyword) + '.png', format = 'png',  dpi = 250)


def CRPS_discard_scaled_IQR(experiments, keyword = None):
    """
    Discard plot of binned *scaled* IQR
    IQR scaled by day-of-year IQR to help remove seasonal cycle in uncertainty.
    """
    exps = experiments

    fig, ax = plt.subplots(figsize=(8, 6))

    # color_themes = {
    #     0: "#3b528b", 
    #     1: "#019bba", 
    #     2: "#2cb212",
    #     3: "#B6A509",
    # }
    color_themes = {
        0: "#1a36d8", 
        1: "#a52dea", 
        2: "#e8395c",
        3: "#f68b2e"
    }

    for i, (exp_type, exp_list) in enumerate(exps.items()):

        all_avg_crps = []

        if i in [0, 1, 2, 3]:
            print(f'Processing experiment type: {exp_type}')

            if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                    data_type = "OBS"
            elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                    data_type = "E3SM"
            
            #identify lengths for accurate preallocation: 
            output_preall = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_list[0]}/{exp_list[0]}_network_SHASH_parameters.pkl')

            all_crps = np.empty((len(exp_list), len(output_preall)))
            all_scaled_iqr = np.empty((len(exp_list), len(output_preall)))

            for iexp, exp_name in enumerate(exp_list):
                print(f'  Processing experiment: {exp_name}')
                
                # Load the output and target data for this experiment
                try: 
                    output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')
                except: 
                    pattern = '/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/exp*_{exp_name}_OOD_INFERENCE_network_SHASH_parameters.pkl'
                    matching_files = glob.glob(pattern)
                    output_preall = load_pickle(matching_files[0]) if matching_files else None

                # open crps: 
                crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

                climo_crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_climatology_values.pkl') 
                mean_climo_crps = np.mean(climo_crps)

                # Load testing target data
                if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                    target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                    # Load climatology statistics
                    climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                    target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
        
                elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                    target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                    climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                    target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]

                # Scale Climo CRPS by day of year mean: 
                climo_crps_xr = xr.DataArray(climo_crps, coords=[target['time']], dims=["time"])
                daily_climo_crps = climo_crps_xr.groupby('time.dayofyear').mean('time')
                scaled_climo_crps = climo_crps_xr.groupby('time.dayofyear') / daily_climo_crps
                scaled_climo_crps = scaled_climo_crps.sortby('time')
                mean_climo_crps = np.mean(scaled_climo_crps)

                # Calculate IQR: 
                iqr = iqr_basic(output)
                iqr_xr = xr.DataArray(iqr, coords=[target['time']], dims=["time"])
                # scale iqr by day of year mean:
                daily_iqr = iqr_xr.groupby('time.dayofyear').mean('time')
                scaled_iqr = iqr_xr.groupby('time.dayofyear') / daily_iqr
                #ungroup scaled_iqr: 
                scaled_iqr = scaled_iqr.sortby('time')

                # Scale CRPS by day of year as well: 
                crps_xr = xr.DataArray(crps, coords=[target['time']], dims=["time"])
                # scale crps by day of year mean:
                daily_crps = crps_xr.groupby('time.dayofyear').mean('time')
                scaled_crps = crps_xr.groupby('time.dayofyear') / daily_crps
                scaled_crps = scaled_crps.sortby('time')
                crps = scaled_crps.values
                
                percentiles = np.linspace(100, 0, 21)

                # Sort by IQR
                scaled_iqr_sorted_indices = np.argsort(scaled_iqr.values)
                scaled_iqr_sorted = scaled_iqr.isel(time=scaled_iqr_sorted_indices)

                avg_crps = np.full([len(exp_list), len(percentiles)], np.nan)
                avg_scaled_iqr = []
                sample_index = np.zeros((len(scaled_iqr), len(percentiles)))

                for ip, p in enumerate(percentiles):
                    # percentage of samples to keep for each round of the loop
                    num_to_keep = int(len(scaled_iqr_sorted) * p / 100)

                    indices = scaled_iqr_sorted_indices[:num_to_keep]

                    if len(indices) == 0:
                        avg_crps[iexp, ip] = np.nan
                        avg_scaled_iqr.append(np.nan)
                    else:
                        avg_crps[iexp, ip] = np.mean(crps[indices])
                        avg_scaled_iqr.append(np.mean(scaled_iqr[indices]))
                        sample_index[:len(indices), ip] = indices
          
                ax.plot(percentiles, avg_crps[iexp], alpha = 0.3, linewidth = 1.2, color=color_themes[i])

            # Calculate mean across all experiments for this type
            mean_avg_crps = np.nanmean(avg_crps, axis=0)

            plt.plot(percentiles, mean_avg_crps, label=f'{exp_type}', alpha = 1, linewidth = 2.2, color = color_themes[i])

            if i in [0, 2]: 
                ax.axhline(y=mean_climo_crps, color=color_themes[i], linestyle='--', label=f'CRPS Mean Climatology for {data_type}')

            plt.gca().invert_xaxis()
            ax.set_ylabel('Average Scaled CRPS')
            ax.set_xlabel('Scaled IQR Percentile (% Data Remaining)')
            ax.set_xlim([100, 1])
            plt.tight_layout()
            plt.legend()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/CRPS_discard_scaled_IQR_Z500_{keyword}.png', format='png', dpi=250)

def combined_success_discard_scaled_IQR(experiments, iqr_scaling = True, keyword = None):
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
    exps = experiments

    fig, ax = plt.subplots(figsize = (8, 6))
    plt.gca().invert_xaxis()  # high confidence = low IQR = right side of plot

    color_themes = {
        0: "#3b528b", 
        1: "#019bba", 
        2: "#33c316",
        3: "#B6B309",
        }
    for i, (exp_type, exp_list) in enumerate(exps.items()):

        all_avg_success = []

        if i in [0, 1, 2, 3]:
                print(f'Processing experiment type: {exp_type}')

                if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                        data_type = "OBS"
                elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "OBS(E3SM)sv", "E3SM(E3SM)sv"]:
                        data_type = "E3SM"

                for iexp, exp_name in enumerate(exp_list):
                    print(f'  Processing experiment: {exp_name}')
                    
                    try:
                        output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')
                    except FileNotFoundError:
                        pattern = f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/exp*_{exp_name}_OOD_INFERENCE_network_SHASH_parameters.pkl'
                        matching_files = glob.glob(pattern)
                        output_preall = load_pickle(matching_files[0]) if matching_files else None

                    # open crps: 
                    crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

                    climo_crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_climatology_values.pkl') 
                    mean_climo_crps = np.mean(climo_crps)

                    # Load testing target data
                    if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                        target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                        # Load climatology statistics
                        climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                        target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
            
                    elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                        target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                        climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                        target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]

                    if iqr_scaling:
                        # Scale Climo CRPS by day of year mean: 
                        climo_crps_xr = xr.DataArray(climo_crps, coords=[target['time']], dims=["time"])
                        daily_climo_crps = climo_crps_xr.groupby('time.dayofyear').mean('time')
                        scaled_climo_crps = climo_crps_xr.groupby('time.dayofyear') / daily_climo_crps
                        scaled_climo_crps = scaled_climo_crps.sortby('time')
                        climo_crps = scaled_climo_crps.values
                        mean_climo_crps = np.mean(scaled_climo_crps)

                        # Calculate IQR: 
                        iqr = iqr_basic(output)
                        iqr_xr = xr.DataArray(iqr, coords=[target['time']], dims=["time"])
                        # scale iqr by day of year mean:
                        daily_iqr = iqr_xr.groupby('time.dayofyear').mean('time')
                        scaled_iqr = iqr_xr.groupby('time.dayofyear') / daily_iqr
                        #ungroup scaled_iqr: 
                        scaled_iqr = scaled_iqr.sortby('time')

                        # Scale CRPS by day of year as well: 
                        crps_xr = xr.DataArray(crps, coords=[target['time']], dims=["time"])
                        # scale crps by day of year mean:
                        daily_crps = crps_xr.groupby('time.dayofyear').mean('time')
                        scaled_crps = crps_xr.groupby('time.dayofyear') / daily_crps
                        scaled_crps = scaled_crps.sortby('time')
                        crps = scaled_crps.values

                        label = "Scaled"

                    else: 
                        iqr = iqr_basic(output)
                        label = " " 
                    
                    percentiles = np.linspace(100, 0, 21)

                    # Sort by IQR
                    if iqr_scaling:
                        iqr_sorted_indices = np.argsort(scaled_iqr.values)
                        iqr_sorted = scaled_iqr.isel(time=iqr_sorted_indices)
                        iqr_sorted = iqr_sorted
                    else:
                        iqr_sorted_indices = np.argsort(iqr)
                        iqr_sorted = iqr[iqr_sorted_indices]
                    
                    avg_success_ratio = np.empty([len(exp_list), len(percentiles)])
                    avg_iqr = []
                    sample_index = np.empty((len(iqr), len(percentiles)))

                    for ip, p in enumerate(percentiles):
                        # percentage of samples to keep for each round of the loop
                        num_to_keep = int(len(iqr_sorted) * p / 100)

                        indices = iqr_sorted_indices[:num_to_keep]

                        if len(indices) == 0:
                            avg_success_ratio[iexp, ip] = np.nan
                            avg_iqr.append(np.nan)
                        else:
                            success_ratio = np.sum(crps[indices] < climo_crps[indices]) / len(indices)
                            avg_success_ratio[iexp, ip] = success_ratio
                            sample_index[:len(indices), ip] = indices      

                        # all_avg_success.append(avg_success_ratio)

                    all_avg_success.append(avg_success_ratio[iexp, :].copy())

                    ax.plot(percentiles, avg_success_ratio[iexp], alpha = 0.25, linewidth = 1.2, color=color_themes[i])

                mean_avg_success = np.nanmean(all_avg_success, axis=0)
                ax.plot(percentiles, mean_avg_success, alpha = 0.7, linewidth = 2.2, color = color_themes[i], label = f'{exp_type}')

                if i in [0, 2]: 
                    ax.axhline(y=mean_climo_crps, color=color_themes[i], linestyle='--', label=f'CRPS Mean Climatology for {data_type}')
                
                ax.set_xlabel(f'{label} IQR Percentile (% Data Remaining)')
                ax.set_ylabel(f'Proportion of Samples with Lower {label} Network CRPS')
                ax.set_ylim(0.5, 0.70)
                ax.set_xlim(101, 4)
                ax.set_title('Increasing Confidence Success Ratio Discard Plot')
                plt.tight_layout()
                
        leg = plt.legend()
        for lh in leg.legendHandles: 
                lh.set_alpha(1)

        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/combined_success_discard_{keyword}.png', format = 'png',  dpi = 250) 



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

# def composite_input_maps_variance(experiments, keyword = None):
#     """
#     For E3SM(OBS), select the high variance samples and plot composite input maps for those samples. 
#     Clustering analysis to identify different clusters of input maps. 
#     Frequency analysis for MJO and ENSO phases within those specific samples. 

#     """
#     exps = experiments
#     exp_type_names = list(exps.keys())

#     colormaps = ["BrBG", "RdBu_r", "PuOr_r"]
#     units = ['(mm/day)', '(K)', '(m)']
#     vars = ["Total Precip", "Skin Temp", "Z500"]

#     for iexp_type, (exp_type, exp_list) in enumerate(exps.items()):
#         print(f'Processing experiment type: {exp_type}')

#         if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
#                     data_type = "OBS"
#             elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
#                     data_type = "E3SM"

#         for iexp_name, exp_name in enumerate(exp_list):  # Add this inner loop
#             print(f'  Processing experiment: {exp_name}')
#             config = utils.get_config(exp_name)

            



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
            if exp_type in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                # Load climatology statistics
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
       
            elif exp_type in ["E3SM-short(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "OBS(E3SM)sv", "E3SM(E3SM)sv"]:
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
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/IQR_monthly_analysis_{exp_type}_{confidence_level_low}to{confidence_level_high}_{keyword}.png', format = 'png', dpi = 250)


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
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/monthly_analysis_{exp_type}_{confidence_level_low}to{confidence_level_high}_{keyword}.png', format = 'png', dpi = 250)

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


def variance_OM_analysis(experiments, scale_target = True, scale_IQR = False, keyword = None):
    """
    Epistemic Uncertainty Analysis : 
    - Discard plot of CRPS vs Variance across random seeds for a given experiment type, comparing across experiment types: 
        Experiments contains multiple experiment types
        For each experiment type: 
            - Load outputs from each random seed
            - Make if block for how to calculate variance: 
                If by np.var, calculate np.var across all random seeds for each output sample
            - Create discard plot of CRPS mean (y axis) binned by variance across seeds (x axis) 
            - Overlay all experiment types on one plot for comparison
    """

    exps = experiments

    color_themes = {
        0: "#211bd2", 
        1: "#8b1dcf", 
        2: "#d03232",
        3: "#e88a0f"
    }
    fig1, ax1 = plt.subplots(figsize = (10, 7))
    fig2, ax2 = plt.subplots(figsize = (10, 7))
    fig4, ax4 = plt.subplots(figsize = (10, 7))

    variance_all_model_types = []
    crps_all_model_types = []

    for i, (exp_type, exp_list) in enumerate(exps.items()):
        print(f'Processing experiment type: {exp_type}')

        if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                data_type = "OBS"
        elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                data_type = "E3SM"
        
        #identify lengths for accurate preallocation: 
        try:
            output_preall = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_list[0]}/{exp_list[0]}_network_SHASH_parameters.pkl')
        except FileNotFoundError:
            pattern = f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_list[0]}/exp*_{exp_list[0]}_OOD_INFERENCE_network_SHASH_parameters.pkl'
            matching_files = glob.glob(pattern)
            output_preall = load_pickle(matching_files[0]) if matching_files else None

        all_crps = np.empty((len(exp_list), len(output_preall)))
        all_mean_shash = np.empty((len(exp_list), len(output_preall)))

        for iexp, exp_name in enumerate(exp_list):
            print(f'  Processing experiment: {exp_name}')

            config = utils.get_config(exp_name)

            if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                # Load climatology statistics
                # climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                # target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
    
            elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "OBS(E3SM)sv", "E3SM(E3SM)sv"]:
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                # climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                # target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]

            # Load the output and target data for this experiment
            try:
                output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')
            except FileNotFoundError:
                pattern = f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/exp*_{exp_name}_OOD_INFERENCE_network_SHASH_parameters.pkl'
                matching_files = glob.glob(pattern)
                output = load_pickle(matching_files[0]) if matching_files else None

            climo_crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_climatology_values.pkl')
            mean_climo_crps = np.mean(climo_crps)

            # DETERMINISTIC CALCULATION METHOD: Mean of Shash
            output_SHASH = Shash(output)
            network_mean_tensor = output_SHASH.mean()

            # store mean shash values as numpy values: 
            all_mean_shash[iexp] = network_mean_tensor.numpy()

            # open crps: 
            crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

            if scale_IQR == True:
                print("Scaling CRPS because IQR is also scaled")
                # Scale Climo CRPS by day of year mean: 
                climo_crps_xr = xr.DataArray(climo_crps, coords=[target['time']], dims=["time"])
                daily_climo_crps = climo_crps_xr.groupby('time.dayofyear').mean('time')
                scaled_climo_crps = climo_crps_xr.groupby('time.dayofyear') / daily_climo_crps
                scaled_climo_crps = scaled_climo_crps.sortby('time')
                mean_climo_crps = np.mean(scaled_climo_crps)

                # Scale CRPS by day of year as well: 
                crps_xr = xr.DataArray(crps, coords=[target['time']], dims=["time"])
                # scale crps by day of year mean:
                daily_crps = crps_xr.groupby('time.dayofyear').mean('time')
                scaled_crps = crps_xr.groupby('time.dayofyear') / daily_crps
                scaled_crps = scaled_crps.sortby('time')
                crps = scaled_crps.values
            elif scale_IQR == False:
                print("Not scaling IQR or CRPS values")
                pass

            all_crps[iexp] = crps

        crps_all_model_types.append(all_crps)

        print(f"all crps shape: {all_crps.shape}, all mean shash shape: {all_mean_shash.shape}")
        # Calculate variance across random seeds for each sample
        variance_across_seeds = np.var(all_mean_shash, axis=0)
        variance_all_model_types.append(variance_across_seeds)

        # FIGURE: Variance Analysis of E3SM(OBS) : ---------------
        if exp_type in ["E3SM(OBS)", "E3SM(OBS)sv", "E3SM(E3SM)", "E3SM(E3SM)sv", "OBS(OBS)", "OBS(OBS)sv", "OBS(E3SM)", "OBS(E3SM)sv"]: 
            config = utils.get_config(exp_list[0])
            
            # Analyze the high variance samples in E3SM(OBS)
            # Select high variance samples based on threshold:
            var_lim = 20 # 20%
            high_variance_indices = np.argsort(variance_across_seeds)[-int(0.2 * variance_across_seeds.shape[0]):]
            print(f"number of high variance indices: {len(high_variance_indices)}")
            non_high_variance_indices = np.unique(np.setdiff1d(np.arange(variance_across_seeds.shape[0]), high_variance_indices))
            # how many samples is 20% of the data: 
            sample_size_lim = 0.2 * variance_across_seeds.shape[0]
            # find low variance indices corresponding to the lowest 20% of variance values:
            low_variance_indices = np.argsort(variance_across_seeds)[:int(sample_size_lim)]
            print(f"number of low variance indices: {len(low_variance_indices)}")

            all_input_maps_high_var = []
            all_input_maps_non_high_var = []
            all_input_maps_low_var = []

            # Open Target: 
            # Load testing target data
            if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                data_type = "OBS"
                data_vars = ["tp", "skt", "z"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
                target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]

                if exp_type in ["E3SM(OBS)sv", "OBS(OBS)sv"]:
                    # scale target by day of year variance: 
                    daily_target_grouped_var = target.groupby('time.dayofyear').var('time')
                    scaled_target = target.groupby('time.dayofyear') / daily_target_grouped_var
                    scaled_target = scaled_target.sortby('time')
                    target = scaled_target

            elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                data_type = "E3SM"
                data_vars = ["PRECT", "TS", "Z500"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]
                target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]

                if exp_type in ["E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                    # scale target by day of year variance:
                    daily_target_grouped_var = target.groupby('time.dayofyear').var('time')
                    scaled_target = target.groupby('time.dayofyear') / daily_target_grouped_var
                    scaled_target = scaled_target.sortby('time')
                    target = scaled_target

            high_variance_dates = target.time[high_variance_indices]
            print(f"length of high var dates: {len(high_variance_dates)}")
            non_high_var_dates = target.time[non_high_variance_indices]
            print(f"length of non high var dates: {len(non_high_var_dates)}")
            low_variance_dates = target.time[low_variance_indices]

            high_var_input_maps = input_maps.sel(time = high_variance_dates)
            non_high_var_input_maps = input_maps.sel(time = non_high_var_dates)
            low_var_input_maps = input_maps.sel(time = low_variance_dates)

            # Composite Plot of Input Maps: SIMPLE MEAN
            all_input_maps_high_var.append(high_var_input_maps)
            all_input_maps_non_high_var.append(non_high_var_input_maps)
            all_input_maps_low_var.append(low_var_input_maps)

            # Analyze ENSO in HIGH VARIANCE Selected Dates: 
            enso_dates_pkl = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_daily_enso_timestamps.pkl')
            enso_baseline_frequencies = analysis.analysis_metrics.baseline_enso_frequencies(data_type)

            # check which key (category) each of the target dates falls into, and create a list with either "EN", "LN" or "N"
            enso_phase = []
            for date in high_variance_dates.values:
                if date in enso_dates_pkl['El Nino']:
                    enso_phase.append("EN")
                elif date in enso_dates_pkl['La Nina']:
                    enso_phase.append("LN")
                else:
                    enso_phase.append("N")

            test_dates = target.time

            if data_type == "OBS":
                non_leap_EN_dates = [
                    date for date in enso_dates_pkl['El Nino']
                    if not (date.astype('datetime64[M]').astype(int) % 12 + 1 == 2 and 
                            (date.astype('datetime64[D]') - date.astype('datetime64[M]')).astype(int) + 1 == 29)]
                non_leap_LN_dates = [
                    date for date in enso_dates_pkl['La Nina']
                    if not (date.astype('datetime64[M]').astype(int) % 12 + 1 == 2 and 
                            (date.astype('datetime64[D]') - date.astype('datetime64[M]')).astype(int) + 1 == 29)]
                non_leap_N_dates = [
                    date for date in enso_dates_pkl['Neutral']
                    if not (date.astype('datetime64[M]').astype(int) % 12 + 1 == 2 and 
                            (date.astype('datetime64[D]') - date.astype('datetime64[M]')).astype(int) + 1 == 29)]
            elif data_type == "E3SM":
                non_leap_EN_dates = enso_dates_pkl['El Nino']
                non_leap_LN_dates = enso_dates_pkl['La Nina']
                non_leap_N_dates = enso_dates_pkl['Neutral']


            # dates must fall within min and max of test_dates: 
            non_leap_EN_dates = [date for date in non_leap_EN_dates if date >= test_dates.min() and date <= test_dates.max()]
            non_leap_LN_dates = [date for date in non_leap_LN_dates if date >= test_dates.min() and date <= test_dates.max()]
            non_leap_N_dates = [date for date in non_leap_N_dates if date >= test_dates.min() and date <= test_dates.max()]

            en_dates_in_test = test_dates.sel(time=non_leap_EN_dates)
            ln_dates_in_test = test_dates.sel(time=non_leap_LN_dates)
            n_dates_in_test = test_dates.sel(time=non_leap_N_dates)

            # print(f"El Nino proportion in high variance dates relative to all El Nino dates in test set: {enso_phase.count('EN')/len(en_dates_in_test)*100:.1f}% ({enso_phase.count('EN')} out of {len(en_dates_in_test)})")
            # print(f"La Nina proportion in high variance dates relative to all La Nina dates in test set: {enso_phase.count('LN')/len(ln_dates_in_test)*100:.1f}% ({enso_phase.count('LN')} out of {len(ln_dates_in_test)})")
            # print(f"Neutral proportion in high variance dates relative to all Neutral dates in test set: {enso_phase.count('N')/len(n_dates_in_test)*100:.1f}% ({enso_phase.count('N')} out of {len(n_dates_in_test)})")

            # ENSO Figure 1: Relative proportion of high var samples 
            fig, ax_enso = plt.subplots(figsize=(8, 6))
            bin_edges = np.array([-0.5, 0.5, 1.5, 2.5])
            bin_centers = np.array([0, 1, 2])
            bar_width = 0.4

            num_LN = enso_phase.count('LN')
            num_EN = enso_phase.count('EN')
            num_N = enso_phase.count('N')
            sum_total_phases = num_EN + num_LN + num_N
            enso_phase_dist = [num_EN / sum_total_phases, 
                            num_LN / sum_total_phases, 
                            num_N / sum_total_phases]

            bars = ax_enso.bar(bin_centers, enso_phase_dist, width=bar_width, color="#fbc13a", alpha=0.7, edgecolor='black')

            # Reference lines for each ENSO phase
            enso_phases = ['El Nino', 'La Nina', 'Neutral']
            for k, phase in enumerate(enso_phases):
                freq = enso_baseline_frequencies[phase]
                # Draw a horizontal line across the width of the bar for phase i
                ax_enso.hlines(y=freq, xmin=k - bar_width/2, xmax=k + bar_width/2, 
                        color="#3C3B3B", linewidth=2, linestyle='-', 
                        label='Reference' if k==0 else None)

            ax_enso.set_ylim([0, max(max(enso_phase_dist), max(enso_baseline_frequencies.values())) * 1.15])
            ax_enso.set_xticks(bin_centers)
            ax_enso.set_xticklabels(['El Nino', 'La Nina', 'Neutral'])
            ax_enso.set_xlabel('ENSO Phase')
            ax_enso.set_ylabel('Density')
            ax_enso.set_title(f'ENSO Phase Distribution | E3SM(OBS) High Variance over Means Samples {keyword}')

            # Legend with only one entry for reference lines
            handles, labels = ax_enso.get_legend_handles_labels()
            if 'Reference' in labels:
                idx = labels.index('Reference')
                ax_enso.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
            else:
                ax_enso.legend()
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_phase_distribution_E3SM-OBS_high_var_OM_{keyword}.png', format='png', dpi=250)
            plt.close()

            # ENSO Figure 2:  Difference plot between high var EN and all other EN:
            fig, ax_enso2 = plt.subplots(figsize=(8, 6))


            ######## MJO @ VERIFICATION DAY ########
            phase_timestamps = analysis_metrics.mjo_timestamps(data_type, config)

            if exp_type in ["E3SM(E3SM)", "E3SM(E3SM)sv", "OBS(E3SM)", "OBS(E3SM)sv"]:
                # Check the calendar type of phase_timestamps
                phase_sample = phase_timestamps['time'].values[0]
                high_var_sample = high_variance_dates.values[0]

                print(f"Phase timestamps type: {type(phase_sample)}")
                print(f"High variance dates type: {type(high_var_sample)}")

                # Convert phase_timestamps to cftime.DatetimeNoLeap if needed
                if isinstance(high_var_sample, cftime.DatetimeNoLeap) and not isinstance(phase_sample, cftime.DatetimeNoLeap):
                    # Phase timestamps needs to be converted to NoLeap calendar
                    # Skip Feb 29 dates since they don't exist in NoLeap calendar
                    phase_time_noleap = []
                    phase_indices_to_keep = []
                    
                    for idx, t in enumerate(phase_timestamps['time'].values):
                        ts = pd.Timestamp(t)
                        # Skip February 29th
                        if ts.month == 2 and ts.day == 29:
                            continue
                        phase_time_noleap.append(
                            cftime.DatetimeNoLeap(ts.year, ts.month, ts.day)
                        )
                        phase_indices_to_keep.append(idx)
                    
                    # Filter the dataset to exclude Feb 29 dates
                    phase_timestamps = phase_timestamps.isel(time=phase_indices_to_keep)
                    
                    # Assign the converted time coordinate
                    phase_timestamps = phase_timestamps.assign_coords(time=xr.CFTimeIndex(phase_time_noleap))
                    
                    # Now exact selection will work
                    selected_mjo_phases = phase_timestamps.sel(time=high_variance_dates.values)
                    
                elif isinstance(phase_sample, cftime.DatetimeNoLeap) and not isinstance(high_var_sample, cftime.DatetimeNoLeap):
                    # High variance dates needs to be converted to NoLeap calendar
                    high_var_noleap = []
                    for t in high_variance_dates.values:
                        ts = pd.Timestamp(t)
                        if ts.month == 2 and ts.day == 29:
                            continue
                        high_var_noleap.append(
                            cftime.DatetimeNoLeap(ts.year, ts.month, ts.day)
                        )
                    selected_mjo_phases = phase_timestamps.sel(time=high_var_noleap)
                    
                else:
                    # Both are same type, direct selection
                    selected_mjo_phases = phase_timestamps.sel(time=high_variance_dates.values)
            else:
                # For non-E3SM experiments, select directly
                selected_mjo_phases = phase_timestamps.sel(time=high_variance_dates)

            
            # missing_dates = high_variance_dates[~high_variance_dates.isin(phase_timestamps['time'])]
            # print(f"Number of missing dates: {len(missing_dates)}")
            # if len(missing_dates) > 0:
            #     print(f"First few missing dates: {missing_dates[:5].values}")
            #     print(f"Last few missing dates: {missing_dates[-5:].values}")

            # # Check time resolution
            # print(f"High variance dates sample: {high_variance_dates[:3].values}")
            # print(f"Phase timestamps sample: {phase_timestamps['time'][:3].values}")
            # print(f'high variance dates range: {high_variance_dates.min().values} to {high_variance_dates.max().values}')
            # print(f'high variance dates type: {type(high_variance_dates)}')
            # print(f"phase timesteps range and type: {phase_timestamps.time.min().values} to {phase_timestamps.time.max().values}, type: {type(phase_timestamps.time)}")
           

            # if "OBS" in data_type:
            #     mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_phase_frequencies_ERA5_1940_2023.pkl')
            # elif "E3SM" in data_type:
            #     mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_frequencies_E3SM_1850_2014.pkl')

            # counts = np.bincount(selected_mjo_phases["phase"], minlength=9)
            # total = counts.sum()
            # densities = counts / total

            # phases = np.arange(0, 9)
            # phase_labels = [str(i) for i in phases]

            # fig, ax_mjo = plt.subplots(figsize=(8, 6))
            # bar_width = 0.8

            # # Bar plot for density
            # bars = ax_mjo.bar(phases, densities, width=bar_width, color="#9c4781", alpha=0.7, edgecolor='black', label='Selected Samples')

            # # Reference lines for each phase
            # for i, freq in enumerate(mjo_ref_frequencies_all_data):
            #     # Draw a horizontal line across the width of the bar for phase i
            #     ax_mjo.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, color="#3C3B3B", linewidth=2, linestyle='-', label='Reference' if i==0 else None)

            # ax_mjo.set_xticks(phases)
            # ax_mjo.set_xticklabels(phase_labels)
            # ax_mjo.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
            # ax_mjo.set_xlabel('MJO Phase')
            # ax_mjo.set_ylabel('Density')
            # ax_mjo.set_title('MJO Phase Distribution for High Variance E3SM(OBS) Samples | {keyword}')
            # handles, labels = ax_mjo.get_legend_handles_labels()
            # # Only show one legend entry for the reference lines
            # if 'Reference' in labels:
            #     idx = labels.index('Reference')
            #     ax_mjo.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
            # else:
            #     ax_mjo.legend()
            # plt.tight_layout()
            # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_E3SM-OBS_high_var_{keyword}.png', format='png', dpi=250)
            # plt.close()

            ######## MJO @ INITIALIZATION DAY ########

            lagtime_days = config['databuilder']['lagtime']

            if isinstance(high_variance_dates.values[0], cftime.DatetimeNoLeap):
                # Use datetime.timedelta for cftime
                high_variance_input_dates = high_variance_dates - datetime.timedelta(days=lagtime_days)
            else:
                # Use np.timedelta64 for numpy datetime64
                high_variance_input_dates = high_variance_dates - np.timedelta64(lagtime_days, 'D')
                
            # high_variance_input_dates = high_variance_dates - np.timedelta64(config['databuilder']['lagtime'])
            high_var_dates = high_variance_input_dates.astype('datetime64[D]')
            phase_dates = phase_timestamps.time.astype('datetime64[D]')

            # Find valid dates that exist in both
            valid_mask = np.isin(high_var_dates, phase_dates)
            valid_high_var_dates = high_var_dates[valid_mask]

            print(f"Using {valid_mask.sum()} of {len(high_var_dates)} high variance dates")

            # Select only valid dates
            input_mjo_phases = phase_timestamps.sel(time=valid_high_var_dates)

            if "OBS" in data_type:
                mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_phase_frequencies_ERA5_1940_2023.pkl')
            elif "E3SM" in data_type:
                mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_frequencies_E3SM_1850_2014.pkl')

            counts = np.bincount(input_mjo_phases["phase"], minlength=9)
            total = counts.sum()
            densities = counts / total

            phases = np.arange(0, 9)
            phase_labels = [str(i) for i in phases]

            fig, ax_mjo = plt.subplots(figsize=(8, 6))
            bar_width = 0.8

            # Bar plot for density
            bars = ax_mjo.bar(phases, densities, width=bar_width, color="#4a479c", alpha=0.7, edgecolor='black', label='Selected Samples')

            # Reference lines for each phase
            for k, freq in enumerate(mjo_ref_frequencies_all_data):
                # Draw a horizontal line across the width of the bar for phase k
                ax_mjo.hlines(y=freq, xmin=k - bar_width/2, xmax=k + bar_width/2, color="#3C3B3B", linewidth=2, linestyle='-', label='Reference' if k==0 else None)

            ax_mjo.set_xticks(phases)
            ax_mjo.set_xticklabels(phase_labels)
            ax_mjo.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
            ax_mjo.set_xlabel('MJO Phase')
            ax_mjo.set_ylabel('Density')
            ax_mjo.set_title(f'MJO Phase Distribution for High Variance E3SM(OBS) Samples \n Initialization Day {keyword}')
            handles, labels = ax_mjo.get_legend_handles_labels()
            # Only show one legend entry for the reference lines
            if 'Reference' in labels:
                idx = labels.index('Reference')
                ax_mjo.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
            else:
                ax_mjo.legend()
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_E3SM-OBS_high_var_OM_INIT_day_{keyword}.png', format='png', dpi=250)
            plt.close()

            ## Analyze ENSO & MJO(initialization day) in LOW Variance Dates: ------

            enso_phase = []
            for date in low_variance_dates.values: # LOW variance dates
                if date in enso_dates_pkl['El Nino']:
                    enso_phase.append("EN")
                elif date in enso_dates_pkl['La Nina']:
                    enso_phase.append("LN")
                else:
                    enso_phase.append("N")

            print(f"El Nino proportion in high variance over Means dates relative to all El Nino dates in test set: {enso_phase.count('EN')/len(en_dates_in_test)*100:.1f}% ({enso_phase.count('EN')} out of {len(en_dates_in_test)})")
            print(f"La Nina proportion in high variance over Means dates relative to all La Nina dates in test set: {enso_phase.count('LN')/len(ln_dates_in_test)*100:.1f}% ({enso_phase.count('LN')} out of {len(ln_dates_in_test)})")
            print(f"Neutral proportion in high variance over Means dates relative to all Neutral dates in test set: {enso_phase.count('N')/len(n_dates_in_test)*100:.1f}% ({enso_phase.count('N')} out of {len(n_dates_in_test)})")

            # ENSO Figure 1: Relative proportion of high var samples 
            fig, ax_enso = plt.subplots(figsize=(8, 6))
            bin_edges = np.array([-0.5, 0.5, 1.5, 2.5])
            bin_centers = np.array([0, 1, 2])
            bar_width = 0.4

            num_LN = enso_phase.count('LN')
            num_EN = enso_phase.count('EN')
            num_N = enso_phase.count('N')
            sum_total_phases = num_EN + num_LN + num_N
            enso_phase_dist = [num_EN / sum_total_phases, 
                            num_LN / sum_total_phases, 
                            num_N / sum_total_phases]

            bars = ax_enso.bar(bin_centers, enso_phase_dist, width=bar_width, color="#fbc13a", alpha=0.7, edgecolor='black')

            # Reference lines for each ENSO phase
            enso_phases = ['El Nino', 'La Nina', 'Neutral']
            for j, phase in enumerate(enso_phases):
                freq = enso_baseline_frequencies[phase]
                # Draw a horizontal line across the width of the bar for phase j
                ax_enso.hlines(y=freq, xmin=j - bar_width/2, xmax=j + bar_width/2, 
                        color="#3C3B3B", linewidth=2, linestyle='-', 
                        label='Reference' if j==0 else None)

            ax_enso.set_ylim([0, max(max(enso_phase_dist), max(enso_baseline_frequencies.values())) * 1.15])
            ax_enso.set_xticks(bin_centers)
            ax_enso.set_xticklabels(['El Nino', 'La Nina', 'Neutral'])
            ax_enso.set_xlabel('ENSO Phase')
            ax_enso.set_ylabel('Density')
            ax_enso.set_title(f'ENSO Phase Distribution | E3SM(OBS) Low Variance over Means Samples | {keyword}')

            # Legend with only one entry for reference lines
            handles, labels = ax_enso.get_legend_handles_labels()
            if 'Reference' in labels:
                idx = labels.index('Reference')
                ax_enso.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
            else:
                ax_enso.legend()
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_phase_distribution_E3SM-OBS_low_var_OM_{keyword}.png', format='png', dpi=250)
            plt.close()

            ######## MJO @ INITIALIZATION DAY ########

            low_variance_input_dates = low_variance_dates - np.timedelta64(config['databuilder']['lagtime'])
            low_var_dates = low_variance_input_dates.astype('datetime64[D]')
            phase_dates = phase_timestamps.time.astype('datetime64[D]')

            # Find valid dates that exist in both
            valid_mask = np.isin(low_var_dates, phase_dates)
            valid_low_var_dates = low_var_dates[valid_mask]

            print(f"Using {valid_mask.sum()} of {len(low_var_dates)} low variance dates")

            # Select only valid dates
            input_mjo_phases = phase_timestamps.sel(time=valid_low_var_dates)

            if "OBS" in data_type:
                mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_phase_frequencies_ERA5_1940_2023.pkl')
            elif "E3SM" in data_type:
                mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_frequencies_E3SM_1850_2014.pkl')

            counts = np.bincount(input_mjo_phases["phase"], minlength=9)
            total = counts.sum()
            densities = counts / total

            phases = np.arange(0, 9)
            phase_labels = [str(i) for i in phases]

            fig, ax_mjo = plt.subplots(figsize=(8, 6))
            bar_width = 0.8

            # Bar plot for density
            bars = ax_mjo.bar(phases, densities, width=bar_width, color="#4a479c", alpha=0.7, edgecolor='black', label='Selected Samples')

            # Reference lines for each phase
            for j, freq in enumerate(mjo_ref_frequencies_all_data):
                # Draw a horizontal line across the width of the bar for phase j
                ax_mjo.hlines(y=freq, xmin=j - bar_width/2, xmax=j + bar_width/2, color="#3C3B3B", linewidth=2, linestyle='-', label='Reference' if j==0 else None)

            ax_mjo.set_xticks(phases)
            ax_mjo.set_xticklabels(phase_labels)
            ax_mjo.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
            ax_mjo.set_xlabel('MJO Phase')
            ax_mjo.set_ylabel('Density')
            ax_mjo.set_title(f'MJO Phase Distribution for Low Variance over Means E3SM(OBS) Samples \n Initialization Day | {keyword}')
            handles, labels = ax_mjo.get_legend_handles_labels()
            # Only show one legend entry for the reference lines
            if 'Reference' in labels:
                idx = labels.index('Reference')
                ax_mjo.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
            else:
                ax_mjo.legend()
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_E3SM-OBS_low_var_OM_INIT_day_{keyword}.png', format='png', dpi=250)
            plt.close()


            # Composite difference plot: El Nino High variance to El Nino all other samples
            # en_dates_in_test and high_var dates overlap: 
            overlapping_en_dates = np.array([date for date in en_dates_in_test.values if date in high_variance_dates.values])
            # en_dates_in_test and non_high_var_dates overlap:
            overlapping_non_high_en_dates = np.array([date for date in en_dates_in_test.values if date in non_high_var_dates.values])

            high_var_en_input_maps = input_maps.sel(time=overlapping_en_dates)
            high_var_en_input_map_mean = high_var_en_input_maps.mean(dim='time')
            non_high_var_en_input_maps = input_maps.sel(time=overlapping_non_high_en_dates)
            non_high_var_en_input_map_mean = non_high_var_en_input_maps.mean(dim='time')

            en_diff_input_map = high_var_en_input_map_mean - non_high_var_en_input_map_mean

            fig, ax = plt.subplots(3, 3, figsize=(15, 11), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
            cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
            vmin_list = np.zeros(3)
            vmax_list = np.zeros(3)

            # Plot three rows of maps: (1st row) High variance el ninos (2nd row) non-high variance el ninos (3rd row) difference
            for k in range(3):
                vmin_list = [-1, -1.5, -1.5]
                vmax_list = [1, 1.5, 1.5]

                # High Variance El Nino
                im1 = ax[0, k].pcolormesh(
                    high_var_en_input_map_mean['lon'],
                    high_var_en_input_map_mean['lat'],
                    high_var_en_input_map_mean[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[0, k].coastlines()
                ax[0, k].set_title(f'High Variance over Means El Nino Input Map: {variable_names[k]} \n N = {len(overlapping_en_dates)}')
                plt.colorbar(im1, ax=ax[0, k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')

                # Non-High Variance El Nino
                im2 = ax[1, k].pcolormesh(
                    non_high_var_en_input_map_mean['lon'],
                    non_high_var_en_input_map_mean['lat'],
                    non_high_var_en_input_map_mean[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[1, k].coastlines()
                ax[1, k].set_title(f'Non-High Variance over Means El Nino Input Map: {variable_names[k]} \n N = {len(overlapping_non_high_en_dates)}')
                plt.colorbar(im2, ax=ax[1, k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')

                # Difference Map
                im3 = ax[2, k].pcolormesh(
                    en_diff_input_map['lon'],
                    en_diff_input_map['lat'],
                    en_diff_input_map[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[2, k].coastlines()
                ax[2, k].set_title(f'Difference Map (High Var - Non-High Var) OM: {variable_names[k]}')
                plt.colorbar(im3, ax=ax[2, k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_high_var_vs_non_high_var_OM_input_maps_E3SM-OBS_{keyword}.png', format='png', dpi=250)
            plt.close()
            
            
            # Composite Plot of Input Maps: SIMPLE MEAN
            combined_input_maps = xr.concat(all_input_maps_high_var, dim='time')
            mean_input_map = combined_input_maps.mean(dim='time')

            fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
            cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
            vmin_list = np.zeros(3)
            vmax_list = np.zeros(3)

            for k in range(3):
                vmin_list = [-0.7, -1, -1]
                vmax_list = [0.7, 1, 1]

                im = ax[k].pcolormesh(
                    mean_input_map['lon'],
                    mean_input_map['lat'],
                    mean_input_map[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[k].coastlines()
                ax[k].set_title(f'Mean Input Map: {variable_names[k]} | High Variance over Means E3SM(OBS) > {var_lim}')
                plt.colorbar(im, ax=ax[k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/high_var_OM_composite_input_maps_E3SM-OBS_{keyword}.png', format='png', dpi=250)
            plt.close()

            #  PLOT: LOW Variance Input Map Signals: 
            combined_low_var_input_maps = xr.concat(all_input_maps_low_var, dim='time')
            mean_low_var_input_map = combined_low_var_input_maps.mean(dim='time')

            fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
            cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
            vmin_list = np.zeros(3)
            vmax_list = np.zeros(3)

            for k in range(3):
                vmin_list = [-0.7, -1, -1]
                vmax_list = [0.7, 1, 1]

                im = ax[k].pcolormesh(
                    mean_low_var_input_map['lon'],
                    mean_low_var_input_map['lat'],
                    mean_low_var_input_map[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[k].coastlines()
                ax[k].set_title(f'Mean Low Variance Input Map: {variable_names[k]} | E3SM(OBS) \n Bottom 20% Variance over Means')
                plt.colorbar(im, ax=ax[k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/low_var_OM_composite_maps_E3SM-OBS_{keyword}.png', format='png', dpi=250)
            plt.close()

            # Difference Plot : High Var Input Maps vs Low Var Input Maps! 
            combined_high_var_input_maps = xr.concat(all_input_maps_high_var, dim='time')
            combined_non_high_var_input_maps = xr.concat(all_input_maps_non_high_var, dim='time')
            mean_high_var_input_map = combined_high_var_input_maps.mean(dim='time')
            mean_non_high_var_input_map = combined_non_high_var_input_maps.mean(dim='time')
            mean_diff_input_map = mean_high_var_input_map - mean_non_high_var_input_map

            fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
            cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
            vmin_list = np.zeros(3)
            vmax_list = np.zeros(3)

            for k in range(3):
                vmin_list = [-0.7, -1, -1]
                vmax_list = [0.7, 1, 1]

                im = ax[k].pcolormesh(
                    mean_input_map['lon'],
                    mean_input_map['lat'],
                    mean_diff_input_map[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[k].coastlines()
                ax[k].set_title(f'Mean Difference Input Map: {variable_names[k]} | E3SM(OBS) \n Variance > {var_lim} - Variance <= {var_lim} over Means')
                plt.colorbar(im, ax=ax[k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/high_var_OM_composite_DIFF_maps_E3SM-OBS_{keyword}.png', format='png', dpi=250)
            plt.close()


        # # FIGURE : CRPS vs Variance as DISCARD Plot: --------------------- SOMETHING IS WRONG!!! 
        # percentiles = np.linspace(100, 0, 21)
        # var_sorted_indices = np.argsort(variance_across_seeds)  

        # plt.figure(fig4.number)
        # for iexp, exp_name in enumerate(exp_list):

        #     avg_crps = np.empty([len(exp_list), len(percentiles)])
        #     avg_variance = []
        #     sample_index = np.zeros((len(variance_across_seeds), len(percentiles)))

        #     # Sort by Variance
        #     var_sorted = variance_across_seeds[var_sorted_indices]

        #     for ip, p in enumerate(percentiles):
        #         # percentage of samples to keep for each round of the loop
        #         num_to_keep = int(len(var_sorted) * p / 100)
                
        #         indices = var_sorted_indices[:num_to_keep]

        #         if len(indices) == 0:
        #             avg_crps[iexp, ip] = np.nan
        #             avg_variance.append(np.nan)
        #         else:
        #             avg_crps[iexp, ip] = np.mean(all_crps[iexp, indices])
        #             avg_variance.append(np.mean(variance_across_seeds[indices]))
        #             sample_index[:len(indices), ip] = indices

        #     ax4.plot(percentiles, avg_crps[iexp], alpha = 0.3, linewidth = 1.2, color=color_themes[i])

        # mean_crps_line = np.mean(avg_crps, axis = 0)

        # ax4.plot(percentiles, mean_crps_line, alpha = 0.6, linewidth = 2, color = color_themes[i], label = f'{exp_type}')

        # if i in [0, 2]: 
        #     ax4.axhline(y=mean_climo_crps, color=color_themes[i], linestyle='--', label=f'CRPS Mean Climatology for {data_type}')

        # plt.gca().invert_xaxis()
        # ax4.set_ylabel('Average CRPS')
        # ax4.set_xlabel('Model Variance Percentile (% Data Remaining)')
        # ax4.set_xlim([100, 1])
        # plt.tight_layout()
        # plt.legend()
   
        # FIGURE : Standard CRPS vs Variance Binned Plot -----------------
        # Bin the variance values
        n_bins = 21  # You can adjust the number of bins
        bin_edges = np.linspace(np.min(variance_across_seeds), np.max(variance_across_seeds), n_bins + 1)
        bin_indices = np.digitize(variance_across_seeds, bin_edges) - 1  # bins are 0-indexed

        mean_crps_per_bin = np.full((len(exp_list), n_bins), np.nan)  # Initialize with NaN
        bin_centers = np.array([(bin_edges[b] + bin_edges[b+1]) / 2 for b in range(n_bins)])

        all_valid_mean_crps = []

        plt.figure(fig1.number) 

        for exp in range(len(exp_list)):
            for b in range(n_bins):
                in_bin = bin_indices == b
                if np.any(in_bin):
                    mean_crps = np.mean(all_crps[exp, in_bin]) 
                    mean_crps_per_bin[exp, b] = mean_crps

            # Create mask to filter out empty bins (NaN values)
            valid_mask = ~np.isnan(mean_crps_per_bin[exp])
            valid_bin_centers = bin_centers[valid_mask]
            valid_mean_crps = mean_crps_per_bin[exp, valid_mask]

            all_valid_mean_crps.append(valid_mean_crps)
            
            if exp == 0:
                if i in [0, 2]: 
                    ax1.axhline(y=mean_climo_crps, color=color_themes[i], linestyle='--', label=f'CRPS Mean Climatology for {data_type}')
            else:
                ax1.plot(valid_bin_centers, valid_mean_crps, alpha=0.25, linewidth=1.1, 
                        color=color_themes[i])
        
        mean_line_crps = np.nanmean(all_valid_mean_crps, axis=0)
        ax1.plot(valid_bin_centers, mean_line_crps, alpha=0.7, linewidth=2.2, 
                color=color_themes[i], label=f'{exp_type}')

        plt.figure(fig1.number)
        ax1.set_ylabel('CRPS')
        ax1.set_xlabel('Variance')
        ax1.set_xlim([0, 0.03])
        ax1.set_ylim([0.35, 1.5])
        plt.title(f'CRPS vs Variance OM Plot')
        plt.legend()
        plt.tight_layout()

    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/CRPS_vs_variance_OM_multiple_exps_{keyword}.png', format = 'png', dpi = 250)
    plt.close()

    # plt.figure(fig4.number)
    # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/CRPS_vs_Variance_OM_percentile_multiple_exps_DISCARD.png', format='png', dpi=250 )
    # plt.close()

    # VARIANCE DISTRIBUTION HISTOGRAMS: -----------------

    plt.figure(fig2.number)  
    # Plot distribution of variance across model types: 
    model_types = list(exps.keys())
    colors = [color_themes[k] for k in range(len(model_types))]
    ax2.hist(variance_all_model_types, bins=150, alpha=0.7, density=True, 
         color=colors, label=model_types, 
         stacked=True, histtype='barstacked')
    ax2.set_ylabel('Density')
    ax2.set_xlabel('Variance')
    ax2.set_xlim([0, 0.045])
    plt.title(f'Variance over Means Distribution Across Random Seeds {keyword}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/variance_OM_distribution_multiple_exps_STACKED_{keyword}.png', 
                format='png', dpi=250)
    plt.close()

    # Distribution of VARIANCE over MEANS across model types - 4 Row Subplots: -----------------

    fig3, axes = plt.subplots(nrows=4, ncols=1, figsize=(11, 11), sharex=True)
    model_types = list(exps.keys())
    colors = [color_themes[i] for i in range(len(model_types))]

    for k, (variance_data, model_type) in enumerate(zip(variance_all_model_types, model_types)):
        axes[k].hist(variance_data, bins=150, alpha=0.4, density=True, 
                    linewidth=2, color=colors[k], edgecolor=colors[k], 
                    histtype='stepfilled', label = f'Mean Variance: {np.mean(variance_data):.5f} \n Mean CRPS: {np.mean(crps_all_model_types[k]):.4f}')
        axes[k].set_ylabel('Density')
        axes[k].set_title(f'{model_type}')
        axes[k].grid(True, alpha=0.5, which='both', linestyle='-', linewidth=0.5)
        axes[k].minorticks_on()
        axes[k].grid(True, alpha=0.4, which='minor', linestyle=':', linewidth=0.5)
        axes[k].set_xlim([0, 0.06])
        axes[k].legend()

    # Only set xlabel on bottom subplot
    axes[-1].set_xlabel('Variance')

    plt.suptitle(f'Variance over MEANS Distribution Across Random Seeds {keyword}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/variance_over_MEANs_distribution_multiple_exps_STEP_{keyword}.png', 
                format='png', dpi=250)
    plt.close()


    # PLOT: CRPS Distribution for Low Var, Medium Var, High Var Samples Across Model Types: -----------------

    num_bins = 6
    threshold = 12

    # Create TWO figures - one for percentiles, one for raw variance
    fig4, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 10), sharex=True)
    fig5, axes2 = plt.subplots(nrows=4, ncols=1, figsize=(8, 10), sharex=True)
    model_types = list(exps.keys())

    for k, model_type in enumerate(model_types):
        if k < len(variance_all_model_types) and k < len(crps_all_model_types):
            variance_data = variance_all_model_types[k]
            crps_data_all_seeds = crps_all_model_types[k]  # Shape: (n_seeds, n_samples)
        
        # Take mean across seeds for this model type
        crps_data = np.mean(crps_data_all_seeds, axis=0)  # Shape: (n_samples,)

        crps_data = crps_data[variance_data <= 0.066]
        variance_data = variance_data[variance_data <= 0.066]
        
        print(f'Model {model_type}: variance shape {variance_data.shape}, crps shape {crps_data.shape}')
        
        # ============= PERCENTILE VERSION =============
        # Convert variance to percentiles for this model type
        variance_percentiles = stats.rankdata(variance_data, method='average') / len(variance_data) * 100
        
        # Create bins based on percentiles (0-100)
        bin_edges_pct = np.linspace(0, 100, num_bins + 1)
        bin_centers_pct = (bin_edges_pct[:-1] + bin_edges_pct[1:]) / 2

        sample_counts_pct = []
        mean_crps_per_bin = np.full((num_bins), np.nan)

        for b in range(num_bins):
            in_bin = (variance_percentiles >= bin_edges_pct[b]) & (variance_percentiles < bin_edges_pct[b+1])
            if np.any(in_bin):  # Only plot if bin contains data
                crps_in_bin = crps_data[in_bin]
                mean_crps_per_bin[b] = np.mean(crps_in_bin) if len(crps_in_bin) > 0 else np.nan

                if len(crps_in_bin) >= threshold:  # Double-check data exists
                    a = 0.4

                    violin_parts1 = axes[k].violinplot(crps_in_bin, positions=[bin_centers_pct[b]], widths=8, showmeans=True)

                    # Set color and alpha for violin bodies
                    for pc in violin_parts1['bodies']:
                        pc.set_facecolor(color_themes[k])
                        pc.set_alpha(a)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(0.2)

                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                        if partname in violin_parts1:
                            violin_parts1[partname].set_edgecolor(color_themes[k])
                            violin_parts1[partname].set_alpha(a)  # Slightly more opaque for visibility
                    
                    # Set color for means if shown
                    if 'cmeans' in violin_parts1:
                        violin_parts1['cmeans'].set_edgecolor('black')
                        violin_parts1['cmeans'].set_alpha(1.0)

                    sample_counts_pct.append((bin_centers_pct[b], len(crps_in_bin)))
                else:
                    sample_counts_pct.append((bin_centers_pct[b], 0))
            else:
                sample_counts_pct.append((bin_centers_pct[b], 0))
        
        # Get current y-axis limits and expand the lower bound for percentile plot
        y_min, y_max = axes[k].get_ylim()
        y_range = y_max - y_min
        new_y_min = y_min - 0.15 * y_range
        axes[k].set_ylim(new_y_min, y_max)
        
        # Add sample count annotations below the violins
        for l, (bin_center, count) in enumerate(sample_counts_pct):
            if count >= threshold:
                axes[k].text(bin_center, new_y_min + 0.05 * y_range, f'n={count} \n CRPS={mean_crps_per_bin[l]:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)
                
        axes[k].set_title(f'{model_type} \n Mean Variance: {np.mean(variance_data):.5f}')
        axes[k].set_ylabel('CRPS')
        axes[k].grid(True, alpha=0.35, which='both', linestyle='-', linewidth=0.5)

        # ============= RAW VARIANCE VERSION =============
        # Create bins based on raw variance values
        bin_edges_raw = np.linspace(np.min(variance_data), np.max(variance_data), num_bins + 1)
        bin_centers_raw = (bin_edges_raw[:-1] + bin_edges_raw[1:]) / 2

        sample_counts_raw = []
        mean_crps_per_bin = np.full((num_bins), np.nan)

        for b in range(num_bins):
            in_bin = (variance_data >= bin_edges_raw[b]) & (variance_data < bin_edges_raw[b+1])
            if np.any(in_bin):  # Only plot if bin contains data
                crps_in_bin = crps_data[in_bin]
                mean_crps_per_bin[b] = np.mean(crps_in_bin) if len(crps_in_bin) > 0 else np.nan

                if len(crps_in_bin) > 0: # always plot all violins
                    if len(crps_in_bin) >= threshold:  # plot sufficient samples
                        a = 0.4
                    else:
                        a = 0.0

                    violin_parts2 = axes2[k].violinplot(crps_in_bin, positions=[bin_centers_raw[b]], widths=bin_centers_raw[1]-bin_centers_raw[0], showmeans=True)

                    # Set color and alpha for violin bodies
                    for pc in violin_parts2['bodies']:
                        pc.set_facecolor(color_themes[k])
                        pc.set_alpha(a)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(0.2)

                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                        if partname in violin_parts2:
                            violin_parts2[partname].set_edgecolor(color_themes[k])
                            violin_parts2[partname].set_alpha(a)  # Slightly more opaque for visibility
                    
                    # Set color for means if shown
                    if 'cmeans' in violin_parts2:
                        violin_parts2['cmeans'].set_edgecolor('black')
                        violin_parts2['cmeans'].set_alpha(1.0)
                    
                    sample_counts_raw.append((bin_centers_raw[b], len(crps_in_bin)))
                else:
                    sample_counts_raw.append((bin_centers_raw[b], 0))
            else:
                sample_counts_raw.append((bin_centers_raw[b], 0))
        
        # Get current y-axis limits and expand the lower bound for raw variance plot
        y_min2, y_max2 = axes2[k].get_ylim()
        y_range2 = y_max2 - y_min2
        new_y_min2 = y_min2 - 0.15 * y_range2
        axes2[k].set_ylim(new_y_min2, y_max2)
        
        # Add sample count annotations below the violins
        for l, (bin_center, count) in enumerate(sample_counts_raw):
            if count > 0:
                axes2[k].text(bin_center, new_y_min2 + 0.05 * y_range2, f'n={count} \n CRPS={mean_crps_per_bin[l]:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)
                
        axes2[k].set_title(f'{model_type} \n Mean Variance: {np.mean(variance_data):.5f}')
        axes2[k].set_ylabel('CRPS')
        axes2[k].grid(True, alpha=0.35, which='both', linestyle='-', linewidth=0.5)
        axes2[k].legend()

    # Finalize percentile plot
    axes[-1].set_xlabel('Variance Percentile')
    axes[-1].set_xlim([0, 100])
    plt.figure(fig4.number)
    plt.suptitle(f'CRPS Distribution Across Variance over Means Percentiles {keyword}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/crps_distribution_across_variance_OM_percentiles_{keyword}.png', 
                format='png', dpi=250)
    plt.close()

    # Finalize raw variance plot
    axes2[-1].set_xlabel('Raw Variance')
    plt.figure(fig5.number)
    plt.suptitle(f'CRPS Distribution Across Raw Variance over Means {keyword}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/crps_distribution_across_raw_variance_OM_{keyword}.png', 
                format='png', dpi=250)
    plt.close()



    # After the main violin plot, create a summary trend plot
    fig_trend, ax_trend = plt.subplots(figsize=(10, 6))

    for k, model_type in enumerate(model_types):
        if k < len(variance_all_model_types) and k < len(crps_all_model_types):
            # Calculate mean CRPS for each percentile bin
            variance_data = variance_all_model_types[k]
            crps_data = np.mean(crps_all_model_types[k], axis=0)
            variance_percentiles = stats.rankdata(variance_data, method='average') / len(variance_data) * 100
            
            bin_edges = np.linspace(0, 100, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            bin_means = []
            bin_stds = []
            
            for b in range(num_bins):
                in_bin = (variance_percentiles >= bin_edges[b]) & (variance_percentiles < bin_edges[b+1])
                if np.any(in_bin):
                    crps_in_bin = crps_data[in_bin]
                    if len(crps_in_bin) > 10:
                        bin_means.append(np.median(crps_in_bin))
                        bin_stds.append(np.std(crps_in_bin))
                    else:
                        bin_means.append(np.nan)
                        bin_stds.append(np.nan)
                else:
                    bin_means.append(np.nan)
                    bin_stds.append(np.nan)
            
            # Plot trend line with error bars
            ax_trend.errorbar(bin_centers, bin_means, yerr=bin_stds, 
                            marker='o', linewidth=2, capsize=5, 
                            color=color_themes[k], label=model_type)

    ax_trend.set_xlabel('Variance Percentile')
    ax_trend.set_ylabel('Median CRPS')
    ax_trend.set_title('CRPS Trend Across Variance Percentiles')
    ax_trend.legend()
    ax_trend.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/crps_trend_across_variance_OM_percentiles_{keyword}.png', 
                format='png', dpi=250)



def variance_OIQR_analysis(experiments, scale_target = True, scale_IQR = False, keyword = None):
    """
    Epistemic Uncertainty Analysis : 
    - Discard plot of CRPS vs Variance across random seeds for a given experiment type, comparing across experiment types: 
        Experiments contains multiple experiment types
        For each experiment type: 
            - Load outputs from each random seed
            - Make if block for how to calculate variance: 
                If by np.var, calculate np.var across all random seeds for each output sample
            - Create discard plot of CRPS mean (y axis) binned by variance across seeds (x axis) 
            - Overlay all experiment types on one plot for comparison
    """

    exps = experiments

    color_themes = {
        0: "#211bd2", 
        1: "#8b1dcf", 
        2: "#d03232",
        3: "#e88a0f"
    }
    fig1, ax1 = plt.subplots(figsize = (10, 7))
    fig2, ax2 = plt.subplots(figsize = (10, 7))
    fig4, ax4 = plt.subplots(figsize = (10, 7))

    variance_all_model_types = []
    crps_all_model_types = []

    for i, (exp_type, exp_list) in enumerate(exps.items()):
        print(f'Processing experiment type: {exp_type}')

        if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                data_type = "OBS"
        elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                data_type = "E3SM"
        
        #identify lengths for accurate preallocation: 
        try:
            output_preall = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_list[0]}/{exp_list[0]}_network_SHASH_parameters.pkl')
        except FileNotFoundError:
            pattern = f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_list[0]}/exp*_{exp_list[0]}_OOD_INFERENCE_network_SHASH_parameters.pkl'
            matching_files = glob.glob(pattern)
            output_preall = load_pickle(matching_files[0]) if matching_files else None


        all_crps = np.empty((len(exp_list), len(output_preall)))
        all_IQR = np.empty((len(exp_list), len(output_preall)))

        for iexp, exp_name in enumerate(exp_list):
            print(f'  Processing experiment: {exp_name}')

            config = utils.get_config(exp_name)

            if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                # Load climatology statistics
                # climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                # target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
    
            elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "OBS(E3SM)sv", "E3SM(E3SM)sv"]:
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                # climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                # target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]


            # Load the output and target data for this experiment
            try:
                output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')
            except FileNotFoundError:
                pattern = f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/exp*_{exp_name}_OOD_INFERENCE_network_SHASH_parameters.pkl'
                matching_files = glob.glob(pattern)
                output = load_pickle(matching_files[0]) if matching_files else None

            climo_crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_climatology_values.pkl')
            mean_climo_crps = np.mean(climo_crps)

            # open crps: 
            crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

            # Calculate IQR
            iqr = iqr_basic(output)

            if scale_IQR == True:
                print("Scaling IQR, CRPS Values")
                iqr_xr = xr.DataArray(iqr, coords=[target['time']], dims=["time"])
                # scale iqr by day of year mean:
                daily_iqr = iqr_xr.groupby('time.dayofyear').mean('time')
                scaled_iqr = iqr_xr.groupby('time.dayofyear') / daily_iqr
                #ungroup scaled_iqr: 
                scaled_iqr = scaled_iqr.sortby('time') 
                iqr = scaled_iqr.values

                # Scale Climo CRPS by day of year mean: 
                climo_crps_xr = xr.DataArray(climo_crps, coords=[target['time']], dims=["time"])
                daily_climo_crps = climo_crps_xr.groupby('time.dayofyear').mean('time')
                scaled_climo_crps = climo_crps_xr.groupby('time.dayofyear') / daily_climo_crps
                scaled_climo_crps = scaled_climo_crps.sortby('time')
                mean_climo_crps = np.mean(scaled_climo_crps)

                # Scale CRPS by day of year as well: 
                crps_xr = xr.DataArray(crps, coords=[target['time']], dims=["time"])
                # scale crps by day of year mean:
                daily_crps = crps_xr.groupby('time.dayofyear').mean('time')
                scaled_crps = crps_xr.groupby('time.dayofyear') / daily_crps
                scaled_crps = scaled_crps.sortby('time')
                crps = scaled_crps.values
            else: 
                print("Not scaling IQR or CRPS values")
                pass


            all_IQR[iexp] = iqr

            all_crps[iexp] = crps

        crps_all_model_types.append(all_crps)

        # Calculate variance across random seeds for each sample
        variance_across_seeds = np.var(all_IQR, axis=0)
        variance_all_model_types.append(variance_across_seeds)

        # FIGURE: Variance Analysis of E3SM(OBS) : ---------------
        if exp_type in ["E3SM(OBS)", "E3SM(OBS)sv", "E3SM(E3SM)", "E3SM(E3SM)sv", "OBS(OBS)", "OBS(OBS)sv", "OBS(E3SM)", "OBS(E3SM)sv"]: 
            # Analyze the high variance samples in E3SM(OBS)
            # Select high variance samples based on threshold:
            var_lim = 20 # 20%
            high_variance_indices = np.argsort(variance_across_seeds)[-int(0.2 * variance_across_seeds.shape[0]):]
            print(f"number of high variance indices: {len(high_variance_indices)}")
            non_high_variance_indices = np.unique(np.setdiff1d(np.arange(variance_across_seeds.shape[0]), high_variance_indices))
            # how many samples is 20% of the data: 
            sample_size_lim = 0.2 * variance_across_seeds.shape[0]
            # find low variance indices corresponding to the lowest 20% of variance values:
            low_variance_indices = np.argsort(variance_across_seeds)[:int(sample_size_lim)]
            print(f"number of low variance indices: {len(low_variance_indices)}")

            all_input_maps_high_var = []
            all_input_maps_non_high_var = []
            all_input_maps_low_var = []

            # Open Target: 
            # Load testing target data
            if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                data_type = "OBS"
                data_vars = ["tp", "skt", "z"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
                target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]

                if exp_type in ["E3SM(OBS)sv", "OBS(OBS)sv"]:
                    # scale target by day of year variance: 
                    daily_target_grouped_var = target.groupby('time.dayofyear').var('time')
                    scaled_target = target.groupby('time.dayofyear') / daily_target_grouped_var
                    scaled_target = scaled_target.sortby('time')
                    target = scaled_target

            elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                data_type = "E3SM"
                data_vars = ["PRECT", "TS", "Z500"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]
                target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]

                if exp_type in ["E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                    # scale target by day of year variance:
                    daily_target_grouped_var = target.groupby('time.dayofyear').var('time')
                    scaled_target = target.groupby('time.dayofyear') / daily_target_grouped_var
                    scaled_target = scaled_target.sortby('time')
                    target = scaled_target

            high_variance_dates = target.time[high_variance_indices]
            print(f"length of high var dates: {len(high_variance_dates)}")
            non_high_var_dates = target.time[non_high_variance_indices]
            print(f"length of non high var dates: {len(non_high_var_dates)}")
            low_variance_dates = target.time[low_variance_indices]

            high_var_input_maps = input_maps.sel(time = high_variance_dates)
            non_high_var_input_maps = input_maps.sel(time = non_high_var_dates)
            low_var_input_maps = input_maps.sel(time = low_variance_dates)

            # Composite Plot of Input Maps: SIMPLE MEAN
            all_input_maps_high_var.append(high_var_input_maps)
            all_input_maps_non_high_var.append(non_high_var_input_maps)
            all_input_maps_low_var.append(low_var_input_maps)

            # Analyze ENSO in HIGH VARIANCE Selected Dates: 
            enso_dates_pkl = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_daily_enso_timestamps.pkl')
            enso_baseline_frequencies = analysis.analysis_metrics.baseline_enso_frequencies(data_type)

            # check which key (category) each of the target dates falls into, and create a list with either "EN", "LN" or "N"
            enso_phase = []
            for date in high_variance_dates.values:
                if date in enso_dates_pkl['El Nino']:
                    enso_phase.append("EN")
                elif date in enso_dates_pkl['La Nina']:
                    enso_phase.append("LN")
                else:
                    enso_phase.append("N")

            test_dates = target.time

            if data_type == "OBS":
                non_leap_EN_dates = [
                    date for date in enso_dates_pkl['El Nino']
                    if not (date.astype('datetime64[M]').astype(int) % 12 + 1 == 2 and 
                            (date.astype('datetime64[D]') - date.astype('datetime64[M]')).astype(int) + 1 == 29)]
                non_leap_LN_dates = [
                    date for date in enso_dates_pkl['La Nina']
                    if not (date.astype('datetime64[M]').astype(int) % 12 + 1 == 2 and 
                            (date.astype('datetime64[D]') - date.astype('datetime64[M]')).astype(int) + 1 == 29)]
                non_leap_N_dates = [
                    date for date in enso_dates_pkl['Neutral']
                    if not (date.astype('datetime64[M]').astype(int) % 12 + 1 == 2 and 
                            (date.astype('datetime64[D]') - date.astype('datetime64[M]')).astype(int) + 1 == 29)]
            elif data_type == "E3SM":
                non_leap_EN_dates = enso_dates_pkl['El Nino']
                non_leap_LN_dates = enso_dates_pkl['La Nina']
                non_leap_N_dates = enso_dates_pkl['Neutral']
            
            # dates must fall within min and max of test_dates: 
            non_leap_EN_dates = [date for date in non_leap_EN_dates if date >= test_dates.min() and date <= test_dates.max()]
            non_leap_LN_dates = [date for date in non_leap_LN_dates if date >= test_dates.min() and date <= test_dates.max()]
            non_leap_N_dates = [date for date in non_leap_N_dates if date >= test_dates.min() and date <= test_dates.max()]

            en_dates_in_test = test_dates.sel(time=non_leap_EN_dates)
            ln_dates_in_test = test_dates.sel(time=non_leap_LN_dates)
            n_dates_in_test = test_dates.sel(time=non_leap_N_dates)

            print(f"El Nino proportion in high variance OIQR dates relative to all El Nino dates in test set: {enso_phase.count('EN')/len(en_dates_in_test)*100:.1f}% ({enso_phase.count('EN')} out of {len(en_dates_in_test)})")
            print(f"La Nina proportion in high variance OIQR dates relative to all La Nina dates in test set: {enso_phase.count('LN')/len(ln_dates_in_test)*100:.1f}% ({enso_phase.count('LN')} out of {len(ln_dates_in_test)})")
            print(f"Neutral proportion in high variance OIQR dates relative to all Neutral dates in test set: {enso_phase.count('N')/len(n_dates_in_test)*100:.1f}% ({enso_phase.count('N')} out of {len(n_dates_in_test)})")

            # ENSO Figure 1: Relative proportion of high var samples 
            fig, ax_enso = plt.subplots(figsize=(8, 6))
            bin_edges = np.array([-0.5, 0.5, 1.5, 2.5])
            bin_centers = np.array([0, 1, 2])
            bar_width = 0.4

            num_LN = enso_phase.count('LN')
            num_EN = enso_phase.count('EN')
            num_N = enso_phase.count('N')
            sum_total_phases = num_EN + num_LN + num_N
            enso_phase_dist = [num_EN / sum_total_phases, 
                            num_LN / sum_total_phases, 
                            num_N / sum_total_phases]

            bars = ax_enso.bar(bin_centers, enso_phase_dist, width=bar_width, color="#fbc13a", alpha=0.7, edgecolor='black')

            # Reference lines for each ENSO phase
            enso_phases = ['El Nino', 'La Nina', 'Neutral']
            for k, phase in enumerate(enso_phases):
                freq = enso_baseline_frequencies[phase]
                # Draw a horizontal line across the width of the bar for phase i
                ax_enso.hlines(y=freq, xmin=k - bar_width/2, xmax=k + bar_width/2, 
                        color="#3C3B3B", linewidth=2, linestyle='-', 
                        label='Reference' if k==0 else None)

            ax_enso.set_ylim([0, max(max(enso_phase_dist), max(enso_baseline_frequencies.values())) * 1.15])
            ax_enso.set_xticks(bin_centers)
            ax_enso.set_xticklabels(['El Nino', 'La Nina', 'Neutral'])
            ax_enso.set_xlabel('ENSO Phase')
            ax_enso.set_ylabel('Density')
            ax_enso.set_title(f'ENSO Phase Distribution | E3SM(OBS) High Variance (over IQR) Samples {keyword}')

            # Legend with only one entry for reference lines
            handles, labels = ax_enso.get_legend_handles_labels()
            if 'Reference' in labels:
                idx = labels.index('Reference')
                ax_enso.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
            else:
                ax_enso.legend()
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_phase_distribution_E3SM-OBS_high_var_OIQR_{keyword}.png', format='png', dpi=250)
            plt.close()

            # ENSO Figure 2:  Difference plot between high var EN and all other EN:
            # fig, ax_enso2 = plt.subplots(figsize=(8, 6))


            ######## MJO @ VERIFICATION DAY ########
            phase_timestamps = analysis_metrics.mjo_timestamps(data_type, config)

            if exp_type in ["E3SM(E3SM)", "E3SM(E3SM)sv", "OBS(E3SM)", "OBS(E3SM)sv"]:
                # Check the calendar type of phase_timestamps
                phase_sample = phase_timestamps.values[0]
                high_var_sample = high_variance_dates.values[0]

                print(f"Phase timestamps type: {type(phase_sample)}")
                print(f"High variance dates type: {type(high_var_sample)}")

                # Convert phase_timestamps to cftime.DatetimeNoLeap if needed
                if isinstance(high_var_sample, cftime.DatetimeNoLeap) and not isinstance(phase_sample, cftime.DatetimeNoLeap):
                    # Phase timestamps needs to be converted to NoLeap calendar
                    
                    # Convert standard datetime to cftime NoLeap
                    phase_time_noleap = xr.CFTimeIndex([
                        cftime.DatetimeNoLeap(pd.Timestamp(t).year, pd.Timestamp(t).month, pd.Timestamp(t).day)
                        for t in phase_timestamps.values
                    ])
                    
                    # Create new DataArray with converted time coordinate
                    phase_timestamps_converted = phase_timestamps.copy()
                    phase_timestamps_converted['time'] = phase_time_noleap
                    
                    # Now exact selection will work
                    selected_mjo_phases = phase_timestamps_converted.sel(time=high_variance_dates.values)
                    
                elif isinstance(phase_sample, cftime.DatetimeNoLeap) and not isinstance(high_var_sample, cftime.DatetimeNoLeap):
                    # High variance dates needs to be converted to NoLeap calendar
                    high_var_noleap = [
                        cftime.DatetimeNoLeap(pd.Timestamp(t).year, pd.Timestamp(t).month, pd.Timestamp(t).day)
                        for t in high_variance_dates.values
                    ]
                    selected_mjo_phases = phase_timestamps.sel(time=high_var_noleap)
                
                else: 
                    selected_mjo_phases = phase_timestamps.sel(time=high_variance_dates)
            else: 
                selected_mjo_phases = phase_timestamps.sel(time=high_variance_dates)


            # if "OBS" in data_type:
            #     mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_phase_frequencies_ERA5_1940_2023.pkl')
            # elif "E3SM" in data_type:
            #     mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_frequencies_E3SM_1850_2014.pkl')

            # counts = np.bincount(selected_mjo_phases["phase"], minlength=9)
            # total = counts.sum()
            # densities = counts / total

            # phases = np.arange(0, 9)
            # phase_labels = [str(i) for i in phases]

            # fig, ax_mjo = plt.subplots(figsize=(8, 6))
            # bar_width = 0.8

            # # Bar plot for density
            # bars = ax_mjo.bar(phases, densities, width=bar_width, color="#9c4781", alpha=0.7, edgecolor='black', label='Selected Samples')

            # # Reference lines for each phase
            # for i, freq in enumerate(mjo_ref_frequencies_all_data):
            #     # Draw a horizontal line across the width of the bar for phase i
            #     ax_mjo.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, color="#3C3B3B", linewidth=2, linestyle='-', label='Reference' if i==0 else None)

            # ax_mjo.set_xticks(phases)
            # ax_mjo.set_xticklabels(phase_labels)
            # ax_mjo.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
            # ax_mjo.set_xlabel('MJO Phase')
            # ax_mjo.set_ylabel('Density')
            # ax_mjo.set_title('MJO Phase Distribution for High Variance E3SM(OBS) Samples | {keyword}')
            # handles, labels = ax_mjo.get_legend_handles_labels()
            # # Only show one legend entry for the reference lines
            # if 'Reference' in labels:
            #     idx = labels.index('Reference')
            #     ax_mjo.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
            # else:
            #     ax_mjo.legend()
            # plt.tight_layout()
            # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_E3SM-OBS_high_var_{keyword}.png', format='png', dpi=250)
            # plt.close()

            ######## MJO @ INITIALIZATION DAY ########

            high_variance_input_dates = high_variance_dates - np.timedelta64(config['databuilder']['lagtime'])
            high_var_dates = high_variance_input_dates.astype('datetime64[D]')
            phase_dates = phase_timestamps.time.astype('datetime64[D]')

            # Find valid dates that exist in both
            valid_mask = np.isin(high_var_dates, phase_dates)
            valid_high_var_dates = high_var_dates[valid_mask]

            print(f"Using {valid_mask.sum()} of {len(high_var_dates)} high variance dates")

            # Select only valid dates
            input_mjo_phases = phase_timestamps.sel(time=valid_high_var_dates)

            if "OBS" in data_type:
                mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_phase_frequencies_ERA5_1940_2023.pkl')
            elif "E3SM" in data_type:
                mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_frequencies_E3SM_1850_2014.pkl')

            counts = np.bincount(input_mjo_phases["phase"], minlength=9)
            total = counts.sum()
            densities = counts / total

            phases = np.arange(0, 9)
            phase_labels = [str(i) for i in phases]

            fig, ax_mjo = plt.subplots(figsize=(8, 6))
            bar_width = 0.8

            # Bar plot for density
            bars = ax_mjo.bar(phases, densities, width=bar_width, color="#4a479c", alpha=0.7, edgecolor='black', label='Selected Samples')

            # Reference lines for each phase
            for k, freq in enumerate(mjo_ref_frequencies_all_data):
                # Draw a horizontal line across the width of the bar for phase k
                ax_mjo.hlines(y=freq, xmin=k - bar_width/2, xmax=k + bar_width/2, color="#3C3B3B", linewidth=2, linestyle='-', label='Reference' if k==0 else None)

            ax_mjo.set_xticks(phases)
            ax_mjo.set_xticklabels(phase_labels)
            ax_mjo.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
            ax_mjo.set_xlabel('MJO Phase')
            ax_mjo.set_ylabel('Density')
            ax_mjo.set_title(f'MJO Phase Distribution for High Variance (over IQR) E3SM(OBS) Samples \n Initialization Day {keyword}')
            handles, labels = ax_mjo.get_legend_handles_labels()
            # Only show one legend entry for the reference lines
            if 'Reference' in labels:
                idx = labels.index('Reference')
                ax_mjo.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
            else:
                ax_mjo.legend()
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_E3SM-OBS_high_var_OIQR_INIT_day_{keyword}.png', format='png', dpi=250)
            plt.close()

            ## Analyze ENSO & MJO(initialization day) in LOW Variance Dates: ------

            enso_phase = []
            for date in low_variance_dates.values: # LOW variance dates
                if date in enso_dates_pkl['El Nino']:
                    enso_phase.append("EN")
                elif date in enso_dates_pkl['La Nina']:
                    enso_phase.append("LN")
                else:
                    enso_phase.append("N")

            print(f"El Nino proportion in high variance (over IQR) dates relative to all El Nino dates in test set: {enso_phase.count('EN')/len(en_dates_in_test)*100:.1f}% ({enso_phase.count('EN')} out of {len(en_dates_in_test)})")
            print(f"La Nina proportion in high variance (over IQR) dates relative to all La Nina dates in test set: {enso_phase.count('LN')/len(ln_dates_in_test)*100:.1f}% ({enso_phase.count('LN')} out of {len(ln_dates_in_test)})")
            print(f"Neutral proportion in high variance (over IQR) dates relative to all Neutral dates in test set: {enso_phase.count('N')/len(n_dates_in_test)*100:.1f}% ({enso_phase.count('N')} out of {len(n_dates_in_test)})")

            # ENSO Figure 1: Relative proportion of high var samples 
            fig, ax_enso = plt.subplots(figsize=(8, 6))
            bin_edges = np.array([-0.5, 0.5, 1.5, 2.5])
            bin_centers = np.array([0, 1, 2])
            bar_width = 0.4

            num_LN = enso_phase.count('LN')
            num_EN = enso_phase.count('EN')
            num_N = enso_phase.count('N')
            sum_total_phases = num_EN + num_LN + num_N
            enso_phase_dist = [num_EN / sum_total_phases, 
                            num_LN / sum_total_phases, 
                            num_N / sum_total_phases]

            bars = ax_enso.bar(bin_centers, enso_phase_dist, width=bar_width, color="#fbc13a", alpha=0.7, edgecolor='black')

            # Reference lines for each ENSO phase
            enso_phases = ['El Nino', 'La Nina', 'Neutral']
            for j, phase in enumerate(enso_phases):
                freq = enso_baseline_frequencies[phase]
                # Draw a horizontal line across the width of the bar for phase j
                ax_enso.hlines(y=freq, xmin=j - bar_width/2, xmax=j + bar_width/2, 
                        color="#3C3B3B", linewidth=2, linestyle='-', 
                        label='Reference' if j==0 else None)

            ax_enso.set_ylim([0, max(max(enso_phase_dist), max(enso_baseline_frequencies.values())) * 1.15])
            ax_enso.set_xticks(bin_centers)
            ax_enso.set_xticklabels(['El Nino', 'La Nina', 'Neutral'])
            ax_enso.set_xlabel('ENSO Phase')
            ax_enso.set_ylabel('Density')
            ax_enso.set_title(f'ENSO Phase Distribution | E3SM(OBS) Low Variance (over IQR) Samples | {keyword}')

            # Legend with only one entry for reference lines
            handles, labels = ax_enso.get_legend_handles_labels()
            if 'Reference' in labels:
                idx = labels.index('Reference')
                ax_enso.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
            else:
                ax_enso.legend()
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_phase_distribution_E3SM-OBS_low_var_OIQR_{keyword}.png', format='png', dpi=250)
            plt.close()

            ######## MJO @ INITIALIZATION DAY ########

            low_variance_input_dates = low_variance_dates - np.timedelta64(config['databuilder']['lagtime'])
            low_var_dates = low_variance_input_dates.astype('datetime64[D]')
            phase_dates = phase_timestamps.time.astype('datetime64[D]')

            # Find valid dates that exist in both
            valid_mask = np.isin(low_var_dates, phase_dates)
            valid_low_var_dates = low_var_dates[valid_mask]

            print(f"Using {valid_mask.sum()} of {len(low_var_dates)} low variance dates")

            # Select only valid dates
            input_mjo_phases = phase_timestamps.sel(time=valid_low_var_dates)

            if "OBS" in data_type:
                mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_phase_frequencies_ERA5_1940_2023.pkl')
            elif "E3SM" in data_type:
                mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_frequencies_E3SM_1850_2014.pkl')

            counts = np.bincount(input_mjo_phases["phase"], minlength=9)
            total = counts.sum()
            densities = counts / total

            phases = np.arange(0, 9)
            phase_labels = [str(i) for i in phases]

            fig, ax_mjo = plt.subplots(figsize=(8, 6))
            bar_width = 0.8

            # Bar plot for density
            bars = ax_mjo.bar(phases, densities, width=bar_width, color="#4a479c", alpha=0.7, edgecolor='black', label='Selected Samples')

            # Reference lines for each phase
            for j, freq in enumerate(mjo_ref_frequencies_all_data):
                # Draw a horizontal line across the width of the bar for phase j
                ax_mjo.hlines(y=freq, xmin=j - bar_width/2, xmax=j + bar_width/2, color="#3C3B3B", linewidth=2, linestyle='-', label='Reference' if j==0 else None)

            ax_mjo.set_xticks(phases)
            ax_mjo.set_xticklabels(phase_labels)
            ax_mjo.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
            ax_mjo.set_xlabel('MJO Phase')
            ax_mjo.set_ylabel('Density')
            ax_mjo.set_title(f'MJO Phase Distribution for Low Variance (over IQR) E3SM(OBS) Samples \n Initialization Day | {keyword}')
            handles, labels = ax_mjo.get_legend_handles_labels()
            # Only show one legend entry for the reference lines
            if 'Reference' in labels:
                idx = labels.index('Reference')
                ax_mjo.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
            else:
                ax_mjo.legend()
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_E3SM-OBS_low_var_OIQR_INIT_day_{keyword}.png', format='png', dpi=250)
            plt.close()


            # Composite difference plot: El Nino High variance to El Nino all other samples
            # en_dates_in_test and high_var dates overlap: 
            overlapping_en_dates = np.array([date for date in en_dates_in_test.values if date in high_variance_dates.values])
            # en_dates_in_test and non_high_var_dates overlap:
            overlapping_non_high_en_dates = np.array([date for date in en_dates_in_test.values if date in non_high_var_dates.values])

            high_var_en_input_maps = input_maps.sel(time=overlapping_en_dates)
            high_var_en_input_map_mean = high_var_en_input_maps.mean(dim='time')
            non_high_var_en_input_maps = input_maps.sel(time=overlapping_non_high_en_dates)
            non_high_var_en_input_map_mean = non_high_var_en_input_maps.mean(dim='time')

            en_diff_input_map = high_var_en_input_map_mean - non_high_var_en_input_map_mean

            fig, ax = plt.subplots(3, 3, figsize=(15, 11), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
            cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
            vmin_list = np.zeros(3)
            vmax_list = np.zeros(3)

            # Plot three rows of maps: (1st row) High variance el ninos (2nd row) non-high variance el ninos (3rd row) difference
            for k in range(3):
                vmin_list = [-1, -1.5, -1.5]
                vmax_list = [1, 1.5, 1.5]

                # High Variance El Nino
                im1 = ax[0, k].pcolormesh(
                    high_var_en_input_map_mean['lon'],
                    high_var_en_input_map_mean['lat'],
                    high_var_en_input_map_mean[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[0, k].coastlines()
                ax[0, k].set_title(f'High Variance OIQR El Nino Input Map: {variable_names[k]} \n N = {len(overlapping_en_dates)}')
                plt.colorbar(im1, ax=ax[0, k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')

                # Non-High Variance El Nino
                im2 = ax[1, k].pcolormesh(
                    non_high_var_en_input_map_mean['lon'],
                    non_high_var_en_input_map_mean['lat'],
                    non_high_var_en_input_map_mean[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[1, k].coastlines()
                ax[1, k].set_title(f'Non-High Variance OIQR El Nino Input Map: {variable_names[k]} \n N = {len(overlapping_non_high_en_dates)}')
                plt.colorbar(im2, ax=ax[1, k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')

                # Difference Map
                im3 = ax[2, k].pcolormesh(
                    en_diff_input_map['lon'],
                    en_diff_input_map['lat'],
                    en_diff_input_map[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[2, k].coastlines()
                ax[2, k].set_title(f'Difference Map (High Var - Non-High Var): {variable_names[k]}')
                plt.colorbar(im3, ax=ax[2, k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_high_var_vs_non_high_var_OIQR_input_maps_E3SM-OBS_{keyword}.png', format='png', dpi=250)
            plt.close()
            
            
            # Composite Plot of Input Maps: SIMPLE MEAN
            combined_input_maps = xr.concat(all_input_maps_high_var, dim='time')
            mean_input_map = combined_input_maps.mean(dim='time')

            fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
            cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
            vmin_list = np.zeros(3)
            vmax_list = np.zeros(3)

            for k in range(3):
                vmin_list = [-0.7, -1, -1]
                vmax_list = [0.7, 1, 1]

                im = ax[k].pcolormesh(
                    mean_input_map['lon'],
                    mean_input_map['lat'],
                    mean_input_map[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[k].coastlines()
                ax[k].set_title(f'Mean Input Map: {variable_names[k]} | High Variance (over IQR) E3SM(OBS) > {var_lim}')
                plt.colorbar(im, ax=ax[k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/high_var_OIQR_composite_input_maps_E3SM-OBS_{keyword}.png', format='png', dpi=250)
            plt.close()

            #  PLOT: LOW Variance Input Map Signals: 
            combined_low_var_input_maps = xr.concat(all_input_maps_low_var, dim='time')
            mean_low_var_input_map = combined_low_var_input_maps.mean(dim='time')

            fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
            cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
            vmin_list = np.zeros(3)
            vmax_list = np.zeros(3)

            for k in range(3):
                vmin_list = [-0.7, -1, -1]
                vmax_list = [0.7, 1, 1]

                im = ax[k].pcolormesh(
                    mean_low_var_input_map['lon'],
                    mean_low_var_input_map['lat'],
                    mean_low_var_input_map[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[k].coastlines()
                ax[k].set_title(f'Mean Low Variance (over IQR) Input Map: {variable_names[k]} | E3SM(OBS) \n Bottom 20% Variance')
                plt.colorbar(im, ax=ax[k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/low_var_OIQR_composite_maps_E3SM-OBS_{keyword}.png', format='png', dpi=250)
            plt.close()

            # Difference Plot : High Var Input Maps vs Low Var Input Maps! 
            combined_high_var_input_maps = xr.concat(all_input_maps_high_var, dim='time')
            combined_non_high_var_input_maps = xr.concat(all_input_maps_non_high_var, dim='time')
            mean_high_var_input_map = combined_high_var_input_maps.mean(dim='time')
            mean_non_high_var_input_map = combined_non_high_var_input_maps.mean(dim='time')
            mean_diff_input_map = mean_high_var_input_map - mean_non_high_var_input_map

            fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
            cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
            vmin_list = np.zeros(3)
            vmax_list = np.zeros(3)

            for k in range(3):
                vmin_list = [-0.7, -1, -1]
                vmax_list = [0.7, 1, 1]

                im = ax[k].pcolormesh(
                    mean_input_map['lon'],
                    mean_input_map['lat'],
                    mean_diff_input_map[..., k],
                    cmap=cmap_list[k],
                    vmin=vmin_list[k],
                    vmax=vmax_list[k],
                    transform=ccrs.PlateCarree(central_longitude=0)
                )
                ax[k].coastlines()
                ax[k].set_title(f'Mean Difference Input Map: {variable_names[k]} | E3SM(OBS) \n Variance > {var_lim}% - Variance <= {var_lim}% (over IQR)')
                plt.colorbar(im, ax=ax[k], orientation='horizontal', pad=0.05, label=f'{variable_names[k]} Anomaly')
            plt.tight_layout()
            plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/high_var_OIQR_composite_DIFF_maps_E3SM-OBS_{keyword}.png', format='png', dpi=250)
            plt.close()


        # # FIGURE : CRPS vs Variance as DISCARD Plot: --------------------- SOMETHING IS WRONG!!! 
        # percentiles = np.linspace(100, 0, 21)
        # var_sorted_indices = np.argsort(variance_across_seeds)  

        # plt.figure(fig4.number)
        # for iexp, exp_name in enumerate(exp_list):

        #     avg_crps = np.empty([len(exp_list), len(percentiles)])
        #     avg_variance = []
        #     sample_index = np.zeros((len(variance_across_seeds), len(percentiles)))

        #     # Sort by Variance
        #     var_sorted = variance_across_seeds[var_sorted_indices]

        #     for ip, p in enumerate(percentiles):
        #         # percentage of samples to keep for each round of the loop
        #         num_to_keep = int(len(var_sorted) * p / 100)
                
        #         indices = var_sorted_indices[:num_to_keep]

        #         if len(indices) == 0:
        #             avg_crps[iexp, ip] = np.nan
        #             avg_variance.append(np.nan)
        #         else:
        #             avg_crps[iexp, ip] = np.mean(all_crps[iexp, indices])
        #             avg_variance.append(np.mean(variance_across_seeds[indices]))
        #             sample_index[:len(indices), ip] = indices

        #     ax4.plot(percentiles, avg_crps[iexp], alpha = 0.3, linewidth = 1.2, color=color_themes[i])

        # mean_crps_line = np.mean(avg_crps, axis = 0)

        # ax4.plot(percentiles, mean_crps_line, alpha = 0.6, linewidth = 2, color = color_themes[i], label = f'{exp_type}')

        # if i in [0, 2]: 
        #     ax4.axhline(y=mean_climo_crps, color=color_themes[i], linestyle='--', label=f'CRPS Mean Climatology for {data_type}')

        # plt.gca().invert_xaxis()
        # ax4.set_ylabel('Average CRPS')
        # ax4.set_xlabel('Model Variance Percentile (% Data Remaining)')
        # ax4.set_xlim([100, 1])
        # plt.tight_layout()
        # plt.legend()
   
        # FIGURE : Standard CRPS vs Variance Binned Plot -----------------
        # Bin the variance values
        n_bins = 21  # You can adjust the number of bins
        bin_edges = np.linspace(np.min(variance_across_seeds), np.max(variance_across_seeds), n_bins + 1)
        bin_indices = np.digitize(variance_across_seeds, bin_edges) - 1  # bins are 0-indexed

        mean_crps_per_bin = np.full((len(exp_list), n_bins), np.nan)  # Initialize with NaN
        bin_centers = np.array([(bin_edges[b] + bin_edges[b+1]) / 2 for b in range(n_bins)])

        all_valid_mean_crps = []

        plt.figure(fig1.number) 

        for exp in range(len(exp_list)):
            for b in range(n_bins):
                in_bin = bin_indices == b
                if np.any(in_bin):
                    mean_crps = np.mean(all_crps[exp, in_bin]) 
                    mean_crps_per_bin[exp, b] = mean_crps

            # Create mask to filter out empty bins (NaN values)
            valid_mask = ~np.isnan(mean_crps_per_bin[exp])
            valid_bin_centers = bin_centers[valid_mask]
            valid_mean_crps = mean_crps_per_bin[exp, valid_mask]

            all_valid_mean_crps.append(valid_mean_crps)
            
            if exp == 0:
                if i in [0, 2]: 
                    ax1.axhline(y=mean_climo_crps, color=color_themes[i], linestyle='--', label=f'CRPS Mean Climatology for {data_type}')
            else:
                ax1.plot(valid_bin_centers, valid_mean_crps, alpha=0.25, linewidth=1.1, 
                        color=color_themes[i])
        
        mean_line_crps = np.nanmean(all_valid_mean_crps, axis=0)
        ax1.plot(valid_bin_centers, mean_line_crps, alpha=0.7, linewidth=2.2, 
                color=color_themes[i], label=f'{exp_type}')

        plt.figure(fig1.number)
        ax1.set_ylabel('CRPS')
        ax1.set_xlabel('Variance')
        ax1.set_xlim([0, 0.03])
        ax1.set_ylim([0.35, 1.5])
        plt.title(f'CRPS vs Variance over IQR Plot')
        plt.legend()
        plt.tight_layout()

    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/CRPS_vs_variance_OIQR_multiple_exps_{keyword}.png', format = 'png', dpi = 250)
    plt.close()

    plt.figure(fig4.number)
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/CRPS_vs_Variance_OIQR_percentile_multiple_exps_DISCARD.png', format='png', dpi=250 )
    plt.close()

    # VARIANCE DISTRIBUTION HISTOGRAMS: -----------------

    plt.figure(fig2.number)  
    # Plot distribution of variance across model types: 
    model_types = list(exps.keys())
    colors = [color_themes[k] for k in range(len(model_types))]
    ax2.hist(variance_all_model_types, bins=150, alpha=0.7, density=True, 
         color=colors, label=model_types, 
         stacked=True, histtype='barstacked')
    ax2.set_ylabel('Density')
    ax2.set_xlabel('Variance')
    ax2.set_xlim([0, 0.045])
    plt.title(f'Variance over IQR Distribution Across Random Seeds {keyword}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/variance_OIQR_distribution_multiple_exps_STACKED_{keyword}.png', 
                format='png', dpi=250)
    plt.close()

    # Distribution of VARIANCE over IQR across model types - 4 Row Subplots: -----------------

    fig3, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 12), sharex=True)
    model_types = list(exps.keys())
    colors = [color_themes[i] for i in range(len(model_types))]

    for k, (variance_data, model_type) in enumerate(zip(variance_all_model_types, model_types)):
        axes[k].hist(variance_data, bins=150, alpha=0.4, density=True, 
                    linewidth=2, color=colors[k], edgecolor=colors[k], 
                    histtype='stepfilled', label = f'Mean Variance: {np.mean(variance_data):.5f}')
        axes[k].set_ylabel('Density')
        axes[k].set_title(f'{model_type}')
        axes[k].grid(True, alpha=0.5, which='both', linestyle='-', linewidth=0.5)
        axes[k].minorticks_on()
        axes[k].grid(True, alpha=0.4, which='minor', linestyle=':', linewidth=0.5)
        axes[k].set_xlim([0, 0.06])
        axes[k].legend()

    # Only set xlabel on bottom subplot
    axes[-1].set_xlabel('Variance')

    plt.suptitle(f'Variance over IQR Distribution Across Random Seeds {keyword}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/variance_over_IQR_distribution_multiple_exps_STEP_{keyword}.png', 
                format='png', dpi=250)
    plt.close()

    # PLOT: CRPS Distribution for Low Var, Medium Var, High Var Samples Across Model Types: ---------------------------------------

    # PLOT: CRPS Distribution for Low Var, Medium Var, High Var Samples Across Model Types: -----------------
    num_bins = 6
    threshold = 12

    # Create TWO figures - one for percentiles, one for raw variance
    fig4, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 10), sharex=True)
    fig5, axes2 = plt.subplots(nrows=4, ncols=1, figsize=(8, 10), sharex=True)
    model_types = list(exps.keys())

    for k, model_type in enumerate(model_types):
        if k < len(variance_all_model_types) and k < len(crps_all_model_types):
            variance_data = variance_all_model_types[k]
            crps_data_all_seeds = crps_all_model_types[k]  # Shape: (n_seeds, n_samples)
        
        # Take mean across seeds for this model type
        crps_data = np.mean(crps_data_all_seeds, axis=0)  # Shape: (n_samples,)

        crps_data = crps_data[variance_data <= 0.066]
        variance_data = variance_data[variance_data <= 0.066]
        
        print(f'Model {model_type}: variance shape {variance_data.shape}, crps shape {crps_data.shape}')
        
        # ============= PERCENTILE VERSION =============
        # Convert variance to percentiles for this model type
        variance_percentiles = stats.rankdata(variance_data, method='average') / len(variance_data) * 100
        
        # Create bins based on percentiles (0-100)
        bin_edges_pct = np.linspace(0, 100, num_bins + 1)
        bin_centers_pct = (bin_edges_pct[:-1] + bin_edges_pct[1:]) / 2

        sample_counts_pct = []
        mean_crps_per_bin = np.full((num_bins), np.nan)

        for b in range(num_bins):
            in_bin = (variance_percentiles >= bin_edges_pct[b]) & (variance_percentiles < bin_edges_pct[b+1])
            if np.any(in_bin):  # Only plot if bin contains data
                crps_in_bin = crps_data[in_bin]
                mean_crps_per_bin[b] = np.mean(crps_in_bin) if len(crps_in_bin) > 0 else np.nan

                if len(crps_in_bin) >= threshold:  # Double-check data exists
                    a = 0.4

                    violin_parts1 = axes[k].violinplot(crps_in_bin, positions=[bin_centers_pct[b]], widths=8, showmeans=True)

                    # Set color and alpha for violin bodies
                    for pc in violin_parts1['bodies']:
                        pc.set_facecolor(color_themes[k])
                        pc.set_alpha(a)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(0.2)

                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                        if partname in violin_parts1:
                            violin_parts1[partname].set_edgecolor(color_themes[k])
                            violin_parts1[partname].set_alpha(a)  # Slightly more opaque for visibility
                    
                    # Set color for means if shown
                    if 'cmeans' in violin_parts1:
                        violin_parts1['cmeans'].set_edgecolor('black')
                        violin_parts1['cmeans'].set_alpha(1.0)

                    sample_counts_pct.append((bin_centers_pct[b], len(crps_in_bin)))
                else:
                    sample_counts_pct.append((bin_centers_pct[b], 0))
            else:
                sample_counts_pct.append((bin_centers_pct[b], 0))
        
        # Get current y-axis limits and expand the lower bound for percentile plot
        y_min, y_max = axes[k].get_ylim()
        y_range = y_max - y_min
        new_y_min = y_min - 0.18 * y_range
        axes[k].set_ylim(new_y_min, y_max)
        
        # Add sample count annotations below the violins
        for l, (bin_center, count) in enumerate(sample_counts_pct):
            if count >= 0:
                # mean crps for crps_in_bin
                axes[k].text(bin_center, new_y_min + 0.05 * y_range, f'n={count} \n CRPS={mean_crps_per_bin[l]:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)
                
        axes[k].set_title(f'{model_type} \n Mean Variance: {np.mean(variance_data):.5f}')
        axes[k].set_ylabel('CRPS')
        axes[k].grid(True, alpha=0.35, which='both', linestyle='-', linewidth=0.5)

        # ============= RAW VARIANCE VERSION =============
        # Create bins based on raw variance values
        bin_edges_raw = np.linspace(np.min(variance_data), np.max(variance_data), num_bins + 1)
        bin_centers_raw = (bin_edges_raw[:-1] + bin_edges_raw[1:]) / 2

        sample_counts_raw = []
        mean_crps_per_bin = np.full((num_bins), np.nan)
        for b in range(num_bins):
            in_bin = (variance_data >= bin_edges_raw[b]) & (variance_data < bin_edges_raw[b+1])
            if np.any(in_bin):  # Only plot if bin contains data
                crps_in_bin = crps_data[in_bin]
                mean_crps_per_bin[b] = np.mean(crps_in_bin) if len(crps_in_bin) > 0 else np.nan

                if len(crps_in_bin) > 0: # always plot all violins
                    if len(crps_in_bin) >= threshold:  # plot sufficient samples
                        a = 0.4
                    else:
                        a = 0.0

                    violin_parts2 = axes2[k].violinplot(crps_in_bin, positions=[bin_centers_raw[b]], widths=bin_centers_raw[1]-bin_centers_raw[0], showmeans=True)

                    # Set color and alpha for violin bodies
                    for pc in violin_parts2['bodies']:
                        pc.set_facecolor(color_themes[k])
                        pc.set_alpha(a)
                        pc.set_edgecolor('black')
                        pc.set_linewidth(0.2)

                    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                        if partname in violin_parts2:
                            violin_parts2[partname].set_edgecolor(color_themes[k])
                            violin_parts2[partname].set_alpha(a)  # Slightly more opaque for visibility
                    
                    # Set color for means if shown
                    if 'cmeans' in violin_parts2:
                        violin_parts2['cmeans'].set_edgecolor('black')
                        violin_parts2['cmeans'].set_alpha(1.0)
                    
                    sample_counts_raw.append((bin_centers_raw[b], len(crps_in_bin)))
                else:
                    sample_counts_raw.append((bin_centers_raw[b], 0))
            else:
                sample_counts_raw.append((bin_centers_raw[b], 0))
        
        # Get current y-axis limits and expand the lower bound for raw variance plot
        y_min2, y_max2 = axes2[k].get_ylim()
        y_range2 = y_max2 - y_min2
        new_y_min2 = y_min2 - 0.15 * y_range2
        axes2[k].set_ylim(new_y_min2, y_max2)
        
        # Add sample count annotations below the violins
        for l, (bin_center, count) in enumerate(sample_counts_raw):
            if count > 0:
                axes2[k].text(bin_center, new_y_min2 + 0.05 * y_range2, f'n={count} \n CRPS={mean_crps_per_bin[l]:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)

        axes2[k].set_title(f'{model_type} \n Mean Variance: {np.mean(variance_data):.5f}')
        axes2[k].set_ylabel('CRPS')
        axes2[k].grid(True, alpha=0.35, which='both', linestyle='-', linewidth=0.5)
        axes2[k].legend()

    # Finalize percentile plot
    axes[-1].set_xlabel('Variance over IQR Percentile')
    axes[-1].set_xlim([0, 100])
    plt.figure(fig4.number)
    plt.suptitle(f'CRPS Distribution Across Variance over IQR Percentiles {keyword}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/crps_distribution_across_variance_OIQR_percentiles_{keyword}.png', 
                format='png', dpi=250)
    plt.close()

    # Finalize raw variance plot
    axes2[-1].set_xlabel('Raw Variance over IQR')
    plt.figure(fig5.number)
    plt.suptitle(f'CRPS Distribution Across Raw Variance over IQR {keyword}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/crps_distribution_across_raw_variance_OIQR_{keyword}.png', 
                format='png', dpi=250)
    plt.close()

    # After the main violin plot, create a summary trend plot
    fig_trend, ax_trend = plt.subplots(figsize=(10, 6))

    for k, model_type in enumerate(model_types):
        if k < len(variance_all_model_types) and k < len(crps_all_model_types):
            # Calculate mean CRPS for each percentile bin
            variance_data = variance_all_model_types[k]
            crps_data = np.mean(crps_all_model_types[k], axis=0)
            variance_percentiles = stats.rankdata(variance_data, method='average') / len(variance_data) * 100
            
            bin_edges = np.linspace(0, 100, num_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            bin_means = []
            bin_stds = []
            
            for b in range(num_bins):
                in_bin = (variance_percentiles >= bin_edges[b]) & (variance_percentiles < bin_edges[b+1])
                if np.any(in_bin):
                    crps_in_bin = crps_data[in_bin]
                    if len(crps_in_bin) > 10:
                        bin_means.append(np.median(crps_in_bin))
                        bin_stds.append(np.std(crps_in_bin))
                    else:
                        bin_means.append(np.nan)
                        bin_stds.append(np.nan)
                else:
                    bin_means.append(np.nan)
                    bin_stds.append(np.nan)
            
            # Plot trend line with error bars
            # ax_trend.errorbar(bin_centers, bin_means, yerr=bin_stds, 
            #                 marker='o', linewidth=2, capsize=5, 
            #                 color=color_themes[k], label=model_type)

    ax_trend.set_xlabel('Variance over IQR Percentile')
    ax_trend.set_ylabel('Median CRPS')
    ax_trend.set_title('CRPS Trend Across Variance over IQR Percentiles')
    ax_trend.legend()
    ax_trend.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/crps_trend_across_variance_OIQR_percentiles_{keyword}.png', 
                format='png', dpi=250)



def variance_analysis_success_plot(experiments, scaling_method = None, keyword = None):
    """
    Epistemic Uncertainty Analysis : 
    - Discard plot of CRPS vs Variance across random seeds for a given experiment type, comparing across experiment types: 
        Experiments contains multiple experiment types
        For each experiment type: 
            - Load outputs from each random seed
            - Make if block for how to calculate variance: 
                If by np.var, calculate np.var across all random seeds for each output sample
            - Create discard plot of CRPS mean (y axis) binned by variance across seeds (x axis) 
            - Overlay all experiment types on one plot for comparison
    """

    exps = experiments

    color_themes = {
        0: "#3b528b", 
        1: "#019bba", 
        2: "#33c316",
        3: "#B6B309",
        }
    
    fig1, ax1 = plt.subplots(figsize = (8, 6))
    fig3, ax3 = plt.subplots(figsize = (8, 6))
    fig4, ax4 = plt.subplots(figsize = (8, 6))

    variance_all_model_types = []

    for i, (exp_type, exp_list) in enumerate(exps.items()):

        if i in [1, 3]:

            print(f'Processing experiment type: {exp_type}')

            if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv"]:
                    data_type = "OBS"
            elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "E3SM(E3SM)sv"]:
                    data_type = "E3SM"
            
            #identify lengths for accurate preallocation: 
            output_preall = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_list[0]}/{exp_list[0]}_network_SHASH_parameters.pkl')

            all_crps = np.empty((len(exp_list), len(output_preall)))
            all_mean_shash = np.empty((len(exp_list), len(output_preall)))

            for iexp, exp_name in enumerate(exp_list):
                print(f'  Processing experiment: {exp_name}')

                # Load the output and target data for this experiment
                try:
                    output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')
                except FileNotFoundError:
                    pattern = f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/exp*_{exp_name}_OOD_INFERENCE_network_SHASH_parameters.pkl'
                    matching_files = glob.glob(pattern)
                    output_preall = load_pickle(matching_files[0]) if matching_files else None

                # open crps: 
                crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

                if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                    target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
            
                elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                    target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
        
                climo_crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_climatology_values.pkl')
                
                if keyword == "scaled_IQR":
                    # Scale Climo CRPS by day of year mean: 
                    climo_crps_xr = xr.DataArray(climo_crps, coords=[target['time']], dims=["time"])
                    daily_climo_crps = climo_crps_xr.groupby('time.dayofyear').mean('time')
                    scaled_climo_crps = climo_crps_xr.groupby('time.dayofyear') / daily_climo_crps
                    scaled_climo_crps = scaled_climo_crps.sortby('time')
                    mean_climo_crps = np.mean(scaled_climo_crps)

                    # Scale CRPS by day of year as well: 
                    crps_xr = xr.DataArray(crps, coords=[target['time']], dims=["time"])
                    # scale crps by day of year mean:
                    daily_crps = crps_xr.groupby('time.dayofyear').mean('time')
                    scaled_crps = crps_xr.groupby('time.dayofyear') / daily_crps
                    scaled_crps = scaled_crps.sortby('time')
                    crps = scaled_crps.values

                # DETERMINISTIC CALCULATION METHOD: Mean of Shash
                output_SHASH = Shash(output)
                network_mean_tensor = output_SHASH.mean()

                # store mean shash values as numpy values: 
                all_mean_shash[iexp] = network_mean_tensor.numpy()

                # print(f"crps shape: {crps.shape}, mean shash shape: {network_mean_tensor.shape}")

                all_crps[iexp] = crps

            # Calculate variance across random seeds for each sample
            variance_across_seeds = np.var(all_mean_shash, axis=0)
            # print(f"shape of variance: {variance_across_seeds.shape}")
            variance_all_model_types.append(variance_across_seeds)

    # Plot Variance across all model types as scatter plot with CRPS: 
    fig, ax = plt.subplots(figsize = (10, 7))

    for i, (exp_type, exp_list) in enumerate(exps.items()):
        
        print(f'size variance all model types: {np.array(variance_all_model_types).shape}, size all_crps: {np.array(all_crps).shape}')
        ax.scatter(variance_all_model_types[i], np.mean(all_crps[i], axis=0), 
                alpha=0.3, s=10, color=color_themes[i], label=exp_type)

    ax.set_xlabel('Model Variance')
    ax.set_ylabel('CRPS')
    ax.set_title(f'CRPS vs Variance across Random Seeds | {keyword}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/CRPS_vs_Variance_SCATTER_{keyword}.png', format='png', dpi=250)
    plt.close()




    # # FIGURE 4 : CRPS vs Variance as DISCARD Plot: ------------------------------------------
    # percentiles = np.linspace(100, 0, 21)

    # plt.figure(fig4.number)
    # for iexp, exp_name in enumerate(exp_list):

    #     avg_success_ratio = np.empty([len(exp_list), len(percentiles)])
    #     avg_variance = []
    #     sample_index = np.zeros((len(variance_across_seeds), len(percentiles)))

    #     # Sort by Variance
    #     var_sorted_indices = np.argsort(variance_across_seeds)
    #     var_sorted = variance_across_seeds[var_sorted_indices]

    #     for ip, p in enumerate(percentiles):
    #         # percentage of samples to keep for each round of the loop
    #         num_to_keep = int(len(var_sorted) * p / 100)
            
    #         indices = var_sorted_indices[:num_to_keep]

    #         if len(indices) == 0:
    #             avg_success_ratio[iexp, ip] = np.nan
    #             avg_variance.append(np.nan)
    #         else:
    #             success_ratio = np.sum(all_crps[iexp][indices] < scaled_climo_crps[indices]) / len(indices)
    #             avg_success_ratio[iexp, ip] = success_ratio
    #             sample_index[:len(indices), ip] = indices

    #     if iexp == 0:
    #         ax4.plot(percentiles, avg_success_ratio[iexp], alpha = 0.65, linewidth = 2.5, color=color_themes[i], label = f'{exp_type}')
    #     else:
    #         ax4.plot(percentiles, avg_success_ratio[iexp], alpha = 0.65, linewidth = 2.5, color=color_themes[i])

    # plt.gca().invert_xaxis()
    # ax4.set_ylabel('Success Ratio')
    # ax4.set_xlabel('Model Variance Percentile (% Data Remaining)')
    # ax4.set_xlim([100, 1])
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/Success_Ratio_vs_Variance_percentile_{keyword}_DISCARD.png', format='png', dpi=250 )
    # plt.close()

    # # FIGURE 1 : Scaled SUCCESS RATIO vs Variance Binned Plot
    # # Bin the variance values
    # n_bins = 28  # You can adjust the number of bins
    # bin_edges = np.linspace(np.min(variance_across_seeds), np.max(variance_across_seeds), n_bins + 1)
    # # bin_edges = np.percentile(variance_across_seeds, np.linspace(0, 100, n_bins + 1))
    # bin_indices = np.digitize(variance_across_seeds, bin_edges) - 1  # bins are 0-indexed

    # success_ratio_per_bin = np.full((len(exp_list), n_bins), np.nan)  # Initialize with NaN
    # bin_centers = np.array([(bin_edges[b] + bin_edges[b+1]) / 2 for b in range(n_bins)])

    # plt.figure(fig1.number) 

    # for exp in range(len(exp_list)):
    #     for b in range(n_bins):
    #         in_bin = bin_indices == b
    #         if np.any(in_bin):
    #             success_ratio = np.sum(all_crps[exp][in_bin] < scaled_climo_crps[in_bin]) / len(in_bin) 
    #             success_ratio_per_bin[exp, b] = success_ratio


    #     # Create mask to filter out empty bins (NaN values)
    #     valid_mask = ~np.isnan(success_ratio_per_bin[exp])
    #     valid_bin_centers = bin_centers[valid_mask]
    #     valid_success_ratios = success_ratio_per_bin[exp, valid_mask]
        
    #     if exp == 0:
    #         ax1.plot(valid_bin_centers, valid_success_ratios, alpha=0.35, linewidth=1.5, 
    #                 color=color_themes[i])
    #         # Add invisible line with alpha=1 for legend only
    #         ax1.plot([], [], alpha=1.0, linewidth=2.5, 
    #                 color=color_themes[i], label=f"{exp_type}")
    #     else:
    #         ax1.plot(valid_bin_centers, valid_success_ratios, alpha=0.35, linewidth=1.5, 
    #                 color=color_themes[i])

    # plt.figure(fig1.number)
    # ax1.set_ylabel('Success Ratio')
    # ax1.set_xlabel('Variance')
    # ax1.set_xlim([0, 0.075])
    # # ax1.set_ylim([0.35, 1.2])
    # plt.title(f'Success Ratio (scaled) vs Variance Plot')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/scaled_success_ratio_vs_variance_{keyword}.png', format = 'png', dpi = 250)
    # plt.close()


def m2m_sample_transfer(experiments, selection_method = 'scaled_iqr_by_percentage', confidence = 20, keyword = None):
    """
    Select samples from either OBS(OBS) or E3SM(E3SM) using variety of methods: 
        - select most confident sample from each month from each random seed
        - select most confident percentage of samples from total samples from each random seed
        - select dates that are input by hand
    
    Find selected samples that were run through opposing model type (OBS->E3SM or E3SM->OBS) and compare results to original model type.
    
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
            
            try:
                output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')
            except FileNotFoundError:
                pattern = f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/exp*_{exp_name}_OOD_INFERENCE_network_SHASH_parameters.pkl'
                matching_files = glob.glob(pattern)
                output = load_pickle(matching_files[0]) if matching_files else None

            # open crps: 
            crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

                ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr = iqr_basic(output)
            
            # Load testing target data
            if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                data_type = "OBS"
                data_vars = ["tp", "skt", "z"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
                target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]

                if exp_type in ["E3SM(OBS)sv", "OBS(OBS)sv"]:
                    # scale target by day of year variance: 
                    daily_target_grouped_var = target.groupby('time.dayofyear').var('time')
                    scaled_target = target.groupby('time.dayofyear') / daily_target_grouped_var
                    scaled_target = scaled_target.sortby('time')
                    target = scaled_target
        
            elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                data_type = "E3SM"
                data_vars = ["PRECT", "TS", "Z500"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]
                target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]

                if exp_type in ["E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                    # scale target by day of year variance:
                    daily_target_grouped_var = target.groupby('time.dayofyear').var('time')
                    scaled_target = target.groupby('time.dayofyear') / daily_target_grouped_var
                    scaled_target = scaled_target.sortby('time')
                    target = scaled_target

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

            elif selection_method == 'scaled_iqr_by_percentage': # scale IQR by day of year 
                # create xarray object containing iqr and corresponding time coordinate: 
                iqr_xr = xr.DataArray(iqr, coords=[target['time']], dims=["time"])
                # scale iqr by day of year:
                daily_iqr = iqr_xr.groupby('time.dayofyear').mean('time')
                scaled_iqr = iqr_xr.groupby('time.dayofyear') / daily_iqr
                #ungroup scaled_iqr: 
                scaled_iqr = scaled_iqr.sortby('time')
                # print(f"iqr_xr.values: {iqr_xr.values}, scaled_iqr.values: {scaled_iqr.values}")
                # print(f"scaled iqr shape: {scaled_iqr.shape}")

                # select narrowest percentage of scaled IQR based on confidence level: 
                num_to_select = int(len(scaled_iqr) * (confidence / 100))
                selected_indices = np.argsort(scaled_iqr.values)[:num_to_select]
                print(f"selected {len(selected_indices)} samples based on scaled IQR by percentage")
            
            elif selection_method == 'simple_iqr_percentage':
                # select narrowest percentage of IQR based on confidence level: 
                num_to_select = int(len(iqr) * (confidence / 100))
                selected_indices = np.argsort(iqr)[:num_to_select]
                print(f"selected {len(selected_indices)} samples based on simple IQR by percentage")

            elif selection_method == 'no_scaling':
                # select narrowest percentage of IQR based on confidence level: 
                num_to_select = int(len(iqr) * (confidence / 100))
                selected_indices = np.argsort(iqr)[:num_to_select]
                print(f"selected {len(selected_indices)} samples based on no scaling IQR by percentage")
            else: 
                print(f"choose sample selection method or code another one up")

            # identify target dates for these conf samples
            selected_target_dates = target['time'][selected_indices]
            if "E3SM" in data_type: 
                selected_target_dates_exact = pd.to_datetime([
                f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}" 
                for dt in selected_target_dates.values])
                selected_input_dates_exact = pd.to_datetime([
                f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}" 
                for dt in (selected_target_dates - pd.Timedelta(days=config['databuilder']['lagtime'])).values])
            else: 
                selected_target_dates_exact = selected_target_dates
                selected_input_dates_exact = selected_target_dates - pd.Timedelta(days=config['databuilder']['lagtime'])

            # use lagtime to identify input dates for these conf samples
            lagtime = config['databuilder']['lagtime']
            selected_input_dates = selected_target_dates - pd.Timedelta(days=lagtime)

            selected_samples["output1"] = output[selected_indices]
            selected_samples["target_date1"] = selected_target_dates
            selected_samples["input_date1"] = selected_input_dates
            selected_samples["crps1"] = crps[selected_indices]
            selected_samples["input_maps1"] = input_maps.sel(time=selected_target_dates)
            if selection_method == 'scaled_iqr_by_percentage':
                selected_samples["iqr1"] = scaled_iqr[selected_indices]
            elif selection_method == 'simple_iqr_percentage' or selection_method == 'no_scaling':
                selected_samples["iqr1"] = iqr[selected_indices]

            # accumulate all selected indices from the test dataset: 
            all_selected_indices.extend(selected_indices)

            # ---- identify ENSO and MJO phase of selected samples ----------------
            ###### ENSO ########
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

            ######## MJO ########
            phase_timestamps = analysis_metrics.mjo_timestamps(data_type, config)

            # selected_mjo_phase = phase_timestamps.sel(time=selected_target_dates)
            selected_mjo_phase = phase_timestamps.sel(time=selected_target_dates_exact)
            selected_samples["mjo_phase"] = selected_mjo_phase

            input_mjo_phase = phase_timestamps.sel(time=selected_input_dates_exact)
            selected_samples["input_mjo_phase"] = input_mjo_phase

            data_from_all_seeds1[str(exp_name)] = selected_samples

        # print(f"all selected indices: {all_selected_indices}, len: {len(all_selected_indices)}")
        
        # Find corresponding samples in opposing model type: 
        if exp_type in ["OBS(OBS)"]:
            base_exp = "OBS(OBS)"
            opposing_exp = "E3SM(OBS)"
            models = ["exp189", "exp195", "exp196", "exp197", "exp198", "exp199", "exp263", "exp264", "exp265", "exp266", "exp267", "exp268"]
        elif exp_type in ["E3SM(E3SM)"]: 
            base_exp = "E3SM(E3SM)"
            opposing_exp = "OBS(E3SM)"
            models = ["exp206", "exp207", "exp208", "exp209", "exp210", "exp211", "exp212", "exp213", "exp214", "exp215", "exp216", "exp217"]
        elif exp_type in ["OBS(OBS)sv"]:
            base_exp = "OBS(OBS)sv"
            opposing_exp = "E3SM(OBS)sv"
            models = ["exp222", "exp246", "exp247", "exp248", "exp249", "exp250", "exp251", "exp252", "exp253", "exp254", "exp255", "exp256"]
        elif exp_type in ["E3SM(E3SM)sv"]:
            base_exp = "E3SM(E3SM)sv"
            opposing_exp = "OBS(E3SM)sv"
            models = ["exp223", "exp224", "exp225", "exp269", "exp270", "exp271", "exp272", "exp273", "exp274", "exp275", "exp276", "exp277"]

        # For each selected sample from original model, find corresponding sample in opposing model using input date
        # Collect all target dates from 'data_from_all_seeds[exp_name]["target_date1"]'
        all_target1_dates = []
        all_output1 = []
        all_iqr1 = []
        all_crps1 = []
        all_input1 = []
        all_enso_phases1 = []
        all_mjo_phases1 = []
        all_inputmaps1 = []
        all_input_mjo_phases1 = []

        for iexp, exp_name in enumerate(exp_list):
            all_target1_dates.extend(data_from_all_seeds1[exp_name]["target_date1"])
            all_output1.extend(data_from_all_seeds1[exp_name]["output1"])
            all_iqr1.extend(data_from_all_seeds1[exp_name]["iqr1"])
            all_crps1.extend(data_from_all_seeds1[exp_name]["crps1"])
            all_input1.extend(data_from_all_seeds1[exp_name]["input_date1"].values)
            all_enso_phases1.extend(data_from_all_seeds1[exp_name]["enso_phase"])
            all_mjo_phases1.extend(data_from_all_seeds1[exp_name]["mjo_phase"]["phase"].values.tolist())
            all_input_mjo_phases1.extend(data_from_all_seeds1[exp_name]["input_mjo_phase"]["phase"].values.tolist())
            
        for ood_model in models: 
            print(f"  Processing opposing model: {ood_model}")
            selected_samples = {}
            config = utils.get_config(ood_model)
            
            # Load the output and target data for this experiment
            try:
                output_ood = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')
            except FileNotFoundError:
                pattern = f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/exp*_{exp_name}_OOD_INFERENCE_network_SHASH_parameters.pkl'
                matching_files = glob.glob(pattern)
                output_ood = load_pickle(matching_files[0]) if matching_files else None

            # open crps: 
            crps_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{ood_model}/{ood_model}_CRPS_network_values.pkl')

            ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr_ood = iqr_basic(output_ood)
            print(f"iqr_ood shape: {iqr_ood.shape}")

            if selection_method == 'scaled_iqr_by_percentage': # scale IQR by day of year
                # create xarray object containing iqr and corresponding time coordinate: 
                iqr_ood_xr = xr.DataArray(iqr_ood, coords=[target['time']], dims=["time"])
                # scale iqr by day of year:
                daily_iqr_ood = iqr_ood_xr.groupby('time.dayofyear').mean('time')
                scaled_iqr_ood = iqr_ood_xr.groupby('time.dayofyear') / daily_iqr_ood

            elif selection_method == 'simple_iqr_percentage':
                pass
            
            elif selection_method == 'no_scaling':
                pass

            
            # Load testing target data
            if opposing_exp in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                target_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                climatology_stats_ood = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                target_ood = (target_ood['y'] - climatology_stats_ood['z'][2]) / climatology_stats_ood['z'][3]
        
            elif opposing_exp in ["E3SM-short(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                target_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                climatology_stats_ood = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                target_ood = (target_ood['y'] - climatology_stats_ood['Z500'][2]) / climatology_stats_ood['Z500'][3]

            elif opposing_exp in ["OBS(OBS)sv", "E3SM(OBS)sv"]:
                target_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                climatology_stats_ood = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                target_ood = (target_ood['y'] - climatology_stats_ood['z'][2]) / climatology_stats_ood['z'][3]
                # scale target by day of year variance: 
                daily_target_grouped_var = target_ood.groupby('time.dayofyear').var('time')
                scaled_target = target_ood.groupby('time.dayofyear') / daily_target_grouped_var
                scaled_target = scaled_target.sortby('time')
                target_ood = scaled_target
                print(f"scaled target ood shape: {target_ood.shape}")

            elif opposing_exp in ["E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                target_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                climatology_stats_ood = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                target_ood = (target_ood['y'] - climatology_stats_ood['Z500'][2]) / climatology_stats_ood['Z500'][3]
                # scale target by day of year variance: 
                daily_target_grouped_var = target_ood.groupby('time.dayofyear').var('time')
                scaled_target = target_ood.groupby('time.dayofyear') / daily_target_grouped_var
                scaled_target = scaled_target.sortby('time')
                target_ood = scaled_target
                print(f"scaled target ood shape: {target_ood.shape}")

            # use target_dates1 to identify results from other models for these conf samples
            selected_samples["output2"] = output_ood[all_selected_indices]
            selected_samples["target_date2"] = all_target1_dates
            selected_samples["input_date2"] = selected_input_dates
            selected_samples["crps2"] = crps_ood[all_selected_indices]
            selected_samples["enso_phase2"] = data_from_all_seeds1[exp_list[0]]["enso_phase"]
            selected_samples["mjo_phase2"] = data_from_all_seeds1[exp_list[0]]["mjo_phase"]
            selected_samples["input_maps2"] = data_from_all_seeds1[exp_list[0]]["input_maps1"]

            if selection_method == 'scaled_iqr_by_percentage':
                selected_samples["iqr2"] = scaled_iqr_ood[all_selected_indices]
                # FIND MOST CONFIDENT OOD FOR INPUT MAP 
                num_to_select_confident_ood = int(len(scaled_iqr_ood) * (confidence / 100))
                ood_iqr_indices = np.argsort(scaled_iqr_ood.values)
                most_confident_ood_indices = ood_iqr_indices[:num_to_select_confident_ood]
                most_confident_ood_dates = target_ood['time'][most_confident_ood_indices]
                selected_samples["confident_ood_indices"] = most_confident_ood_indices
                print(f"selected {len(selected_samples['confident_ood_indices'])} most confident OOD samples based on scaled IQR by percentage")
                confident_ood_input_maps = input_maps.sel(time=most_confident_ood_dates)
                confident_ood_crps = crps_ood[most_confident_ood_indices]

            elif selection_method == 'simple_iqr_percentage':
                selected_samples["iqr2"] = iqr_ood[all_selected_indices]
                # FIND MOST CONFIDENT OOD FOR INPUT MAP 
                num_to_select_confident_ood = int(len(iqr_ood) * (confidence / 100))
                ood_iqr_indices = np.argsort(iqr_ood)
                most_confident_ood_indices = ood_iqr_indices[:num_to_select_confident_ood]
                most_confident_ood_dates = target_ood['time'][most_confident_ood_indices]
                selected_samples["confident_ood_indices"] = most_confident_ood_indices
                print(f"selected {len(selected_samples['confident_ood_indices'])} most confident OOD samples based on simple IQR by percentage")
                confident_ood_input_maps = input_maps.sel(time=most_confident_ood_dates)
                confident_ood_crps = crps_ood[most_confident_ood_indices]

            elif selection_method == 'no_scaling':
                selected_samples["iqr2"] = iqr_ood[all_selected_indices]
                # FIND MOST CONFIDENT OOD FOR INPUT MAP 
                num_to_select_confident_ood = int(len(iqr_ood) * (confidence / 100))
                ood_iqr_indices = np.argsort(iqr_ood)
                most_confident_ood_indices = ood_iqr_indices[:num_to_select_confident_ood]
                most_confident_ood_dates = target_ood['time'][most_confident_ood_indices]
                selected_samples["confident_ood_indices"] = most_confident_ood_indices
                print(f"selected {len(selected_samples['confident_ood_indices'])} most confident OOD samples based on no scaling IQR by percentage")
                confident_ood_input_maps = input_maps.sel(time=most_confident_ood_dates)
                confident_ood_crps = crps_ood[most_confident_ood_indices]

            selected_samples["confident_ood_input_maps"] = confident_ood_input_maps
            selected_samples["confident_ood_target_dates"] = most_confident_ood_dates
            selected_samples["confident_ood_crps"] = confident_ood_crps

            data_from_all_seeds2[str(ood_model)] = selected_samples

        all_target2_dates = []
        all_output2 = []
        all_iqr2 = []
        all_crps2 = []
        all_input2 = []
        all_confident_ood_inputmaps = []
        all_confident_ood_dates = []
        all_confident_ood_crps = []

        for imod, ood_model in enumerate(models):
            target_dates = data_from_all_seeds2[ood_model]["target_date2"]
            if hasattr(target_dates, 'values'):
                all_target2_dates.extend(target_dates.values)
            else:
                all_target2_dates.extend(target_dates)
            # all_target2_dates.extend(data_from_all_seeds2[str(ood_model)]["target_date2"].values)
            all_output2.extend(data_from_all_seeds2[str(ood_model)]["output2"])
            all_iqr2.extend(data_from_all_seeds2[str(ood_model)]["iqr2"])
            all_crps2.extend(data_from_all_seeds2[str(ood_model)]["crps2"])
            all_input2.extend(data_from_all_seeds2[str(ood_model)]["input_date2"].values)
            all_confident_ood_inputmaps.extend(data_from_all_seeds2[str(ood_model)]["confident_ood_input_maps"])
            all_confident_ood_dates.extend(data_from_all_seeds2[str(ood_model)]["confident_ood_target_dates"].values)
            all_confident_ood_crps.extend(data_from_all_seeds2[str(ood_model)]["confident_ood_crps"])
    
        # ---- PLOTTING ---------------------------------------------------------

        # SUMMARY PLOT: 
        # 4 panels: (1) shash curves (2) MJO phase distributions (3) ENSO phase distribution (4) target value distribution
        # (1) shash curves from all output1 in one color, and all output2 in another color
        x = np.linspace(-5, 5, 100)
        all_output1 = np.array(all_output1)
        all_output2 = np.array(all_output2)
        dist1 = Shash(all_output1)
        dist2 = Shash(all_output2)
        p1 = dist1.prob(x).numpy()
        p2 = dist2.prob(x).numpy()

        # (2) MJO phase distribution from selected_target_dates1
        mjo_baseline_frequencies = analysis.analysis_metrics.baseline_mjo_frequencies(data_type)
        # print(f"mjo baseline frequencies: {mjo_baseline_frequencies}")
        if "OBS" in data_type:
            mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_phase_frequencies_ERA5_1940_2023.pkl')
        elif "E3SM" in data_type:
            mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_frequencies_E3SM_1850_2014.pkl')

        ### MJO phase distribution VALIDATION DAY - CHECK EACH SEED SEPARATELY

        # Create a figure with subplots for each random seed
        n_seeds = len(exp_list)
        fig, axes = plt.subplots(2, (n_seeds + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()

        phases = np.arange(0, 9)
        phase_labels = [str(i) for i in phases]
        bar_width = 0.8

        # Plot for each seed
        for idx, exp_name in enumerate(exp_list):
            ax = axes[idx]
            
            # Get MJO phases for this seed
            seed_mjo_phases = data_from_all_seeds1[exp_name]["mjo_phase"].phase.values
            
            # Count each phase
            counts = np.bincount(seed_mjo_phases, minlength=9)
            total = counts.sum()
            densities = counts / total
            
            # Bar plot for density
            bars = ax.bar(phases, densities, width=bar_width, color="#7d4b94", alpha=0.7, 
                        edgecolor='black', label='Selected Samples' if idx == 0 else None)
            
            # Reference lines for each phase
            for i, freq in enumerate(mjo_ref_frequencies_all_data):
                ax.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, 
                        color="#3C3B3B", linewidth=2, linestyle='-', 
                        label='Reference' if (idx == 0 and i == 0) else None)
            
            ax.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
            ax.set_xticks(phases)
            ax.set_xticklabels(phase_labels, fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.set_title(f'{exp_name}\n(n={total})', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide extra subplots if odd number of seeds
        for idx in range(n_seeds, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'MJO Phase Distribution by Random Seed | {exp_type} {keyword} | {confidence}% Confidence', 
                    fontsize=12, y=1.00)
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_by_seed_{exp_type}_{confidence}_{keyword}.png', 
                    format='png', dpi=250, bbox_inches='tight')

         ### MJO phase distribution INITIALIZATION DAY - CHECK EACH SEED SEPARATELY

        # Create a figure with subplots for each random seed
        n_seeds = len(exp_list)
        fig, axes = plt.subplots(2, (n_seeds + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()

        phases = np.arange(0, 9)
        phase_labels = [str(i) for i in phases]
        bar_width = 0.8

        # Plot for each seed
        for idx, exp_name in enumerate(exp_list):
            ax = axes[idx]
            
            # Get MJO phases for this seed
            seed_input_mjo_phases = data_from_all_seeds1[exp_name]["input_mjo_phase"].phase.values
            
            # Count each phase
            counts = np.bincount(seed_input_mjo_phases, minlength=9)
            total = counts.sum()
            densities = counts / total
            
            # Bar plot for density
            bars = ax.bar(phases, densities, width=bar_width, color="#564b94", alpha=0.7, 
                        edgecolor='black', label='Selected Samples' if idx == 0 else None)
            
            # Reference lines for each phase
            for i, freq in enumerate(mjo_ref_frequencies_all_data):
                ax.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, 
                        color="#3C3B3B", linewidth=2, linestyle='-', 
                        label='Reference' if (idx == 0 and i == 0) else None)
            
            ax.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
            ax.set_xticks(phases)
            ax.set_xticklabels(phase_labels, fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.set_title(f'{exp_name}\n(n={total})', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide extra subplots if odd number of seeds
        for idx in range(n_seeds, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'MJO Phase Distribution by Random Seed on Initialization Day | {exp_type} {keyword} | {confidence}% Confidence', 
                    fontsize=12, y=1.00)
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_by_seed_{exp_type}_{confidence}_INIT_day_{keyword}.png', 
                    format='png', dpi=250, bbox_inches='tight')


        # (3) ENSO phase distribution from selected_target_dates1
        all_enso_phases = []
        for exp_name in exp_list:
            all_enso_phases.extend(data_from_all_seeds1[exp_name]["enso_phase"])
        all_enso_phases = np.array(all_enso_phases)
        # print(f"all enso phases: {all_enso_phases}, type: {type(all_enso_phases)}, len: {len(all_enso_phases)}")
        # calculate frequency of enso phase relative to prevalence in total target dataset
        enso_baseline_frequencies = analysis.analysis_metrics.baseline_enso_frequencies(data_type)
        # print(f"ENSO baseline frequencies: {enso_baseline_frequencies}")

        # print(f'all target 1 values type: {type(all_target1_dates)}, examp: {all_target1_dates[0]}, type examp: {type(all_target1_dates[0])}')
        # (4) Target value distribution from selected_target_dates1
        if "OBS" in data_type:
            # Extract numpy.datetime64 values from DataArrays
            date_values = [np.datetime64(date_da.values) for date_da in all_target1_dates]
            # print(f" date_values max date: {np.max(date_values)}, min date: {np.min(date_values)}")
            selected_target_values = target.sel(time=date_values)
        elif "E3SM" in data_type:
            date_values = [date_da.values.item() for date_da in all_target1_dates]
            selected_target_values = target.sel(time=date_values)
        
        # Mean IQR for all output1
        mean_iqr1 = np.mean(all_iqr1)
        print(f"Mean IQR for all output1: {mean_iqr1}")
        mean_iqr2 = np.mean(all_iqr2)
        print(f"Mean IQR for all output2: {mean_iqr2}")

        # Mean CRPS for all output1
        mean_crps1 = np.mean(all_crps1)
        print(f"Mean CRPS for all output1: {mean_crps1}")
        mean_crps2 = np.mean(all_crps2)
        print(f"Mean CRPS for all output2: {mean_crps2}")

        # Plot 1: SHASH Curves
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.hist(
            climatology_data, x, density=True, color="silver", alpha=0.75, label="climatology"
        )
        ax1.plot(x, p1[:, 0], alpha = 0.4, linewidth=0.5, label=f"{base_exp}\nIQR: {mean_iqr1:.2f}\nCRPS: {mean_crps1:.2f}", color='#46039f')
        ax1.plot(x, p2[:, 0], alpha = 0.4, linewidth=0.5, label=f"{opposing_exp}\nIQR: {mean_iqr2:.2f}\nCRPS: {mean_crps2:.2f}", color='#bd3786')
        ax1.plot(x, p1, alpha= 0.4, linewidth=0.5, color='#46039f')
        ax1.plot(x, p2, alpha= 0.4, linewidth=0.5, color='#bd3786')
        ax1.set_xlabel(f'Standardized {target_var} Anomaly')
        ax1.set_ylabel('Probability Density')
        ax1.set_title(f'SHASH Curves | {confidence}% Most Confident | {data_type} {keyword}')  # Added data_type to title
        ax1.set_ylim([0, 0.8])
        ax1.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/shash_curves_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)
        plt.show()

        # Plot 2: MJO Phase Distribution - VERIFICATION DAY
        # all_mjo_phases = []
        # for exp_name in exp_list:
        #     all_mjo_phases.extend(data_from_all_seeds1[exp_name]["mjo_phase"]["phase"].values.tolist())
        # all_mjo_phases = np.array(all_mjo_phases)
        # counts = np.bincount(all_mjo_phases, minlength=9)
        # total = counts.sum()
        # densities = counts / total

        # phases = np.arange(0, 9)
        # phase_labels = [str(i) for i in phases]

        # fig, ax = plt.subplots(figsize=(8, 6))
        # bar_width = 0.8

        # # Bar plot for density
        # bars = ax.bar(phases, densities, width=bar_width, color="#7d4b94", alpha=0.7, edgecolor='black', label='Selected Samples')

        # # Reference lines for each phase
        # for i, freq in enumerate(mjo_ref_frequencies_all_data):
        #     # Draw a horizontal line across the width of the bar for phase i
        #     ax.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, color="#3C3B3B", linewidth=2, linestyle='-', label='Reference' if i==0 else None)

        # ax.set_xticks(phases)
        # ax.set_xticklabels(phase_labels)
        # ax.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
        # ax.set_xlabel('MJO Phase')
        # ax.set_ylabel('Density')
        # ax.set_title('MJO Phase Distribution | {keyword}')
        # handles, labels = ax.get_legend_handles_labels()
        # # Only show one legend entry for the reference lines
        # if 'Reference' in labels:
        #     idx = labels.index('Reference')
        #     ax.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
        # else:
        #     ax.legend()
        # plt.tight_layout()
        # plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)
        # plt.show()

        # Plot 2: MJO Phase Distribution - INITIALIZATION DAY
        all_input_mjo_phases = []
        for exp_name in exp_list:
            all_input_mjo_phases.extend(data_from_all_seeds1[exp_name]["input_mjo_phase"]["phase"].values.tolist())
        all_input_mjo_phases = np.array(all_input_mjo_phases)
        counts = np.bincount(all_input_mjo_phases, minlength=9)
        total = counts.sum()
        densities = counts / total

        phases = np.arange(0, 9)
        phase_labels = [str(i) for i in phases]

        fig, ax = plt.subplots(figsize=(8, 6))
        bar_width = 0.8

        # Bar plot for density
        bars = ax.bar(phases, densities, width=bar_width, color="#534b94", alpha=0.7, edgecolor='black', label='Selected Samples')

        # Reference lines for each phase
        for i, freq in enumerate(mjo_ref_frequencies_all_data):
            # Draw a horizontal line across the width of the bar for phase i
            ax.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, color="#3C3B3B", linewidth=2, linestyle='-', label='Reference' if i==0 else None)

        ax.set_xticks(phases)
        ax.set_xticklabels(phase_labels)
        ax.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
        ax.set_xlabel('MJO Phase')
        ax.set_ylabel('Density')
        ax.set_title(f'MJO Phase Distribution {keyword}')
        handles, labels = ax.get_legend_handles_labels()
        # Only show one legend entry for the reference lines
        if 'Reference' in labels:
            idx = labels.index('Reference')
            ax.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
        else:
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_{exp_type}_{confidence}_INIT_day_{keyword}.png', format='png', dpi=250)
        plt.show()

        # PLOT 9: Histogram check of ENSO index strength
        # Plot actual values of ENSO index for ALL samples and also for selected samples as histogram of "El Nino", "La Nina"
        if data_type == "E3SM":
            print("Data source is E3SM - daily ENSO data")
            Nino34 = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/ENSO_Data/E3SM/ENSO_ne30pg2_HighRes/nino.member0201_daily_linterp_shifted.nc')
            Nino34 = Nino34.nino34
            nino34_index = Nino34.sel(time=slice(str(config["databuilder"]["input_years"][0]), str(config["databuilder"]["input_years"][1])))

        elif data_type == "OBS":
            print("Data source is ERA5 - daily ENSO data")            
            Nino34 = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/ENSO_Data/OBS/nino34.long.anom_daily_linterp_shifted.nc')
            Nino34 = Nino34.value
            # full_time = xr.cftime_range(start='1850-01-01', end='2024-12-31', freq='D', calendar='noleap')
            # Nino34 = Nino34.reindex(time=full_time)
            nino34_index = Nino34.sel(time=slice(str(config["databuilder"]["input_years"][0]), str(config["databuilder"]["input_years"][1])))
        
        # Collect index values for histogram of all samples:
        all_enso_values = {'El Nino': [], 'La Nina': [], 'Neutral': []}

        def safe_sel_nino34(nino34_index, date, data_type):
            """
            Selects the Nino3.4 index value for a given date, handling E3SM vs OBS time formats.
            Returns np.nan if not found.
            """
            try:
                if data_type == "E3SM":
                    # Convert any date type to string 'YYYY-MM-DD'
                    if hasattr(date, 'strftime'):
                        date_str = date.strftime('%Y-%m-%d')
                    elif hasattr(date, 'year') and hasattr(date, 'month') and hasattr(date, 'day'):
                        date_str = f"{date.year:04d}-{date.month:02d}-{date.day:02d}"
                    elif isinstance(date, (np.datetime64, pd.Timestamp)):
                        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                    else:
                        date_str = str(date)[:10]
                    return nino34_index.sel(time=date_str).values.item()
                else:
                    # Use date directly for OBS/ERA5
                    return nino34_index.sel(time=date).values.item()
            except Exception as e:
                # print(f"Date {date} not found in nino34_index, skipping. ({e})")
                return np.nan

        for date in target['time'].values:
            # For ENSO phase assignment, compare as numpy.datetime64
            if date in enso_dates_pkl['El Nino']:
                phase = 'El Nino'
            elif date in enso_dates_pkl['La Nina']:
                phase = 'La Nina'
            else:
                phase = 'Neutral'
            nino_val = safe_sel_nino34(nino34_index, date, data_type)
            all_enso_values[phase].append(nino_val)

        # collect index values for histogram from most confident selected target dates: 
        selected_enso_values = {'El Nino': [], 'La Nina': [], 'Neutral': []}
        for exp_name in exp_list:
            selected_target_dates = data_from_all_seeds1[exp_name]["target_date1"]
            for date in selected_target_dates.values:
                if date in enso_dates_pkl['El Nino']:
                    phase = 'El Nino'
                elif date in enso_dates_pkl['La Nina']:
                    phase = 'La Nina'
                else:
                    phase = 'Neutral'
                nino_val = safe_sel_nino34(nino34_index, date, data_type)
                selected_enso_values[phase].append(nino_val)
                
        # print(f"Number of selected El Nino samples: {len(selected_enso_values['El Nino'])}, La Nina samples: {len(selected_enso_values['La Nina'])}, Neutral samples: {len(selected_enso_values['Neutral'])}")
        # print(f"Number of all El Nino samples: {len(all_enso_values['El Nino'])}, La  Nina samples: {len(all_enso_values['La Nina'])}, Neutral samples: {len(all_enso_values['Neutral'])}")
        # print(f" type of selected_enso_values['El Nino']: {type(selected_enso_values['El Nino'])}, type of first element: {type(selected_enso_values['El Nino'][0])}")
        # print(f"first few selected El Nino values: {selected_enso_values['El Nino'][:5]}")
        # Plot histograms
        plt.figure(figsize = (9, 6))
        bins = np.linspace(-5, 5, 100)
        plt.hist(all_enso_values['El Nino'], bins=bins, alpha=0.3, label=f'All Samples - El Nino (N = {len(all_enso_values["El Nino"])})', histtype = 'barstacked', color="#482878", density=True)
        plt.hist(all_enso_values['La Nina'], bins=bins, alpha=0.3, label=f'All Samples - La Nina (N = {len(all_enso_values["La Nina"])})', histtype = 'barstacked', color="#26828e", density=True)
        plt.hist(all_enso_values['Neutral'], bins=bins, alpha=0.3, label=f'All Samples - Neutral (N = {len(all_enso_values["Neutral"])})', histtype = 'barstacked', color="#b5de2b", density=True)
        plt.xlabel('Nino3.4 Index Value')
        plt.ylabel('Density')
        plt.title(f'ENSO Index Value Distribution Across all {data_type} Data')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_index_value_distribution_{exp_type}_ALL_data_{keyword}.png', format='png', dpi=250)
        
        plt.figure(figsize = (9, 6))
        bins = np.linspace(-5, 5, 100)
        plt.hist(selected_enso_values['El Nino'], bins=bins, alpha=0.7, label=f'Selected Samples - El Nino (N = {len(selected_enso_values["El Nino"])})', histtype = 'barstacked', color="#482878", density=True)
        plt.hist(selected_enso_values['La Nina'], bins=bins, alpha=0.7, label=f'Selected Samples - La Nina (N = {len(selected_enso_values["La Nina"])})', histtype = 'barstacked', color="#26828e", density=True)
        plt.hist(selected_enso_values['Neutral'], bins=bins, alpha=0.7, label=f'Selected Samples - Neutral (N = {len(selected_enso_values["Neutral"])})', histtype = 'barstacked', color="#b5de2b", density=True)
        plt.xlabel('Nino3.4 Index Value')
        plt.ylabel('Density')
        plt.title(f'ENSO Index Value Distribution For {confidence}% Most Confident | {data_type} {keyword}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_index_value_distribution_{exp_type}_{confidence}_most_confident_{keyword}.png', format='png', dpi=250)
        
        # Plot 3: ENSO Phase Distribution - RATIO
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        bin_edges = np.array([-0.5, 0.5, 1.5, 2.5])
        bin_centers = np.array([0, 1, 2])
        bar_width = 0.4

        sum_total_phases = len(selected_enso_values['El Nino']) + len(selected_enso_values['La Nina']) + len(selected_enso_values['Neutral'])
        enso_phase_dist = [len(selected_enso_values['El Nino']) / sum_total_phases, 
                        len(selected_enso_values['La Nina']) / sum_total_phases, 
                        len(selected_enso_values['Neutral']) / sum_total_phases]

        bars = ax3.bar(bin_centers, enso_phase_dist, width=bar_width, color='#fb9f3a', alpha=0.7, edgecolor='black')

        # Reference lines for each ENSO phase
        enso_phases = ['El Nino', 'La Nina', 'Neutral']
        for i, phase in enumerate(enso_phases):
            freq = enso_baseline_frequencies[phase]
            # Draw a horizontal line across the width of the bar for phase i
            ax3.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, 
                    color="#3C3B3B", linewidth=2, linestyle='-', 
                    label='Reference' if i==0 else None)

        ax3.set_ylim([0, max(max(enso_phase_dist), max(enso_baseline_frequencies.values())) * 1.15])
        ax3.set_xticks(bin_centers)
        ax3.set_xticklabels(['El Nino', 'La Nina', 'Neutral'])
        ax3.set_xlabel('ENSO Phase')
        ax3.set_ylabel('Density')
        ax3.set_title(f'ENSO Phase Distribution | {confidence}% Most Confident | {data_type} {keyword}')

        # Legend with only one entry for reference lines
        handles, labels = ax3.get_legend_handles_labels()
        if 'Reference' in labels:
            idx = labels.index('Reference')
            ax3.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
        else:
            ax3.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_phase_distribution_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)

## -----------------
## -----------------
        # (3) ENSO phase distribution - CHECK EACH SEED SEPARATELY
        all_enso_phases = []
        for exp_name in exp_list:
            all_enso_phases.extend(data_from_all_seeds1[exp_name]["enso_phase"])
        all_enso_phases = np.array(all_enso_phases)

        # Get baseline frequencies
        enso_baseline_frequencies = analysis.analysis_metrics.baseline_enso_frequencies(data_type)
        print(f"ENSO baseline frequencies: {enso_baseline_frequencies}")

        # Create a figure with subplots for each random seed
        n_seeds = len(exp_list)
        fig, axes = plt.subplots(2, (n_seeds + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()

        # Also store aggregate data for comparison
        all_seed_ratios = {'El Nino': [], 'La Nina': [], 'Neutral': []}

        for idx, exp_name in enumerate(exp_list):
            ax = axes[idx]
            
            # Get ENSO phases for this seed
            seed_enso_phases = np.array(data_from_all_seeds1[exp_name]["enso_phase"])
            
            # Count each phase
            n_el_nino = np.sum(seed_enso_phases == "EN")
            n_la_nina = np.sum(seed_enso_phases == "LN")
            n_neutral = np.sum(seed_enso_phases == "N")
            total = len(seed_enso_phases)
            
            # Calculate ratios
            ratios = [n_el_nino / total, n_la_nina / total, n_neutral / total]
            all_seed_ratios['El Nino'].append(ratios[0])
            all_seed_ratios['La Nina'].append(ratios[1])
            all_seed_ratios['Neutral'].append(ratios[2])
            
            # Plot
            bin_centers = np.array([0, 1, 2])
            bar_width = 0.4
            
            bars = ax.bar(bin_centers, ratios, width=bar_width, color='#fb9f3a', alpha=0.7, edgecolor='black')
            
            # Reference lines for each ENSO phase
            enso_phases = ['El Nino', 'La Nina', 'Neutral']
            for i, phase in enumerate(enso_phases):
                freq = enso_baseline_frequencies[phase]
                ax.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, 
                        color="#3C3B3B", linewidth=2, linestyle='-', 
                        label='Reference' if i==0 else None)
            
            ax.set_ylim([0, max(max(ratios), max(enso_baseline_frequencies.values())) * 1.15])
            ax.set_xticks(bin_centers)
            ax.set_xticklabels(['EN', 'LN', 'N'], fontsize=8)
            ax.set_ylabel('Ratio', fontsize=8)
            ax.set_title(f'{exp_name}\n(n={total})', fontsize=9)
            if "OBS" in data_type:
                ax.set_ylim([0, 0.6])
            elif "E3SM" in data_type:
                ax.set_ylim([0, 0.5])
            
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide extra subplots if odd number of seeds
        for idx in range(n_seeds, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'ENSO Phase Distribution by Random Seed | {exp_type} {keyword} | {confidence}% Confidence', 
                    fontsize=12, y=1.00)
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_frequency_by_seed_{exp_type}_{confidence}_{keyword}.png', 
                    format='png', dpi=250, bbox_inches='tight')
        plt.show()

 ## -----------------
 ## -----------------
 
        # Plot 4: Target Variable Anomaly Distribution
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.hist(selected_target_values, bins=20, density=True, color='#0d0887', alpha=0.7, edgecolor='black')
        ax4.set_xlabel(f'Standardized {target_var} Anomaly')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Target {target_var} Anomaly Distribution | {confidence}% Most Confident | {data_type} {keyword}')  # Added data_type to title
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/target_anomaly_distribution_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)

        # Plot 5: CRPS Distribution
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        shared_bins = np.linspace(min(min(all_crps1), min(all_crps2)), max(max(all_crps1), max(all_crps2)), 20)
        ax5.hist(all_crps1, bins=shared_bins, density=True, color='#2a788e', alpha=0.7, edgecolor='black', label=base_exp)
        ax5.hist(all_crps2, bins=shared_bins, density=True, color='#7ad151', alpha=0.7, edgecolor='black', label=opposing_exp)
        ax5.set_xlabel('CRPS')
        ax5.set_ylabel('Density')
        ax5.set_title(f'CRPS Distribution | {confidence}% Most Confident | {data_type} {keyword}')  # Added data_type to title
        ax5.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/crps_distribution_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)
        plt.show()

        # Plot 6: check temporal distribution of most confident samples by month: 
        if "E3SM" in data_type:
            date_values = [date_da.values.item() if hasattr(date_da.values, 'item') else date_da.values for date_da in all_target1_dates]
            date_array = xr.DataArray(date_values, dims=['time'])
            months = date_array.dt.month.values.tolist()
        elif "OBS" in data_type:
            date_values = [np.datetime64(date_da.values) for date_da in all_target1_dates]
            # Convert to numpy array and extract months using numpy
            date_array = np.array(date_values)
            months = [date.astype('datetime64[M]').astype(int) % 12 + 1 for date in date_array]
        month_names = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
        month_counts = [months.count(m) for m in [1, 2, 3, 10, 11, 12]]
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        ax6.bar(month_names, month_counts, color="#0d7e13", alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Number of Selected Samples')
        ax6.set_title(f'Temporal Distribution of Selected Samples by Month | {confidence}% Most Confident | {data_type} {keyword}')  # Added data_type to title
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/temporal_distribution_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)

        # Plot 7: Plot mean input maps from dates of interest: 
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
        variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
        cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
        vmin_list = np.zeros(3)
        vmax_list = np.zeros(3)

        for i in range(3):
            # Calculate mean input map for variable (prect, temp, Z)
            input_maps_var = []
            for exp_name in exp_list:
                input_maps_var.append(data_from_all_seeds1[exp_name]["input_maps1"].sel(channel=i))

            input_maps_var = xr.concat(input_maps_var, dim='time')
            mean_input_map = input_maps_var.mean(dim='time')

            # vmin_list[i] = mean_input_map.min()
            # vmax_list[i] = -1 * (mean_input_map.min()) 
            # if i == 1: 
            #     vmin_list[i] = -1
            #     vmax_list[i] = 1

            vmin_list = [-0.3, -1, -0.6]
            vmax_list = [0.3, 1, 0.6]

            im = ax[i].pcolormesh(
                mean_input_map['lon'],
                mean_input_map['lat'],
                mean_input_map,
                cmap=cmap_list[i],
                vmin=vmin_list[i],
                vmax=vmax_list[i],
                transform=ccrs.PlateCarree(central_longitude=0)
            )
            ax[i].coastlines()
            ax[i].set_title(f'Mean Input Map: {variable_names[i]} | {confidence}% Most Confident | {data_type} {keyword}')
            plt.colorbar(im, ax=ax[i], orientation='horizontal', pad=0.05, label=f'{variable_names[i]} Anomaly')
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mean_input_maps_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)

        # PLOT Mean Input Maps for Most Confident OOD Samples - compare SST pattern! 
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
        variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
        cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
        vmin_list = np.zeros(3)
        vmax_list = np.zeros(3)

        for i in range(3):
            # Calculate mean input map for variable (prect, temp, Z)
            input_maps_var = []
            for ood_model in models:
                input_maps_var.append(data_from_all_seeds2[ood_model]["confident_ood_input_maps"].sel(channel=i))

            input_maps_var = xr.concat(input_maps_var, dim='time')
            mean_input_map = input_maps_var.mean(dim='time')

            # vmin_list[i] = mean_input_map.min()
            # vmax_list[i] = -1 * (mean_input_map.min()) 
            # if i == 1: 
            #     vmin_list[i] = -1
            #     vmax_list[i] = 1

            vmin_list = [-0.3, -1, -0.6]
            vmax_list = [0.3, 1, 0.6]

            im = ax[i].pcolormesh(
                mean_input_map['lon'],
                mean_input_map['lat'],
                mean_input_map,
                cmap=cmap_list[i],
                vmin=vmin_list[i],
                vmax=vmax_list[i],
                transform=ccrs.PlateCarree(central_longitude=0)
            )
            ax[i].coastlines()
            ax[i].set_title(f'Mean Input Map: {variable_names[i]} | Most Confident OOD Samples | {data_type} {keyword}')
            plt.colorbar(im, ax=ax[i], orientation='horizontal', pad=0.05, label=f'{variable_names[i]} Anomaly')
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mean_input_maps_{variable_names[i]}_OOD_most_confident_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)

        # PLOT DIFFERENCE MAP BETWEEN OBS(OBS) MOST CONFIDENT AND E3SM(OBS) MOST CONFIDENT SAMPLES: 
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
        variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']

        cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
        vmin_list = np.zeros(3)
        vmax_list = np.zeros(3)
        diff_input_maps = []
        for i in range(3):
            # Calculate mean input map for variable (prect, temp, Z) for OBS(OBS)
            input_maps_var1 = []
            for exp_name in exp_list:
                input_maps_var1.append(data_from_all_seeds1[exp_name]["input_maps1"].sel(channel=i))
            input_maps_var1 = xr.concat(input_maps_var1, dim='time')
            mean_input_map1 = input_maps_var1.mean(dim='time')

            # Calculate mean input map for variable (prect, temp, Z) for E3SM(OBS)
            input_maps_var2 = []
            for ood_model in models:
                input_maps_var2.append(data_from_all_seeds2[ood_model]["confident_ood_input_maps"].sel(channel=i))
            input_maps_var2 = xr.concat(input_maps_var2, dim='time')
            mean_input_map2 = input_maps_var2.mean(dim='time')

            # Compare input dates from most confident samples between OBS(OBS) and E3SM(OBS):
            # are they the same dates? how much overlap? 
            overlap_dates = np.intersect1d(
                np.concatenate([data_from_all_seeds1[exp_name]["target_date1"].values for exp_name in exp_list]),
                np.concatenate([data_from_all_seeds2[ood_model]["confident_ood_target_dates"].values for ood_model in models])
            )
            print(f"Number of overlapping target dates between OBS(OBS) and E3SM(OBS) most confident samples: {len(overlap_dates)}")
            print(f"Length of OBS(OBS) most confident samples: {len(np.concatenate([data_from_all_seeds1[exp_name]['target_date1'].values for exp_name in exp_list]))}")
            print(f"Length of E3SM(OBS) most confident samples: {len(np.concatenate([data_from_all_seeds2[ood_model]['confident_ood_target_dates'].values for ood_model in models]))}")
            print(f"CRPS of overlapping dates in OBS(OBS): {np.mean([data_from_all_seeds1[exp_name]['crps1'] for exp_name in exp_list])}")
            print(f"CRPS of overlapping dates in E3SM(OBS): {np.mean([data_from_all_seeds2[ood_model]['confident_ood_crps'] for ood_model in models])}")
            
            # print(f"Overlapping dates: {overlap_dates}")

            # Calculate difference
            diff_map = mean_input_map1 - mean_input_map2
            # diff_input_maps.append(diff_map)

            abs_max = np.max(np.abs(diff_map))
            vmin = -abs_max
            vmax = abs_max 

            im = ax[i].pcolormesh(
                diff_map['lon'],
                diff_map['lat'],
                diff_map,
                cmap=cmap_list[i],
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(central_longitude=0)
            )
            ax[i].coastlines()
            ax[i].set_title(f'Difference in Mean Input Map: {variable_names[i]} \n (IN-dist) - (OO-dist) \n {confidence}% Most Confident | {data_type} {keyword}')
            plt.colorbar(im, ax=ax[i], orientation='horizontal', pad=0.05, label=f'{variable_names[i]} Anomaly Difference')
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/difference_mean_input_maps_OBS_vs_E3SM_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)

        # Plot 8 : Plot mean input maps for each ENSO phase from most confident samples: 
        enso_phases = ['EN', 'LN', 'N']
        # select temperature input maps for each enso phase (channel = 1):
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
        for iphase, phase in enumerate(enso_phases):
            input_maps_var = []
            for exp_name in exp_list:
                phase_mask = np.array(data_from_all_seeds1[exp_name]["enso_phase"]) == phase
                input_maps_var.append(data_from_all_seeds1[exp_name]["input_maps1"].sel(channel=1).isel(time=phase_mask)) # channel 1 = temp (skt/TS)

            input_maps_var = xr.concat(input_maps_var, dim='time')
            if len(input_maps_var['time']) == 0:
                print(f"No samples found for ENSO phase: {phase} in variable {variable_names[i]}")
                continue
            mean_input_map = input_maps_var.mean(dim='time')

            abs_max = np.max(np.abs(mean_input_map))
            vmin = -abs_max
            vmax = abs_max 

            im = ax[iphase].pcolormesh(
                mean_input_map['lon'],
                mean_input_map['lat'],
                mean_input_map,
                cmap=cmap_list[1],
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(central_longitude=0)
            )
            ax[iphase].coastlines()
            ax[iphase].set_title(f'Mean Input Map: {variable_names[1]} | ENSO: {phase} | {confidence}% Most Confident | {data_type} {keyword}')
            plt.colorbar(im, ax=ax[iphase], orientation='horizontal', pad=0.05, label=f'{variable_names[1]} Anomaly')
            plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mean_input_maps_{variable_names[1]}_ENSO_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)


def extract_datetime64(date):
    import numpy as np
    import pandas as pd

    # If it's an xarray DataArray, get the scalar value
    if hasattr(date, 'values'):
        date = date.values
    # If it's still an array, get the item
    if hasattr(date, 'item'):
        date = date.item()
    # If it's a numpy integer type (np.int64, np.int32, etc.)
    if isinstance(date, np.integer):
        # Try to interpret as YYYYMMDD if 8 digits
        val_str = str(date)
        if len(val_str) == 8 and val_str.startswith('20'):
            return np.datetime64(f"{val_str[:4]}-{val_str[4:6]}-{val_str[6:8]}")
        else:
            raise ValueError(f"Cannot convert numpy integer {date} to np.datetime64. Please check your date format.")
    # If it's a string, convert to np.datetime64
    if isinstance(date, str):
        return np.datetime64(date)
    # If it's already np.datetime64, return as is
    if isinstance(date, np.datetime64):
        return date
    # If it's a pandas Timestamp
    if isinstance(date, pd.Timestamp):
        return np.datetime64(date)
    # If it's a datetime.datetime
    if hasattr(date, 'year') and hasattr(date, 'month') and hasattr(date, 'day'):
        return np.datetime64(f"{date.year:04d}-{date.month:02d}-{date.day:02d}")
    # Otherwise, try direct conversion
    return np.datetime64(date, 'ns')

def m2m_sample_transfer_individual(experiments, selection_method = None, confidence = 20, keyword = None):
    """
    For either OBS(OBS) or E3SM(E3SM) experiments, identify samples in the IN-DISTRIBUTION model has low CRPS (lower quartile) AND 
    where the OUT-OF-DISTRIBUTION model has high CRPS (upper quartile). From these samples, select 6 to plot SHASH curves + input maps.
    Repeat for each random seed experiment.
    Aggregate across all random seeds and plot summary statistics:
    - SHASH curves for both models
    - MJO phase distribution
    - ENSO phase distribution
    - Target value distribution
    - CRPS distribution
    - Temporal distribution of selected samples by month
    - Mean input maps from dates of interest
    """

    exps = experiments
    exp_type_names = list(exps.keys())
    exp_types_str = ', '.join(exp_type_names)
    
    for exp_type, exp_list in exps.items():
        print(f'Processing experiment type: {exp_type}')
        ID_data_from_all_seeds = {}
        OOD_data_from_all_seeds = {}
        ID_all_selected_indices = []

        for exp_name in exp_list:
            print(f'  Processing experiment: {exp_name}')
            ID_selected_samples = {}
            config = utils.get_config(exp_name)
            
            # Load the output and target data for this experiment
            output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')

            # open crps: 
            crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

                ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr = iqr_basic(output)
            
            # Load testing target data
            if exp_type in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                data_type = "OBS"
                data_vars = ["tp", "skt", "z"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
                target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
        
            elif exp_type in ["E3SM-short(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                data_type = "E3SM"
                data_vars = ["PRECT", "TS", "Z500"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]
                target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]

            # ID SAMPLE SELECTION LOGIC: 
            # From each Random Seed: Identify where BOTH ID OUTPUT values had 25% lowest CRPS
            if selection_method == "high_low_crps":
                crps_threshold_low = np.percentile(crps, confidence)  # Lower quartile
                ID_selected_indices = np.where(crps <= crps_threshold_low)[0]

            # scale IQR by day of year 
            # create xarray object containing iqr and corresponding time coordinate: 
            iqr_xr = xr.DataArray(iqr, coords=[target['time']], dims=["time"])
            # scale iqr by day of year:
            daily_iqr = iqr_xr.groupby('time.dayofyear').mean('time')
            scaled_iqr = iqr_xr.groupby('time.dayofyear') / daily_iqr
            #ungroup scaled_iqr: 
            scaled_iqr = scaled_iqr.sortby('time')
    
            # identify target dates for these conf samples
            ID_selected_target_dates = target['time'][ID_selected_indices]
            if "E3SM" in data_type: 
                ID_selected_target_dates_exact = pd.to_datetime([
                f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}" 
                for dt in ID_selected_target_dates.values])
            else: 
                ID_selected_target_dates_exact = ID_selected_target_dates

            # use lagtime to identify input dates for these conf samples
            lagtime = config['databuilder']['lagtime']
            ID_selected_input_dates = ID_selected_target_dates - pd.Timedelta(days=lagtime)

            ID_selected_samples["ID_output"] = output[ID_selected_indices]
            ID_selected_samples["ID_target_date"] = ID_selected_target_dates
            ID_selected_samples["ID_input_date"] = ID_selected_input_dates
            ID_selected_samples["ID_iqr"] = scaled_iqr[ID_selected_indices]
            ID_selected_samples["ID_crps"] = crps[ID_selected_indices]
            ID_selected_samples["ID_input_maps"] = input_maps.sel(time=ID_selected_target_dates)

            # accumulate all selected indices from the test dataset: 
            ID_all_selected_indices.extend(ID_selected_indices)

            ID_data_from_all_seeds[str(exp_name)] = ID_selected_samples

        # Collect all data from across random seeds
        all_ID_target_dates = []
        all_ID_output = []
        all_ID_iqr = []
        all_ID_crps = []
        all_ID_input = []

        for iexp, exp_name in enumerate(exp_list):
            all_ID_target_dates.extend(ID_data_from_all_seeds[exp_name]["ID_target_date"])
            all_ID_output.extend(ID_data_from_all_seeds[exp_name]["ID_output"])
            all_ID_iqr.extend(ID_data_from_all_seeds[exp_name]["ID_iqr"])
            all_ID_crps.extend(ID_data_from_all_seeds[exp_name]["ID_crps"])
            all_ID_input.extend(ID_data_from_all_seeds[exp_name]["ID_input_date"].values)

        # print(f"all selected indices: {all_selected_indices}, len: {len(all_selected_indices)}")
        
        # Find corresponding samples in opposing model type: ------------------------------------------------
        if exp_type in ["OBS(OBS)"]:
            base_exp = "OBS(OBS)"
            opposing_exp = "E3SM-short(OBS)"
            models = ["exp189", "exp195", "exp196", "exp197", "exp198", "exp199"]
        elif exp_type in ["E3SM-short(E3SM)"]: 
            base_exp = "E3SM-short(E3SM)"
            opposing_exp = "OBS(E3SM)"
            models = ["exp206", "exp207", "exp208", "exp209", "exp210", "exp211", "exp212", "exp213", "exp214", "exp215", "exp216", "exp217"]

        # For each selected sample from original model, find corresponding sample in opposing model using input data
            
        for ood_model in models: 
            print(f"  Processing opposing model: {ood_model}")
            OOD_selected_samples = {}
            config = utils.get_config(ood_model)
            
            # Load the output and target data for this experiment
            output_ood = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{ood_model}/{ood_model}_network_SHASH_parameters.pkl')

            # open crps: 
            crps_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{ood_model}/{ood_model}_CRPS_network_values.pkl')

                ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr_ood = iqr_basic(output_ood)
            # create xarray object containing iqr and corresponding time coordinate: 
            iqr_ood_xr = xr.DataArray(iqr_ood, coords=[target['time']], dims=["time"])
            # scale iqr by day of year:
            daily_iqr_ood = iqr_ood_xr.groupby('time.dayofyear').mean('time')
            scaled_iqr_ood = iqr_ood_xr.groupby('time.dayofyear') / daily_iqr_ood
            
            # Load testing target data
            if opposing_exp in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                target_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                climatology_stats_ood = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                target_ood = (target_ood['y'] - climatology_stats_ood['z'][2]) / climatology_stats_ood['z'][3]
        
            elif opposing_exp in ["E3SM-short(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                target_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                climatology_stats_ood = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                target_ood = (target_ood['y'] - climatology_stats_ood['Z500'][2]) / climatology_stats_ood['Z500'][3]


            ## OOD SAMPLE SELECTION LOGIC: Identify where OOD OUTPUT values had 25% highest CRPS
            if selection_method == "high_low_crps":
                crps_threshold_high = np.percentile(crps, 100 - confidence)  # Upper quartile
                OOD_selected_indices = np.where(crps >= crps_threshold_high)[0]

            # select sample information from OOD model selections
            OOD_selected_samples["OOD_output"] = output_ood[OOD_selected_indices]
            OOD_selected_samples["OOD_target_dates"] = target_ood['time'][OOD_selected_indices]
            # OOD_selected_samples["OOD_input_date"] = OOD_selected_input_dates
            OOD_selected_samples["OOD_iqr"] = scaled_iqr_ood[OOD_selected_indices]
            OOD_selected_samples["OOD_crps"] = crps_ood[OOD_selected_indices]

            OOD_data_from_all_seeds[str(ood_model)] = OOD_selected_samples

        all_OOD_target_dates = []
        all_OOD_output = []
        all_OOD_iqr = []
        all_OOD_crps = []

        for imod, ood_model in enumerate(models):
            OOD_target_dates = OOD_data_from_all_seeds[ood_model]["OOD_target_dates"]
            if hasattr(OOD_target_dates, 'values'):
                all_OOD_target_dates.extend(OOD_target_dates.values)
            else:
                all_OOD_target_dates.extend(OOD_target_dates)
            all_OOD_output.extend(OOD_data_from_all_seeds[str(ood_model)]["OOD_output"])
            all_OOD_iqr.extend(OOD_data_from_all_seeds[str(ood_model)]["OOD_iqr"])
            all_OOD_crps.extend(OOD_data_from_all_seeds[str(ood_model)]["OOD_crps"])

        # -----------------------------------------------------------------------
        # From ID and OOD selected samples, identify dates that are common between both sets of selected samples
        # and identify their corresponding output,iqr,crps,inputmaps, etc.. into new container
        
        # Convert both lists of dates to numpy datetime64: 
        all_ID_target_dates = np.array([extract_datetime64(date) for date in all_ID_target_dates])
        all_OOD_target_dates = np.array([extract_datetime64(date) for date in all_OOD_target_dates])

        # all_ID_target_dates_day = all_ID_target_dates.astype('datetime64[D]')
        # all_OOD_target_dates_day = all_OOD_target_dates.astype('datetime64[D]')
        common_dates = np.intersect1d(all_ID_target_dates, all_OOD_target_dates)
        print(f"Number of common dates between ID and OOD selections: {len(common_dates)}")

        # ---- TODO identify ENSO and MJO phase of ID+OOD Selected Samples ----------------
        ###### ENSO ########
        common_samples = {}
        common_samples_all_seeds = {}
        all_common_samples = []
        for exp_name in exp_list:
            config = utils.get_config(exp_name)
            
            # Load the output and target data for this experiment
            output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')

            # open crps: 
            crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

                ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr = iqr_basic(output)
            
            # Load testing target data
            if exp_type in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                data_type = "OBS"
                data_vars = ["tp", "skt", "z"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
                target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
        
            elif exp_type in ["E3SM-short(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                data_type = "E3SM"
                data_vars = ["PRECT", "TS", "Z500"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]
                target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]

            iqr_xr = xr.DataArray(iqr, coords=[target['time']], dims=["time"])
            # scale iqr by day of year:
            daily_iqr = iqr_xr.groupby('time.dayofyear').mean('time')
            scaled_iqr = iqr_xr.groupby('time.dayofyear') / daily_iqr
            #ungroup scaled_iqr: 
            scaled_iqr = scaled_iqr.sortby('time')

            # Open ENSO dates for E3SM vs OBS data: 
            enso_dates_pkl = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_daily_enso_timestamps.pkl')

            # check which key (category) each of the target dates falls into, and create a list with either "EN", "LN" or "N"
            enso_phase = []
            for date in common_dates:
                if date in enso_dates_pkl['El Nino']:
                    enso_phase.append("EN")
                elif date in enso_dates_pkl['La Nina']:
                    enso_phase.append("LN")
                else:
                    enso_phase.append("N")

            ######## MJO ########
            phase_timestamps = analysis_metrics.mjo_timestamps(data_type, config)
            # selected_mjo_phase = phase_timestamps.sel(time=selected_target_dates)
            common_mjo_phase = phase_timestamps.sel(time=common_dates)

            ## Identify Common Dates and Data: -----------
            # identify common dates as indices using ID Models (Same for OOD models because same inference dataset) 
            # common_indices = np.where(np.isin(target, common_dates))[0]
            target_times = np.array([extract_datetime64(t) for t in target['time'].values])
            common_dates_set = set(common_dates)
            common_indices = [i for i, t in enumerate(target_times) if t in common_dates_set]

            common_samples["common_dates"] = common_dates
            common_samples["common_ID_output"] = output[common_indices]
            common_samples["common_target_values"] = target.sel(time = common_dates)
            common_samples["common_ID_scaled_iqr"] = scaled_iqr[common_indices]
            common_samples["common_ID_crps"] = crps[common_indices]
            common_samples["enso_phase"] = enso_phase
            common_samples["mjo_phase"] = common_mjo_phase
            common_samples["common_input_maps"] = input_maps.sel(time=common_dates)

            common_samples_all_seeds[str(exp_name)] = common_samples

        all_common_dates = []
        all_common_enso_phases = []
        all_common_target_vals = []
        all_common_mjo_phases = []
        all_common_inputmaps = []
        all_common_ID_output = []
        all_common_ID_iqr = []
        all_common_ID_crps = []

        for iexp, exp_name in enumerate(exp_list):
            all_common_dates.extend(common_samples_all_seeds[exp_name]["common_dates"])
            all_common_target_vals.extend(common_samples_all_seeds[exp_name]["common_target_values"].values)
            all_common_enso_phases.extend(common_samples_all_seeds[exp_name]["enso_phase"])
            all_common_mjo_phases.extend(common_samples_all_seeds[exp_name]["mjo_phase"]["phase"].values)
            all_common_inputmaps.extend(common_samples_all_seeds[exp_name]["common_input_maps"].values)
            all_common_ID_output.extend(common_samples_all_seeds[exp_name]["common_ID_output"])
            all_common_ID_iqr.extend(common_samples_all_seeds[exp_name]["common_ID_scaled_iqr"])
            all_common_ID_crps.extend(common_samples_all_seeds[exp_name]["common_ID_crps"])

        for ood_model in models: 
            print(f"  Processing opposing model: {ood_model}")
            config = utils.get_config(ood_model)
            
            # Load the output and target data for this experiment
            output_ood = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{ood_model}/{ood_model}_network_SHASH_parameters.pkl')

            # open crps: 
            crps_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{ood_model}/{ood_model}_CRPS_network_values.pkl')

                ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr_ood = iqr_basic(output_ood)
            # create xarray object containing iqr and corresponding time coordinate: 
            iqr_ood_xr = xr.DataArray(iqr_ood, coords=[target['time']], dims=["time"])
            # scale iqr by day of year:
            daily_iqr_ood = iqr_ood_xr.groupby('time.dayofyear').mean('time')
            scaled_iqr_ood = iqr_ood_xr.groupby('time.dayofyear') / daily_iqr_ood
            
            # Load testing target data
            if opposing_exp in ["E3SM-short(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)"]:
                target_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                climatology_stats_ood = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')#
                target_ood = (target_ood['y'] - climatology_stats_ood['z'][2]) / climatology_stats_ood['z'][3]
        
            elif opposing_exp in ["E3SM-short(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)"]:
                target_ood = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                climatology_stats_ood = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                target_ood = (target_ood['y'] - climatology_stats_ood['Z500'][2]) / climatology_stats_ood['Z500'][3]

            ## Identify Common Dates and Data: -----------
            # identify common dates as indices using ID Models (Same for OOD models because same inference dataset)

            common_samples["common_OOD_output"] = output_ood[common_indices]
            common_samples["common_OOD_scaled_iqr"] = scaled_iqr_ood[common_indices]
            common_samples["common_OOD_crps"] = crps_ood[common_indices]

            common_samples_all_seeds[str(ood_model)] = common_samples

        all_common_OD_output = []
        all_common_OD_iqr = []
        all_common_OD_crps = []

        for imod, ood_model in enumerate(models):
            all_common_OD_output.extend(common_samples_all_seeds[ood_model]["common_OOD_output"])
            all_common_OD_iqr.extend(common_samples_all_seeds[ood_model]["common_OOD_scaled_iqr"]) 
            all_common_OD_crps.extend(common_samples_all_seeds[ood_model]["common_OOD_crps"])

        # ---- PLOTTING ---------------------------------------------------------

        # SUMMARY PLOT: 
        # 4 panels: (1) shash curves (2) MJO phase distributions (3) ENSO phase distribution (4) target value distribution
        # (1) shash curves from all output1 in one color, and all output2 in another color

        # (2) MJO phase distribution from common dates
        mjo_baseline_frequencies = analysis.analysis_metrics.baseline_mjo_frequencies(data_type)
        # print(f"mjo baseline frequencies: {mjo_baseline_frequencies}")
        if "OBS" in data_type:
            mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_phase_frequencies_ERA5_1940_2023.pkl')
        elif "E3SM" in data_type:
            mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_frequencies_E3SM_1850_2014.pkl')

        # (3) ENSO phase distribution from common dates
        all_enso_phases = []
        for exp_name in exp_list:
            all_enso_phases.extend(common_samples_all_seeds[exp_name]["enso_phase"])
        all_enso_phases = np.array(all_enso_phases)
        # print(f"all enso phases: {all_enso_phases}, type: {type(all_enso_phases)}, len: {len(all_enso_phases)}")
        enso_baseline_frequencies = analysis.analysis_metrics.baseline_enso_frequencies(data_type)

        # if "OBS" in data_type:
        #     date_values = [np.datetime64(date_da.values) for date_da in common_dates]
        #     # print(f" date_values max date: {np.max(date_values)}, min date: {np.min(date_values)}")
        #     selected_target_values = target.sel(time=date_values)
        # elif "E3SM" in data_type:
        #     date_values = [date_da.values.item() for date_da in common_dates]
        #     selected_target_values = target.sel(time=date_values)
        
        # INDIVIDUAL PLOTS: 
        # 2 panels (1) shash curves (2) input map
        # for first random seed in exp_list: 
        first_exp = exp_list[0]
        for i, date in enumerate(common_samples_all_seeds[first_exp]["common_dates"]):
            if i <= 1:
                fig = plt.figure(figsize=(12, 6))
                # First panel: regular axis for SHASH curves
                ax0 = fig.add_subplot(2, 2, 1)
                # Next three panels: GeoAxes for maps
                ax1 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree(central_longitude=180))
                ax2 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree(central_longitude=180))
                ax3 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree(central_longitude=180))
                ax = [ax0, ax1, ax2, ax3]

                # SHASH curves
                ID_iqr = common_samples_all_seeds[first_exp]["common_ID_scaled_iqr"][i]
                OOD_iqr = common_samples_all_seeds[list(common_samples_all_seeds.keys())[0]]["common_OOD_scaled_iqr"][i]
                ID_crps = common_samples_all_seeds[first_exp]["common_ID_crps"][i]
                OOD_crps = common_samples_all_seeds[list(common_samples_all_seeds.keys())[0]]["common_OOD_crps"][i]
                target_val = common_samples_all_seeds[first_exp]["common_target_values"][i]
                target_date = common_samples_all_seeds[first_exp]["common_dates"][i]
                enso_phase = common_samples_all_seeds[first_exp]["enso_phase"][i]
                mjo_phase = common_samples_all_seeds[first_exp]["mjo_phase"]["phase"][i]

                def format_date_string(date_obj):
                    """Convert various date types to string format"""
                    if hasattr(date_obj, 'strftime'):
                        # For cftime objects and pandas datetime
                        return date_obj.strftime('%Y-%m-%d')
                    elif hasattr(date_obj, 'values'):
                        # For xarray DataArray
                        return format_date_string(date_obj.values)
                    else:
                        # For other formats, convert to string
                        return str(date_obj)

                target_date_str = format_date_string(target_date)
                # cut string to 9th digit: 
                target_date_str = target_date_str[:10]

                ID_output_params = common_samples_all_seeds[first_exp]["common_ID_output"]
                OOD_output_params = common_samples_all_seeds[list(common_samples_all_seeds.keys())[0]]["common_OOD_output"]
                x = np.linspace(-5, 5, 100)
                ID_dist = Shash(ID_output_params)
                OOD_dist = Shash(OOD_output_params)
                ID_p = ID_dist.prob(x).numpy()
                OOD_p = OOD_dist.prob(x).numpy()

                ax[0].hist(
                    climatology_data, x, density=True, color="silver", alpha=0.75, label="climatology"
                )

                ax[0].plot(x, ID_p[:, i], linewidth = 0.5, label = f"{base_exp}\nIQR: {ID_iqr:.2f}\nCRPS: {ID_crps:.2f}", color='blue')
                ax[0].plot(x, OOD_p[:, i], linewidth = 0.5, label = f"{opposing_exp}\nIQR: {OOD_iqr:.2f}\nCRPS: {OOD_crps:.2f}", color='orange')
                ax[0].set_xlabel(f"Standardized {config['databuilder']['target_var']} Anomaly")
                ax[0].set_ylabel("probability density")
                ax[0].set_title("Network Shash Prediction -" + str(config["expname"]))
                ax[0].axvline(target_val, color='r', linestyle='dashed', linewidth=1)
                plt.legend()
                ax[0].set_title(f'SHASH Comparison for Target Date: {target_date_str}\n ENSO: {enso_phase}, MJO: {mjo_phase.values}')
                ax[0].set_xlabel('Standardized Anomaly')
                ax[0].set_ylabel('Probability Density')
                ax[0].legend()

                cmaps = ['BrBG', 'RdBu_r', 'PuOr_r']
                labels = ['Precipitation Anomaly (mm/day)', 'Skin Temperature Anomaly (K)', 'Z500 Anomaly (m)']

                # Input map
                for k in range(3): 
                    if k ==0: 
                        vmin = -10
                        vmax = 10
                    else: 
                        vmin = -(np.max(common_samples_all_seeds[first_exp]["common_input_maps"][..., k]))
                        vmax = np.max(common_samples_all_seeds[first_exp]["common_input_maps"][..., k])

                    cf1 = ax[k+1].pcolormesh(common_samples_all_seeds[first_exp]["common_input_maps"].lon, common_samples_all_seeds[first_exp]["common_input_maps"].lat, common_samples_all_seeds[first_exp]["common_input_maps"][i,..., k], cmap=cmaps[k], transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
                # ax.set_title(str(keyword) + ' Composite Map')
                    ax[k+1].coastlines()
                    ax[k+1].set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
                    ax[k+1].set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

                # add colorbar that is same for both plots
                    cbar1 = fig.colorbar(cf1, cmap=cmaps[k], ax=ax[k+1], orientation='vertical', fraction=0.01, pad=0.03)
                    cbar1.set_label(labels[k])

                plt.tight_layout()
                plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/individual_samples/high-low_CRPS_sample_comparison_{target_date_str}_{first_exp}_vs_{list(common_samples_all_seeds.keys())[0]}.png', format='png', dpi=250)

        # Time series of common_dates colored by enso phase: 
        fig, ax = plt.subplots(figsize=(12, 4))
        all_common_dates_np = np.array([extract_datetime64(date) for date in all_common_dates])
        scatter = ax.scatter(all_common_dates_np, all_common_target_vals, c=pd.Categorical(all_common_enso_phases).codes, cmap='viridis', alpha=0.7)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('ENSO Phase')
        cbar.set_ticks([0, 1, 2])
        cbar.set_ticklabels(['EN', 'N', 'LN'])
        ax.set_title(f'Target Values of Common Dates between {base_exp} and {opposing_exp}\nColored by ENSO Phase')
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Standardized {config["databuilder"]["target_var"]} Anomaly')
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/time_series_common_dates_enso_phases_high-low_CRPS_{confidence}percent_{data_type}.png', format='png', dpi=250)

        
             # Plot 2: MJO Phase Distribution
        counts = np.bincount(all_common_mjo_phases, minlength=9)
        total = counts.sum()
        densities = counts / total

        phases = np.arange(0, 9)
        phase_labels = [str(i) for i in phases]

        fig, ax = plt.subplots(figsize=(8, 6))
        bar_width = 0.8

        # Bar plot for density
        bars = ax.bar(phases, densities, width=bar_width, color="#4b7894", alpha=0.7, edgecolor='black', label='Selected Samples')

        # Reference lines for each phase
        for i, freq in enumerate(mjo_ref_frequencies_all_data):
            # Draw a horizontal line across the width of the bar for phase i
            ax.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, color="#3C3B3B", linewidth=2, linestyle='-', label='Reference' if i==0 else None)

        ax.set_xticks(phases)
        ax.set_xticklabels(phase_labels)
        ax.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
        ax.set_xlabel('MJO Phase')
        ax.set_ylabel('Density')
        ax.set_title(f'MJO Phase Distribution {keyword}')
        handles, labels = ax.get_legend_handles_labels()
        # Only show one legend entry for the reference lines
        if 'Reference' in labels:
            idx = labels.index('Reference')
            ax.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
        else:
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_high-low_CRPS_{confidence}_{exp_type}.png', format='png', dpi=250)
        plt.show()

        # Histogram for time-of-year in common_dates, and time-of-month: 
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        all_common_dates_np = np.array([extract_datetime64(date) for date in all_common_dates])
        months = np.array([date.astype('datetime64[M]').astype(int) % 12 + 1 for date in all_common_dates_np])
        days = np.array([date.astype('datetime64[D]').astype(int) % 30 + 1 for date in all_common_dates_np])
        print(f"months: {months}")

        ax[0].hist(months, bins=np.arange(1, 14)-0.5, density=True, color='skyblue', edgecolor='black', alpha=0.7)
        ax[0].set_xticks(np.arange(1, 13))
        ax[0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax[0].set_xlabel('Month')
        ax[0].set_ylabel('Density')
        ax[0].set_title('Distribution of Common Dates by Month')

        ax[1].hist(days, bins=np.arange(1, 32)-0.5, density=True, color='salmon', edgecolor='black', alpha=0.7)
        ax[1].set_xticks(np.arange(1, 32, 2))
        ax[1].set_xlabel('Day of Month')
        ax[1].set_ylabel('Density')
        ax[1].set_title('Distribution of Common Dates by Day of Month')     

        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/time_of_year_and_month_histograms_common_dates_high-low_CRPS_{confidence}_{exp_type}.png', format='png', dpi=250)





def IQR_only_analysis(experiments, selection_method = 'simple_iqr_percentage', confidence = 20, keyword = None):
    """
    Select samples from either OBS(OBS) or E3SM(E3SM) using variety of methods: 
        - select most confident sample from each month from each random seed
        - select most confident percentage of samples from total samples from each random seed
        - select dates that are input by hand
    
    Find selected samples that were run through opposing model type (OBS->E3SM or E3SM->OBS) and compare results to original model type.
    
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
        data_from_all_seeds = {}
        all_selected_indices = []

        for exp_name in exp_list:
            print(f'  Processing experiment: {exp_name}')
            selected_samples = {}
            config = utils.get_config(exp_name)
            
            try:
                output = load_pickle(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_network_SHASH_parameters.pkl')
            except FileNotFoundError:
                pattern = f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/exp*_{exp_name}_OOD_INFERENCE_network_SHASH_parameters.pkl'
                matching_files = glob.glob(pattern)
                output = load_pickle(matching_files[0]) if matching_files else None

            # open crps: 
            crps = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/saved/output/{exp_name}/{exp_name}_CRPS_network_values.pkl')

                ## CALCULATE IQR and Select Samples based on Confidence -------
            iqr = iqr_basic(output)
            
            # Load testing target data
            if exp_type in ["E3SM(OBS)", "E3SM(OBS)", "E3SM-long(OBS)", "OBS(OBS)", "OBS(OBS)sv", "E3SM(OBS)sv"]:
                data_type = "OBS"
                data_vars = ["tp", "skt", "z"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp173_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/ERA5_processed_climo_stats_TP_SKT_Z_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]
                target = (target['y'] - climatology_stats['z'][2]) / climatology_stats['z'][3]

                if exp_type in ["E3SM(OBS)sv", "OBS(OBS)sv"]:
                    # scale target by day of year variance: 
                    daily_target_grouped_var = target.groupby('time.dayofyear').var('time')
                    scaled_target = target.groupby('time.dayofyear') / daily_target_grouped_var
                    scaled_target = scaled_target.sortby('time')
                    target = scaled_target
        
            elif exp_type in ["E3SM(E3SM)", "E3SM-long(E3SM)", "OBS(E3SM)", "E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                data_type = "E3SM"
                data_vars = ["PRECT", "TS", "Z500"]
                target_var = config["databuilder"]["target_var"]
                target = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_unstandardized = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp185_trimmed_test_dat.nc')
                input_maps = input_unstandardized['x']
                climatology_stats = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/E3SM_processed_climo_stats_PRECT_Z500_TS_1981-2010.pkl')
                for l, variable in enumerate(data_vars):
                    mean = climatology_stats[variable][0]
                    std = climatology_stats[variable][1]
                    input_maps.loc[dict(channel=l)] = (input_maps.sel(channel=l) - mean) / std
                climatology_data = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/exp152_E3SM_processed_Z500_climatology_1981-2010.nc')
                climatology_data = (climatology_data['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]
                target = (target['y'] - climatology_stats['Z500'][2]) / climatology_stats['Z500'][3]

                if exp_type in ["E3SM(E3SM)sv", "OBS(E3SM)sv"]:
                    # scale target by day of year variance:
                    daily_target_grouped_var = target.groupby('time.dayofyear').var('time')
                    scaled_target = target.groupby('time.dayofyear') / daily_target_grouped_var
                    scaled_target = scaled_target.sortby('time')
                    target = scaled_target

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

            elif selection_method == 'scaled_iqr_by_percentage': # scale IQR by day of year 
                # create xarray object containing iqr and corresponding time coordinate: 
                iqr_xr = xr.DataArray(iqr, coords=[target['time']], dims=["time"])
                # scale iqr by day of year:
                daily_iqr = iqr_xr.groupby('time.dayofyear').mean('time')
                scaled_iqr = iqr_xr.groupby('time.dayofyear') / daily_iqr
                #ungroup scaled_iqr: 
                scaled_iqr = scaled_iqr.sortby('time')
                # print(f"iqr_xr.values: {iqr_xr.values}, scaled_iqr.values: {scaled_iqr.values}")
                # print(f"scaled iqr shape: {scaled_iqr.shape}")

                # select narrowest percentage of scaled IQR based on confidence level: 
                num_to_select = int(len(scaled_iqr) * (confidence / 100))
                selected_indices = np.argsort(scaled_iqr.values)[:num_to_select]
                print(f"selected {len(selected_indices)} samples based on scaled IQR by percentage")
            
            elif selection_method == 'simple_iqr_percentage':
                # select narrowest percentage of IQR based on confidence level: 
                num_to_select = int(len(iqr) * (confidence / 100))
                selected_indices = np.argsort(iqr)[:num_to_select]
                print(f"selected {len(selected_indices)} samples based on simple IQR by percentage")

            elif selection_method == 'no_scaling':
                # select narrowest percentage of IQR based on confidence level: 
                num_to_select = int(len(iqr) * (confidence / 100))
                selected_indices = np.argsort(iqr)[:num_to_select]
                print(f"selected {len(selected_indices)} samples based on no scaling IQR by percentage")
            else: 
                print(f"choose sample selection method or code another one up")

            # identify target dates for these conf samples
            selected_target_dates = target['time'][selected_indices]
            if "E3SM" in data_type: 
                selected_target_dates_exact = pd.to_datetime([
                f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}" 
                for dt in selected_target_dates.values])
                selected_input_dates_exact = pd.to_datetime([
                f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}" 
                for dt in (selected_target_dates - pd.Timedelta(days=config['databuilder']['lagtime'])).values])
            else: 
                selected_target_dates_exact = selected_target_dates
                selected_input_dates_exact = selected_target_dates - pd.Timedelta(days=config['databuilder']['lagtime'])

            # use lagtime to identify input dates for these conf samples
            lagtime = config['databuilder']['lagtime']
            selected_input_dates = selected_target_dates - pd.Timedelta(days=lagtime)

            selected_samples["output"] = output[selected_indices]
            selected_samples["target_date"] = selected_target_dates
            selected_samples["input_date"] = selected_input_dates
            selected_samples["crps"] = crps[selected_indices]
            selected_samples["input_maps"] = input_maps.sel(time=selected_target_dates)
            if selection_method == 'scaled_iqr_by_percentage':
                selected_samples["iqr"] = scaled_iqr[selected_indices]
            elif selection_method == 'simple_iqr_percentage' or selection_method == 'no_scaling':
                selected_samples["iqr"] = iqr[selected_indices]

            # accumulate all selected indices from the test dataset: 
            all_selected_indices.extend(selected_indices)

            # ---- identify ENSO and MJO phase of selected samples ----------------
            ###### ENSO ########
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

            ######## MJO ########
            phase_timestamps = analysis_metrics.mjo_timestamps(data_type, config)

            # selected_mjo_phase = phase_timestamps.sel(time=selected_target_dates)
            selected_mjo_phase = phase_timestamps.sel(time=selected_target_dates_exact)
            selected_samples["mjo_phase"] = selected_mjo_phase

            input_mjo_phase = phase_timestamps.sel(time=selected_input_dates_exact)
            selected_samples["input_mjo_phase"] = input_mjo_phase

            data_from_all_seeds[str(exp_name)] = selected_samples
      
        # Collect all target dates from 'data_from_all_seeds[exp_name]["target_date"]'
        all_target_dates = []
        all_output = []
        all_iqr = []
        all_crps = []
        all_input = []
        all_enso_phases = []
        all_mjo_phases = []
        all_inputmaps = []
        all_input_mjo_phases = []

        for iexp, exp_name in enumerate(exp_list):
            all_target_dates.extend(data_from_all_seeds[exp_name]["target_date"])
            all_output.extend(data_from_all_seeds[exp_name]["output"])
            all_iqr.extend(data_from_all_seeds[exp_name]["iqr"])
            all_crps.extend(data_from_all_seeds[exp_name]["crps"])
            all_input.extend(data_from_all_seeds[exp_name]["input_date"].values)
            all_enso_phases.extend(data_from_all_seeds[exp_name]["enso_phase"])
            all_mjo_phases.extend(data_from_all_seeds[exp_name]["mjo_phase"]["phase"].values.tolist())
            all_input_mjo_phases.extend(data_from_all_seeds[exp_name]["input_mjo_phase"]["phase"].values.tolist())

        # ---- PLOTTING ---------------------------------------------------------

        # (2) MJO phase distribution from selected_target_dates1
 
        # print(f"mjo baseline frequencies: {mjo_baseline_frequencies}")
        if "OBS" in data_type:
            mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_phase_frequencies_ERA5_1940_2023.pkl')
        elif "E3SM" in data_type:
            mjo_ref_frequencies_all_data = open_data_file(f'/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/mjo_frequencies_E3SM_1850_2014.pkl')

         ### MJO phase distribution INITIALIZATION DAY - CHECK EACH SEED SEPARATELY

        # Create a figure with subplots for each random seed
        n_seeds = len(exp_list)
        fig, axes = plt.subplots(2, (n_seeds + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()

        phases = np.arange(0, 9)
        phase_labels = [str(i) for i in phases]
        bar_width = 0.8

        # Plot for each seed
        for idx, exp_name in enumerate(exp_list):
            ax = axes[idx]
            
            # Get MJO phases for this seed
            seed_input_mjo_phases = data_from_all_seeds[exp_name]["input_mjo_phase"].phase.values
            
            # Count each phase
            counts = np.bincount(seed_input_mjo_phases, minlength=9)
            total = counts.sum()
            densities = counts / total
            
            # Bar plot for density
            bars = ax.bar(phases, densities, width=bar_width, color="#564b94", alpha=0.7, 
                        edgecolor='black', label='Selected Samples' if idx == 0 else None)
            
            # Reference lines for each phase
            for i, freq in enumerate(mjo_ref_frequencies_all_data):
                ax.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, 
                        color="#3C3B3B", linewidth=2, linestyle='-', 
                        label='Reference' if (idx == 0 and i == 0) else None)
            
            ax.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
            ax.set_xticks(phases)
            ax.set_xticklabels(phase_labels, fontsize=8)
            ax.set_ylabel('Density', fontsize=8)
            ax.set_title(f'{exp_name}\n(n={total})', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide extra subplots if odd number of seeds
        for idx in range(n_seeds, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'MJO Phase Distribution by Random Seed on Initialization Day | {exp_type} {keyword} | {confidence}% Confidence', 
                    fontsize=12, y=1.00)
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_by_seed_{exp_type}_{confidence}_INIT_day_{keyword}.png', 
                    format='png', dpi=250, bbox_inches='tight')


        # (3) ENSO phase distribution from selected_target_dates
        all_enso_phases = []
        for exp_name in exp_list:
            all_enso_phases.extend(data_from_all_seeds[exp_name]["enso_phase"])
        all_enso_phases = np.array(all_enso_phases)
        # print(f"all enso phases: {all_enso_phases}, type: {type(all_enso_phases)}, len: {len(all_enso_phases)}")
        # calculate frequency of enso phase relative to prevalence in total target dataset
        enso_baseline_frequencies = analysis.analysis_metrics.baseline_enso_frequencies(data_type)
        # print(f"ENSO baseline frequencies: {enso_baseline_frequencies}")

        # print(f'all target 1 values type: {type(all_target1_dates)}, examp: {all_target1_dates[0]}, type examp: {type(all_target1_dates[0])}')
        # (4) Target value distribution from selected_target_dates1
        if "OBS" in data_type:
            # Extract numpy.datetime64 values from DataArrays
            date_values = [np.datetime64(date_da.values) for date_da in all_target_dates]
            # print(f" date_values max date: {np.max(date_values)}, min date: {np.min(date_values)}")
            selected_target_values = target.sel(time=date_values)
        elif "E3SM" in data_type:
            date_values = [date_da.values.item() for date_da in all_target_dates]
            selected_target_values = target.sel(time=date_values)
        
        # Mean IQR for all output
        mean_iqr = np.mean(all_iqr)
        print(f"Mean IQR for all output: {mean_iqr}")

        # Mean CRPS for all output
        mean_crps = np.mean(all_crps)
        print(f"Mean CRPS for all output: {mean_crps}")

        # Plot 2: MJO Phase Distribution - INITIALIZATION DAY
        all_input_mjo_phases = []
        for exp_name in exp_list:
            all_input_mjo_phases.extend(data_from_all_seeds[exp_name]["input_mjo_phase"]["phase"].values.tolist())
        all_input_mjo_phases = np.array(all_input_mjo_phases)
        counts = np.bincount(all_input_mjo_phases, minlength=9)
        total = counts.sum()
        densities = counts / total

        phases = np.arange(0, 9)
        phase_labels = [str(i) for i in phases]

        fig, ax = plt.subplots(figsize=(8, 6))
        bar_width = 0.8

        # Bar plot for density
        bars = ax.bar(phases, densities, width=bar_width, color="#534b94", alpha=0.7, edgecolor='black', label='Selected Samples')

        # Reference lines for each phase
        for i, freq in enumerate(mjo_ref_frequencies_all_data):
            # Draw a horizontal line across the width of the bar for phase i
            ax.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, color="#3C3B3B", linewidth=2, linestyle='-', label='Reference' if i==0 else None)

        ax.set_xticks(phases)
        ax.set_xticklabels(phase_labels)
        ax.set_ylim([0, max(densities.max(), np.max(mjo_ref_frequencies_all_data)) * 1.15])
        ax.set_xlabel('MJO Phase')
        ax.set_ylabel('Density')
        ax.set_title(f'MJO Phase Distribution {keyword}')
        handles, labels = ax.get_legend_handles_labels()
        # Only show one legend entry for the reference lines
        if 'Reference' in labels:
            idx = labels.index('Reference')
            ax.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
        else:
            ax.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mjo_phase_distribution_{exp_type}_{confidence}_INIT_day_{keyword}.png', format='png', dpi=250)
        plt.show()

        # PLOT 9: Histogram check of ENSO index strength
        # Plot actual values of ENSO index for ALL samples and also for selected samples as histogram of "El Nino", "La Nina"
        if data_type == "E3SM":
            print("Data source is E3SM - daily ENSO data")
            Nino34 = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/ENSO_Data/E3SM/ENSO_ne30pg2_HighRes/nino.member0201_daily_linterp_shifted.nc')
            Nino34 = Nino34.nino34
            nino34_index = Nino34.sel(time=slice(str(config["databuilder"]["input_years"][0]), str(config["databuilder"]["input_years"][1])))

        elif data_type == "OBS":
            print("Data source is ERA5 - daily ENSO data")            
            Nino34 = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/ENSO_Data/OBS/nino34.long.anom_daily_linterp_shifted.nc')
            Nino34 = Nino34.value
            # full_time = xr.cftime_range(start='1850-01-01', end='2024-12-31', freq='D', calendar='noleap')
            # Nino34 = Nino34.reindex(time=full_time)
            nino34_index = Nino34.sel(time=slice(str(config["databuilder"]["input_years"][0]), str(config["databuilder"]["input_years"][1])))
        
        # Collect index values for histogram of all samples:
        all_enso_values = {'El Nino': [], 'La Nina': [], 'Neutral': []}

        def safe_sel_nino34(nino34_index, date, data_type):
            """
            Selects the Nino3.4 index value for a given date, handling E3SM vs OBS time formats.
            Returns np.nan if not found.
            """
            try:
                if data_type == "E3SM":
                    # Convert any date type to string 'YYYY-MM-DD'
                    if hasattr(date, 'strftime'):
                        date_str = date.strftime('%Y-%m-%d')
                    elif hasattr(date, 'year') and hasattr(date, 'month') and hasattr(date, 'day'):
                        date_str = f"{date.year:04d}-{date.month:02d}-{date.day:02d}"
                    elif isinstance(date, (np.datetime64, pd.Timestamp)):
                        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
                    else:
                        date_str = str(date)[:10]
                    return nino34_index.sel(time=date_str).values.item()
                else:
                    # Use date directly for OBS/ERA5
                    return nino34_index.sel(time=date).values.item()
            except Exception as e:
                # print(f"Date {date} not found in nino34_index, skipping. ({e})")
                return np.nan

        for date in target['time'].values:
            # For ENSO phase assignment, compare as numpy.datetime64
            if date in enso_dates_pkl['El Nino']:
                phase = 'El Nino'
            elif date in enso_dates_pkl['La Nina']:
                phase = 'La Nina'
            else:
                phase = 'Neutral'
            nino_val = safe_sel_nino34(nino34_index, date, data_type)
            all_enso_values[phase].append(nino_val)

        # collect index values for histogram from most confident selected target dates: 
        selected_enso_values = {'El Nino': [], 'La Nina': [], 'Neutral': []}
        for exp_name in exp_list:
            selected_target_dates = data_from_all_seeds[exp_name]["target_date"]
            for date in selected_target_dates.values:
                if date in enso_dates_pkl['El Nino']:
                    phase = 'El Nino'
                elif date in enso_dates_pkl['La Nina']:
                    phase = 'La Nina'
                else:
                    phase = 'Neutral'
                nino_val = safe_sel_nino34(nino34_index, date, data_type)
                selected_enso_values[phase].append(nino_val)
                

        # Plot histograms
        plt.figure(figsize = (9, 6))
        bins = np.linspace(-5, 5, 100)
        plt.hist(all_enso_values['El Nino'], bins=bins, alpha=0.3, label=f'All Samples - El Nino (N = {len(all_enso_values["El Nino"])})', histtype = 'barstacked', color="#482878", density=True)
        plt.hist(all_enso_values['La Nina'], bins=bins, alpha=0.3, label=f'All Samples - La Nina (N = {len(all_enso_values["La Nina"])})', histtype = 'barstacked', color="#26828e", density=True)
        plt.hist(all_enso_values['Neutral'], bins=bins, alpha=0.3, label=f'All Samples - Neutral (N = {len(all_enso_values["Neutral"])})', histtype = 'barstacked', color="#b5de2b", density=True)
        plt.xlabel('Nino3.4 Index Value')
        plt.ylabel('Density')
        plt.title(f'ENSO Index Value Distribution Across all {data_type} Data')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_index_value_distribution_{exp_type}_ALL_data_{keyword}.png', format='png', dpi=250)
        
        plt.figure(figsize = (9, 6))
        bins = np.linspace(-5, 5, 100)
        plt.hist(selected_enso_values['El Nino'], bins=bins, alpha=0.7, label=f'Selected Samples - El Nino (N = {len(selected_enso_values["El Nino"])})', histtype = 'barstacked', color="#482878", density=True)
        plt.hist(selected_enso_values['La Nina'], bins=bins, alpha=0.7, label=f'Selected Samples - La Nina (N = {len(selected_enso_values["La Nina"])})', histtype = 'barstacked', color="#26828e", density=True)
        plt.hist(selected_enso_values['Neutral'], bins=bins, alpha=0.7, label=f'Selected Samples - Neutral (N = {len(selected_enso_values["Neutral"])})', histtype = 'barstacked', color="#b5de2b", density=True)
        plt.xlabel('Nino3.4 Index Value')
        plt.ylabel('Density')
        plt.title(f'ENSO Index Value Distribution For {confidence}% Most Confident | {data_type} {keyword}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_index_value_distribution_{exp_type}_{confidence}_most_confident_{keyword}.png', format='png', dpi=250)
        
        # Plot 3: ENSO Phase Distribution - RATIO
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        bin_edges = np.array([-0.5, 0.5, 1.5, 2.5])
        bin_centers = np.array([0, 1, 2])
        bar_width = 0.4

        sum_total_phases = len(selected_enso_values['El Nino']) + len(selected_enso_values['La Nina']) + len(selected_enso_values['Neutral'])
        enso_phase_dist = [len(selected_enso_values['El Nino']) / sum_total_phases, 
                        len(selected_enso_values['La Nina']) / sum_total_phases, 
                        len(selected_enso_values['Neutral']) / sum_total_phases]

        bars = ax3.bar(bin_centers, enso_phase_dist, width=bar_width, color='#fb9f3a', alpha=0.7, edgecolor='black')

        # Reference lines for each ENSO phase
        enso_phases = ['El Nino', 'La Nina', 'Neutral']
        for i, phase in enumerate(enso_phases):
            freq = enso_baseline_frequencies[phase]
            # Draw a horizontal line across the width of the bar for phase i
            ax3.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, 
                    color="#3C3B3B", linewidth=2, linestyle='-', 
                    label='Reference' if i==0 else None)

        ax3.set_ylim([0, max(max(enso_phase_dist), max(enso_baseline_frequencies.values())) * 1.15])
        ax3.set_xticks(bin_centers)
        ax3.set_xticklabels(['El Nino', 'La Nina', 'Neutral'])
        ax3.set_xlabel('ENSO Phase')
        ax3.set_ylabel('Density')
        ax3.set_title(f'ENSO Phase Distribution | {confidence}% Most Confident | {data_type} {keyword}')

        # Legend with only one entry for reference lines
        handles, labels = ax3.get_legend_handles_labels()
        if 'Reference' in labels:
            idx = labels.index('Reference')
            ax3.legend([bars, handles[idx]], ['Selected Samples', 'Reference'])
        else:
            ax3.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_phase_distribution_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)

## -----------------
## -----------------
        # (3) ENSO phase distribution - CHECK EACH SEED SEPARATELY
        all_enso_phases = []
        for exp_name in exp_list:
            all_enso_phases.extend(data_from_all_seeds[exp_name]["enso_phase"])
        all_enso_phases = np.array(all_enso_phases)

        # Get baseline frequencies
        enso_baseline_frequencies = analysis.analysis_metrics.baseline_enso_frequencies(data_type)
        print(f"ENSO baseline frequencies: {enso_baseline_frequencies}")

        # Create a figure with subplots for each random seed
        n_seeds = len(exp_list)
        fig, axes = plt.subplots(2, (n_seeds + 1) // 2, figsize=(15, 8))
        axes = axes.flatten()

        # Also store aggregate data for comparison
        all_seed_ratios = {'El Nino': [], 'La Nina': [], 'Neutral': []}

        for idx, exp_name in enumerate(exp_list):
            ax = axes[idx]
            
            # Get ENSO phases for this seed
            seed_enso_phases = np.array(data_from_all_seeds[exp_name]["enso_phase"])
            
            # Count each phase
            n_el_nino = np.sum(seed_enso_phases == "EN")
            n_la_nina = np.sum(seed_enso_phases == "LN")
            n_neutral = np.sum(seed_enso_phases == "N")
            total = len(seed_enso_phases)
            
            # Calculate ratios
            ratios = [n_el_nino / total, n_la_nina / total, n_neutral / total]
            all_seed_ratios['El Nino'].append(ratios[0])
            all_seed_ratios['La Nina'].append(ratios[1])
            all_seed_ratios['Neutral'].append(ratios[2])
            
            # Plot
            bin_centers = np.array([0, 1, 2])
            bar_width = 0.4
            
            bars = ax.bar(bin_centers, ratios, width=bar_width, color='#fb9f3a', alpha=0.7, edgecolor='black')
            
            # Reference lines for each ENSO phase
            enso_phases = ['El Nino', 'La Nina', 'Neutral']
            for i, phase in enumerate(enso_phases):
                freq = enso_baseline_frequencies[phase]
                ax.hlines(y=freq, xmin=i - bar_width/2, xmax=i + bar_width/2, 
                        color="#3C3B3B", linewidth=2, linestyle='-', 
                        label='Reference' if i==0 else None)
            
            ax.set_ylim([0, max(max(ratios), max(enso_baseline_frequencies.values())) * 1.15])
            ax.set_xticks(bin_centers)
            ax.set_xticklabels(['EN', 'LN', 'N'], fontsize=8)
            ax.set_ylabel('Ratio', fontsize=8)
            ax.set_title(f'{exp_name}\n(n={total})', fontsize=9)
            if "OBS" in data_type:
                ax.set_ylim([0, 0.6])
            elif "E3SM" in data_type:
                ax.set_ylim([0, 0.5])
            
            if idx == 0:
                ax.legend(fontsize=7)

        # Hide extra subplots if odd number of seeds
        for idx in range(n_seeds, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'ENSO Phase Distribution by Random Seed | {exp_type} {keyword} | {confidence}% Confidence', 
                    fontsize=12, y=1.00)
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/enso_frequency_by_seed_{exp_type}_{confidence}_{keyword}.png', 
                    format='png', dpi=250, bbox_inches='tight')
        plt.show()

 ## -----------------
 ## -----------------
 
        # Plot 4: Target Variable Anomaly Distribution
        fig4, ax4 = plt.subplots(figsize=(8, 6))
        ax4.hist(selected_target_values, bins=20, density=True, color='#0d0887', alpha=0.7, edgecolor='black')
        ax4.set_xlabel(f'Standardized {target_var} Anomaly')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Target {target_var} Anomaly Distribution | {confidence}% Most Confident | {data_type} {keyword}')  # Added data_type to title
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/target_anomaly_distribution_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)

        # Plot 5: CRPS Distribution
        fig5, ax5 = plt.subplots(figsize=(8, 6))
        shared_bins = np.linspace(min(all_crps), max(all_crps), 20)
        ax5.hist(all_crps, bins=shared_bins, density=True, color='#2a788e', alpha=0.7, edgecolor='black', label=f'{exp_type}')
        ax5.set_xlabel('CRPS')
        ax5.set_ylabel('Density')
        ax5.set_title(f'CRPS Distribution | {confidence}% Most Confident | {data_type} {keyword}')  # Added data_type to title
        ax5.legend()
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/crps_distribution_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)
        plt.show()

        # Plot 6: check temporal distribution of most confident samples by month: 
        if "E3SM" in data_type:
            date_values = [date_da.values.item() if hasattr(date_da.values, 'item') else date_da.values for date_da in all_target_dates]
            date_array = xr.DataArray(date_values, dims=['time'])
            months = date_array.dt.month.values.tolist()
        elif "OBS" in data_type:
            date_values = [np.datetime64(date_da.values) for date_da in all_target_dates]
            # Convert to numpy array and extract months using numpy
            date_array = np.array(date_values)
            months = [date.astype('datetime64[M]').astype(int) % 12 + 1 for date in date_array]
        month_names = ['Jan', 'Feb', 'Mar', 'Oct', 'Nov', 'Dec']
        month_counts = [months.count(m) for m in [1, 2, 3, 10, 11, 12]]
        fig6, ax6 = plt.subplots(figsize=(8, 6))
        ax6.bar(month_names, month_counts, color="#0d7e13", alpha=0.7, edgecolor='black')
        ax6.set_xlabel('Month')
        ax6.set_ylabel('Number of Selected Samples')
        ax6.set_title(f'Temporal Distribution of Selected Samples by Month | {confidence}% Most Confident | {data_type} {keyword}')  # Added data_type to title
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/temporal_distribution_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)

        # Plot 7: Plot mean input maps from dates of interest: 
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
        variable_names = ['tp', 'skt', 'z'] if "OBS" in data_type else ['PRECT', 'TS', 'Z500']
        cmap_list = ['BrBG', 'RdBu_r', 'PuOr_r']
        vmin_list = np.zeros(3)
        vmax_list = np.zeros(3)

        for i in range(3):
            # Calculate mean input map for variable (prect, temp, Z)
            input_maps_var = []
            for exp_name in exp_list:
                input_maps_var.append(data_from_all_seeds[exp_name]["input_maps"].sel(channel=i))

            input_maps_var = xr.concat(input_maps_var, dim='time')
            mean_input_map = input_maps_var.mean(dim='time')

            # vmin_list[i] = mean_input_map.min()
            # vmax_list[i] = -1 * (mean_input_map.min()) 
            # if i == 1: 
            #     vmin_list[i] = -1
            #     vmax_list[i] = 1

            vmin_list = [-0.3, -1, -0.6]
            vmax_list = [0.3, 1, 0.6]

            im = ax[i].pcolormesh(
                mean_input_map['lon'],
                mean_input_map['lat'],
                mean_input_map,
                cmap=cmap_list[i],
                vmin=vmin_list[i],
                vmax=vmax_list[i],
                transform=ccrs.PlateCarree(central_longitude=0)
            )
            ax[i].coastlines()
            ax[i].set_title(f'Mean Input Map: {variable_names[i]} | {confidence}% Most Confident | {data_type} {keyword}')
            plt.colorbar(im, ax=ax[i], orientation='horizontal', pad=0.05, label=f'{variable_names[i]} Anomaly')
        plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mean_input_maps_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)

        # Plot 8 : Plot mean input maps for each ENSO phase from most confident samples: 
        enso_phases = ['EN', 'LN', 'N']
        # select temperature input maps for each enso phase (channel = 1):
        fig, ax = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
        for iphase, phase in enumerate(enso_phases):
            input_maps_var = []
            for exp_name in exp_list:
                phase_mask = np.array(data_from_all_seeds[exp_name]["enso_phase"]) == phase
                input_maps_var.append(data_from_all_seeds[exp_name]["input_maps"].sel(channel=1).isel(time=phase_mask)) # channel 1 = temp (skt/TS)

            input_maps_var = xr.concat(input_maps_var, dim='time')
            if len(input_maps_var['time']) == 0:
                print(f"No samples found for ENSO phase: {phase} in variable {variable_names[i]}")
                continue
            mean_input_map = input_maps_var.mean(dim='time')

            abs_max = np.max(np.abs(mean_input_map))
            vmin = -abs_max
            vmax = abs_max 

            im = ax[iphase].pcolormesh(
                mean_input_map['lon'],
                mean_input_map['lat'],
                mean_input_map,
                cmap=cmap_list[1],
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(central_longitude=0)
            )
            ax[iphase].coastlines()
            ax[iphase].set_title(f'Mean Input Map: {variable_names[1]} | ENSO: {phase} | {confidence}% Most Confident | {data_type} {keyword}')
            plt.colorbar(im, ax=ax[iphase], orientation='horizontal', pad=0.05, label=f'{variable_names[1]} Anomaly')
            plt.tight_layout()
        plt.savefig(f'/pscratch/sd/p/plutzner/E3SM/saved/figures/COMBINED/mean_input_maps_{variable_names[1]}_ENSO_{exp_type}_{confidence}_{keyword}.png', format='png', dpi=250)






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