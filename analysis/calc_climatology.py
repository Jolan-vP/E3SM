"""
Climatology Calculation for Comparison

Functions
---------
deriveclimatology()
standardize_data()
make_hist()
calc_cdf()
precipitation_regimes()

Classes
---------

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
from databuilder.data_loader import universaldataloader
import pickle 
import gzip
from model.metric import iqr_basic
from analysis.analysis_metrics import maximum_difference
import utils



def deriveclimatology(output, target, number_of_samples, config, climate_data=False):
    """
    Input: Filename for climate data, SHASH parameters for sample
    Output: probability density distribution for given data and shash curve
    """
    imp.reload(shash.shash_torch)

    if climate_data == True:
        with gzip.open(climate_data, "rb") as obj1:
            data = pickle.load(obj1)
        climatology = data["y"] - data["y"].mean() / data["y"].std() # pulling all target values from processed data

    else:
        climatology = target

    # print(f"Climatological Mean = {np.mean(climatology).item()}")

    extreme_samps = maximum_difference(output, target, required_samples= number_of_samples, tau_frozen=True)

    dist = Shash(extreme_samps)

    x_values = np.linspace(np.min(climatology) - 2 , np.max(climatology) + 3, 1000)

    p = dist.prob(x_values).numpy()

    plt.figure(figsize=(8, 5), dpi=200)
    plt.hist(
        climatology, x_values, density=True, color="silver", alpha=0.75, label="climatology"
    )

    plt.plot(x_values, p, linewidth = 0.5 ) #label = samples
    plt.xlabel(f"Standardized {config['databuilder']['target_var']} Anomaly")
    plt.ylabel("probability density")
    plt.title("Network Shash Prediction -" + str(config["expname"]))
    # plt.axvline(valset[:len(output)], color='r', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + '_predictions_w_climatology.png', format='png', bbox_inches ='tight', dpi = 300)
    # plt.xlim([-10, 12])
    # plt.show(block = False)

    
    #print(f"Maximum probability values for each sample: {np.max(p[:,samples])}")
    return p




## Standardize Data ## -----------------------
def standardize_data(time_series):
    ave_data = np.mean(time_series)
    std_data = np.std(time_series)
    stand_data = (time_series - ave_data) / std_data
    return stand_data, ave_data, std_data


##  Make Histogram ## -------------------------
# Enter sample mean series, and bin values
def make_hist(sm_data, bin_vals):
    sm_hist, bins = np.histogram(sm_data, bins=bin_vals, density=True)
    
    #bins = np.linspace(-4, 4, 150)
    bin_centers = (bins[1:] + bins[:-1])*(0.5)

    #sm_hist1, bins1 = make_hist(sm_hist, bins)

    plt.figure()
    plt.plot(bin_centers, sm_hist, color='c', label=' ')
    plt.legend(bbox_to_anchor=(1.56, 1), loc='upper right')


## CDF Calculation ## --------------------------
def calc_cdf(norm_data, deviation_val):
    cdf_val = len(norm_data[norm_data > deviation_val]) / len(norm_data)
    cdf_val = round(cdf_val, 5)







def precip_regime(data, config): 
    """
    - pass data in as variable
    - should be from training target data
    """

    prect_global = data.PRECT.sel(time = slice(str(config["databuilder"]["input_years"][0]) + '-01-01', str(config["databuilder"]["input_years"][1])))

    min_lat, max_lat = config["databuilder"]["target_region"][:2]
    min_lon, max_lon = config["databuilder"]["target_region"][2:]

    if isinstance(prect_global, xr.DataArray):
        mask_lon = (prect_global.lon >= min_lon) & (prect_global.lon <= max_lon)
        mask_lat = (prect_global.lat >= min_lat) & (prect_global.lat <= max_lat)
        prect_regional = prect_global.where(mask_lon & mask_lat, drop=True)

    # average around seattle region 
    prect_regional = prect_regional.mean(dim=['lat', 'lon'])

    target_raw = universaldataloader(prect_regional, config, target_only = True, repackage = False)

    training_target_raw = target_raw * 86400 * 1000  # Convert to mm/day

    # divide precip data into months: 
    max_size = max((training_target_raw.time.dt.month == i).sum().item() for i in range(1, 13))
    monthly_precip = np.full((12, max_size), np.nan)

    ave_monthly_precip = np.full(12, np.nan)

    for i in range(1, 13):  # Months are 1-12
        month_data = training_target_raw.sel(time=training_target_raw.time.dt.month == i)
        if month_data.values.size == 0: 
            print(f"No data for month {i}")
            continue
        else:
            monthly_precip[i-1, :month_data.values.size] = month_data.values
        
        ave_monthly_precip[i-1] = np.nanmean(monthly_precip[i-1])

    median = round(np.nanmedian(ave_monthly_precip), 2)
    mean = round(np.nanmean(ave_monthly_precip), 2)
    # create histogram of raw precipitation data by month of year: 
    plt.figure()
    months = np.arange(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    plt.bar(months, ave_monthly_precip, color = '#3b528b')
    plt.xticks(months, month_names)
    plt.axhline(median, color = '#aa2395', linestyle = ':', label = f"Median = {median}")
    plt.axhline(mean, color = '#ff7f0e', linestyle = ':', label = f"Mean = {mean}")
    plt.legend()
    plt.ylabel("Average Precipitation (mm/day)")
    plt.savefig('/pscratch/sd/p/plutzner/E3SM/saved/figures/exp025/precipitation_by_month.png', format='png', bbox_inches ='tight', dpi = 300)

    plt.show()
    
    print(f"Median monthly precip: {np.nanmedian(ave_monthly_precip)}")
    print(f"Mean monthly precip: {np.nanmean(ave_monthly_precip)}")


def Z500_regime(E3SM_data, ERA5_data): 
    """
    - pass data in as variable
    - should be from training target data
    """
    config = utils.get_config("exp173")

    Z500_baseline_E3SM = E3SM_data
    Z500_baseline_ERA5 = ERA5_data

    min_lat, max_lat = config["databuilder"]["target_region"][:2]
    min_lon, max_lon = config["databuilder"]["target_region"][2:]

    # Convert longitude from -180/180 to 0/360 if needed
    if min_lon < 0:
        min_lon += 360
    if max_lon < 0:
        max_lon += 360

    if isinstance(Z500_baseline_E3SM, xr.DataArray):
        mask_lon = (Z500_baseline_E3SM.lon >= min_lon) & (Z500_baseline_E3SM.lon <= max_lon)
        mask_lat = (Z500_baseline_E3SM.lat >= min_lat) & (Z500_baseline_E3SM.lat <= max_lat)
        Z500_regional_E3SM = Z500_baseline_E3SM.where(mask_lon & mask_lat, drop=True)

    if isinstance(Z500_baseline_ERA5, xr.DataArray):
        mask_lon = (Z500_baseline_ERA5.lon >= min_lon) & (Z500_baseline_ERA5.lon <= max_lon)
        mask_lat = (Z500_baseline_ERA5.lat >= min_lat) & (Z500_baseline_ERA5.lat <= max_lat)
        Z500_regional_ERA5 = Z500_baseline_ERA5.where(mask_lon & mask_lat, drop=True)

    # average around target region 
    Z500_regional_E3SM = Z500_regional_E3SM.mean(dim=['lat', 'lon'])
    Z500_regional_ERA5 = Z500_regional_ERA5.mean(dim=['lat', 'lon'])

    # divide precip data into months: 
    max_size_E3SM = max((Z500_regional_E3SM.time.dt.month == i).sum().item() for i in range(1, 13))
    max_size_ERA5 = max((Z500_regional_ERA5.time.dt.month == i).sum().item() for i in range(1, 13))
    monthly_Z500_E3SM = np.full((12, max_size_E3SM), np.nan)
    monthly_Z500_ERA5 = np.full((12, max_size_ERA5), np.nan)

    ave_monthly_Z500_E3SM = np.full(12, np.nan)
    ave_monthly_Z500_ERA5 = np.full(12, np.nan)

    for i in range(1, 13):  # Months are 1-12
        month_data_E3SM = Z500_regional_E3SM.sel(time=Z500_regional_E3SM.time.dt.month == i)
        month_data_ERA5 = Z500_regional_ERA5.sel(time=Z500_regional_ERA5.time.dt.month == i)
        if month_data_E3SM.values.size == 0: 
            print(f"No data for month {i}")
            continue
        else:
            monthly_Z500_E3SM[i-1, :month_data_E3SM.values.size] = month_data_E3SM.values
        if month_data_ERA5.values.size == 0: 
            print(f"No data for month {i}")
            continue
        else:
            monthly_Z500_ERA5[i-1, :month_data_ERA5.values.size] = month_data_ERA5.values

        ave_monthly_Z500_E3SM[i-1] = np.nanmean(monthly_Z500_E3SM[i-1])
        ave_monthly_Z500_ERA5[i-1] = np.nanmean(monthly_Z500_ERA5[i-1])

    median_E3SM = round(np.nanmedian(ave_monthly_Z500_E3SM), 2)
    mean_E3SM = round(np.nanmean(ave_monthly_Z500_E3SM), 2)
    median_ERA5 = round(np.nanmedian(ave_monthly_Z500_ERA5), 2)
    mean_ERA5 = round(np.nanmean(ave_monthly_Z500_ERA5), 2)

    # create histogram of baseline Z500 data by month of year: 
    plt.figure()
    months = np.arange(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    # Set bar width and positions for side-by-side bars
    bar_width = 0.35
    x_pos_E3SM = months - bar_width/2
    x_pos_ERA5 = months + bar_width/2

    # Create side-by-side bars
    plt.bar(x_pos_E3SM, ave_monthly_Z500_E3SM, width=bar_width, color="#0095cb", label="E3SM Baseline")
    plt.bar(x_pos_ERA5, ave_monthly_Z500_ERA5, width=bar_width, color="#5401ba", label="ERA5 Baseline")

    # Set x-axis ticks to be centered between the bar pairs
    plt.xticks(months, month_names)
    plt.ylim([5000, 6000])
    plt.title("Raw Z500 by Month (1981-2010) | E3SM and ERA5")

    # # Add horizontal lines for statistics
    # plt.axhline(median_E3SM, color="#0d559c", linestyle=':', label=f"E3SM Median = {median_E3SM}")
    # plt.axhline(mean_E3SM, color="#00256f", linestyle=':', label=f"E3SM Mean = {mean_E3SM}")
    # plt.axhline(median_ERA5, color="#7c2ace", linestyle=':', label=f"ERA5 Median = {median_ERA5}")
    # plt.axhline(mean_ERA5, color="#370160", linestyle=':', label=f"ERA5 Mean = {mean_ERA5}")

    plt.legend()
    plt.ylabel("Average Z500 (m)")
    plt.savefig('/pscratch/sd/p/plutzner/E3SM/databuilder/Z500_by_month_E3SM_ERA5.png', format='png', bbox_inches='tight', dpi=300)
    plt.show()