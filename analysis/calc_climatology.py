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



def deriveclimatology(output, target, number_of_samples, config, climate_data=False):
    """
    Input: Filename for climate data, SHASH parameters for sample
    Output: probability density distribution for given data and shash curve
    """
    imp.reload(shash.shash_torch)

    if climate_data == True:
        with gzip.open(climate_data, "rb") as obj1:
            data = pickle.load(obj1)
        climatology = data["y"] # pulling all target values from processed data
    
    else:
        climatology = target

    print(f"Climatologial Mean = {np.mean(climatology)}")

    extreme_samps = maximum_difference(output, required_samples= number_of_samples, tau_frozen=True)

    dist = Shash(extreme_samps)

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