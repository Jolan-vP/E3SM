"""
Climatology Calculation for Comparison

Functions
---------
deriveclimatology()

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
import pickle 
import gzip

def deriveclimatology(output, climate_data, samples):
    """
    Input: Filename for climate data, SHASH parameters for sample
    Output: probability density distribution for given data and shash curve
    """
    imp.reload(shash.shash_torch)

    with gzip.open(climate_data, "rb") as obj1:
        data = pickle.load(obj1)
    climatology = data["y"]

   
    bins_inc = 0.025
    bins = np.arange(-10, 10, bins_inc)

    plt.figure(figsize=(8, 4), dpi=200)

    x = np.arange(-3.5, 5, 0.01)
    dist = Shash(output)
    p = dist.prob(x).numpy()
    
    print(p[:,samples])

    plt.hist(
        climatology, x, density=True, color="gray", alpha=0.75, label="climatology"
    )

    plt.plot(x, p[:, samples], label=samples)
    plt.xlabel("value")
    plt.ylabel("probability density")
    plt.title("Network Shash Prediction")
    plt.legend()
    plt.show()
    return 




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
    return cdf_val