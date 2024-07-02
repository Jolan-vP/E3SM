"""
Climatology Calculation for Comparison

Functions
---------
deriveclimatology()
standardize_data()
make_hist()
calc_cdf()


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



def deriveclimatology(output, climate_data, samples, x, valset):
    """
    Input: Filename for climate data, SHASH parameters for sample
    Output: probability density distribution for given data and shash curve
    """
    imp.reload(shash.shash_torch)

    with gzip.open(climate_data, "rb") as obj1:
        data = pickle.load(obj1)
    climatology = data["y"]
  

    dist = Shash(output)
    p = dist.prob(x).numpy()
    
    # print(p[:,samples])

    plt.figure(figsize=(8, 4), dpi=200)
    plt.hist(
        climatology, x, density=True, color="gray", alpha=0.75, label="climatology"
    )
    y = [0.07, 0.1, 0.05, 0.039, 0.039]
    print(valset[samples][1])
   
    # for isamp, samp in enumerate(samples): 
       # for isamp, samp in enumerate(samples): 
    #     plt.plot(valset[samp][1], y[isamp], 'o')
    #     plt.xlabel("value")
    #     plt.ylabel("probability density")
    #     plt.title("Network Shash Prediction")
    
    plt.plot(x, p[:, samples], linewidth = 0.6 ) #label = samples
    plt.xlabel("value")
    plt.ylabel("probability density")
    plt.title("Network Shash Prediction")
    plt.legend()
    plt.show()

    print(f"Maximum probability values for each sample: {np.max(p[:,samples])}")
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
    return cdf_val