"""
CRPS Module: 

Functions: -------------------
    crps_basic_numeric

(Proper Scoring Functions:)
    crps_gaussian
    crps_discover_bounds
    crps_cdf_single
    crps_quadrature

Classes: ---------------------
    CumulativeSum

"""

import sys
import os
import xarray as xr
import torch
import torchinfo
import random
import numpy as np
import importlib as imp
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import cartopy.crs as ccrs
import json
import pickle
import gzip
import scipy
from scipy import stats
import properscoring as ps
from scipy.stats import rv_continuous
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import warnings
from shash.shash_torch import Shash
from analysis.analysis_metrics import climatologyCDF
import utils

### SET PROPER CONFIG FILE ################################
exp = "exp008"
config = utils.get_config(exp)
###########################################################

# def crps_basic_numeric(pred, target_y, bins, single_cdf=False):
#     # see alternative formulation in ``crps_sample_score``
    
#     crps = np.zeros((len(target_y),))
#     for isample in np.arange(0, len(target_y)):
#         ibin = np.argmin(np.abs(bins[:-1] - target_y[isample]))

#         pdf, __ = np.histogram(pred[:, isample], bins, density=True)  # PDF = shash.dist()
#         pdf = pdf / (np.sum(pdf) ) #* np.diff(bins)[0]
#         cdf = np.cumsum(pdf) #/ np.sum(pdf)
        
        
#         cdf_sample = cdf
#         term_1 = np.sum((cdf_sample[:ibin]) ** 2)
#         term_2 = np.sum((cdf_sample[ibin:] - 1) ** 2)
#         crps[isample] = (term_1 + term_2)  * np.diff(bins)[0]

#     return crps



# PROPER SCORING RULES - https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
# Original github - https://github.com/properscoring/properscoring/tree/master/properscoring
# CRPS is a generalization of mean absolute error and is easily calculated from a finite number of samples of a probability distribution.
    

def _discover_bounds(cdf_array, x_values, tolerance = 1e-7):
    """
    Uses scipy's general continuous distribution methods
    which compute the ppf from the cdf, then use the ppf
    to find the lower and upper limits of the distribution.
    """
    class CustomDistribution(rv_continuous):
        def __init__(self, x_values, cdf_values, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._ppf_interpolator = interp1d(cdf_values, x_values, bounds_error=False, fill_value=(x_values[0], x_values[-1]))

        def _ppf(self, q):
            return self._ppf_interpolator(q)

    bounds = np.zeros((len(cdf_array[0,:]), 2))

    # Loop through each sample and calculate the PPF for upper and lower tol bounds
    for i in range(len(bounds)):
        custom_scipy_shash = CustomDistribution(a=x_values[0], b=x_values[-1], x_values=x_values, cdf_values=cdf_array[:, i])
        ppf_value = custom_scipy_shash.ppf([tolerance, 1-tolerance])
        bounds[i, :] = ppf_value
    
    # print(f"The shape of the upper and lower distribution limits array is {bounds.shape}")
    return bounds



def _crps_single(target, cdf_func, cdf_array, x_values, xmin=None, xmax=None, tol=1e-6):
    """
     Parameters
    ----------
    x : np.ndarray
        Observations/Target associated with the forecast distribution cdf_func sample(s)
    cdf_func : callable or scipy.stats.distribution
        Function which returns the the cumulative density of the
        forecast distribution at value x.  This can also be an object with
        a callable cdf() method such as a scipy.stats.distribution object.
    xmin : np.ndarray or scalar
        The lower bounds for integration, this is required to perform
        quadrature.
    xmax : np.ndarray or scalar
        The upper bounds for integration, this is required to perform
        quadrature.
    tol : float , optional
        The desired accuracy of the CRPS, larger values will speed
        up integration. If tol is set to None, bounds errors or integration
        tolerance errors will be ignored.
    Returns:
        CRPS
    """
    # TODO: this function is pretty slow.  Look for clever ways to speed it up.
    
    # allow for directly passing in scipy.stats distribution objects.
    # cdf = getattr(cdf_func, 'cdf', cdf_func)
    cdf = cdf_func

    # Check if cdf is callable
    if not callable(cdf):
        raise TypeError("cdf_func must be a callable function or an object with a callable 'cdf' method.")
    
    # # if bounds aren't given, discover them
    if xmin is None or xmax is None:
        # Note that infinite values for xmin and xmax are valid, but
        # it slows down the resulting quadrature significantly.
        xmin, xmax = _discover_bounds(cdf)

    # Ignore specific warnings
    warnings.filterwarnings('ignore', message='Lower integral did not evaluate to within tolerance!')
    warnings.filterwarnings('ignore', message='Upper integral did not evaluate to within tolerance!')

    # make sure the bounds haven't clipped the cdf.
    if (tol is not None) and (cdf(xmin, cdf_array, x_values) >= tol) or (cdf(xmax, cdf_array, x_values) <= (1. - tol)):
        raise ValueError('CDF does not meet tolerance requirements at %s '
                         'extreme(s)! Consider using function defaults '
                         'or using infinities at the bounds. '
                         % ('lower' if cdf(xmin, cdf_array, x_values) >= tol else 'upper'))

    # CRPS = int_-inf^inf (F(y) - H(x))**2 dy
    #      = int_-inf^x F(y)**2 dy + int_x^inf (1 - F(y))**2 dy

    def lhs(y):
        # left hand side of CRPS integral
        return np.square(cdf(y, cdf_array, x_values))
    # use quadrature to integrate the lhs
    lhs_int, lhs_tol = integrate.quad(lhs, xmin, target)
    # make sure the resulting CRPS will be with tolerance
    if (tol is not None) and (lhs_tol >= 0.5 * tol):
        # raise ValueError('Lower integral did not evaluate to within tolerance! '
        #                  'Tolerance achieved: %f , Value of integral: %f \n'
        #                  'Consider setting the lower bound to -np.inf.' %
        #                  (lhs_tol, lhs_int))
        warnings.warn('Lower integral did not evaluate to within tolerance! \n')

    def rhs(y):
        # right hand side of CRPS integral
        return np.square(1. - cdf(y, cdf_array, x_values))
    rhs_int, rhs_tol = integrate.quad(rhs, target, xmax)
    # make sure the resulting CRPS will be with tolerance
    if (tol is not None) and (rhs_tol >= 0.5 * tol):
        warnings.warn('Upper integral did not evaluate to within tolerance! \n'
                         'Tolerance achieved: %f , Value of integral: %f \n'
                         'Consider setting the upper bound to np.inf or if '
                         'you already have, set warn_level to `ignore`.' %
                         (rhs_tol, rhs_int))

    return lhs_int + rhs_int

_crps_cdf = np.vectorize(_crps_single)


def crps_quadrature(x, cdf_or_dist, xmin=None, xmax=None, tol=1e-6):
    """
    Compute the continuously ranked probability score (CPRS) for a given
    forecast distribution (cdf) and observation (x) using numerical quadrature.

    This implementation allows the computation of CRPS for arbitrary forecast
    distributions. If gaussianity can be assumed ``crps_gaussian`` is faster.

    Parameters
    ----------
    x : np.ndarray
        Observations associated with the forecast distribution cdf_or_dist
    cdf_or_dist : callable or scipy.stats.distribution
        Function which returns the the cumulative density of the
        forecast distribution at value x.  This can also be an object with
        a callable cdf() method such as a scipy.stats.distribution object.
    xmin : np.ndarray or scalar
        The lower bounds for integration, this is required to perform
        quadrature.
    xmax : np.ndarray or scalar
        The upper bounds for integration, this is required to perform
        quadrature.
    tol : float , optional
        The desired accuracy of the CRPS, larger values will speed
        up integration. If tol is set to None, bounds errors or integration
        tolerance errors will be ignored.

    Returns
    -------
    crps : np.ndarray
        The continuously ranked probability score of an observation x
        given forecast distribution.
    """
    print("Inside crps_quadrature:")
    print("Type of cdf:", type(cdf_or_dist))

    return _crps_cdf(x, cdf_or_dist, xmin, xmax, tol)




class CumulativeSum:
    def __init__(self, axis=1):
        """
        Initialize the CumulativeSum class with the specified axis.

        Parameters:
        axis: the axis along which the cumulative sum is computed (default is 1)
        """
        self.axis = axis

    def __call__(self, p, target, x_values):
        """
        Make the CumulativeSum class instance callable.

        Parameters:
        p: the input array of PDF distributions
        target: the target values for which the CDF is computed

        Returns:
        A callable function that computes the CDF at given x values.
        """
        # check if target contains nans:
        if np.isnan(target).any():
            raise ValueError("The target array contains NaNs.")

        # Ensure the input PDF array is normalized
        p_sum = np.sum(p, axis=0, keepdims=True)
        if not np.allclose(p_sum, 1):
            p = p / p_sum

        cdf_array = np.cumsum(p, axis=0)

        if len(cdf_array.shape) == 1:
            cdf_array = cdf_array / cdf_array[-1]
            

        else:
            cdf_array = cdf_array / cdf_array[-1, :]
            assert cdf_array[-1, :].all() == 1 #check normalization

        # plt.figure()
        # for i in range(cdf_array.shape[1]):
        #     plt.plot(x_values, cdf_array[:, i], linewidth = 1)    
        # plt.savefig(str(config["figure_dir"]) + str(exp) + "cdfs.png", format='png', bbox_inches ='tight', dpi = 300)

        # Interpolate to compute the CDF at the target values for all samples (PRINTING FOR CHECK PURPOSES ONLY)
        # calculated_cdf_values = np.zeros(len(target))
        # for i, truth in enumerate(target.values):
        #     x1 = np.where(x_values <= truth)[-1]
        #     x2 = np.where(x_values >= truth)[0]

        #     cdf1 = cdf_array[x1, i]
        #     cdf2 = cdf_array[x2, i]
        #     print(f"cdf1 shape: {cdf1.shape}")
        #     print(f"cdf2 shape: {cdf2.shape}")

        #     cdfs = cdf1 + (cdf2 - cdf1) * (truth - x_values[x1]) / (x_values[x2] - x_values[x1])
        #     calculated_cdf_values[i] = np.round(cdfs, 6)
        #     #print(f"cdf for sample {i} (target = {np.round(truth, 3)}) : {calculated_cdf_values[i]}")
        
        def cdf_function(target, cdf1D, x_values):
            # Interpolate for each row of the 1-dimensional CDF array (single sample at a time)
            x1 = np.where(x_values <= target)[0][-1]
            x2 = np.where(x_values >= target)[0][0] # TODO: capture edge cases

            cdf1 = cdf1D[x1]
            cdf2 = cdf1D[x2]

            cdf = cdf1 + (cdf2 - cdf1) * (target - x_values[x1]) / (x_values[x2] - x_values[x1])
            calculated_cdf_value = np.round(cdf, 6)
         
            return calculated_cdf_value

        return cdf_function
    
    def cdf_array_output(p):
        p_sum = np.sum(p, axis=0, keepdims=True)
        if not np.allclose(p_sum, 1):
            p = p / p_sum

        cdf_array = np.cumsum(p, axis=0)
        cdf_array = cdf_array / cdf_array[-1, :]
        return cdf_array
    

def CRPScompare(crps_scores, crps_climatology_scores):
    """
    crps_scores: 1D array of CRPS scores
    crps_climatology_scores: 1D array of CRPS scores
    """
    # Calculate the mean CRPS scores for the forecast and climatology
    CRPS_forecast = np.round(np.mean(crps_scores), 4)
    CRPS_climatology = np.round(np.mean(crps_climatology_scores), 4)

    # Calculate proportion of forecast CRPS scores that are better (lower) than climatology CRPS average
    print(f"Length of CRPS scores: {len(crps_scores)}")
    print(f"Length of CRPS climatology scores: {len(crps_climatology_scores)}")

    better_than_climatology = 100 * np.sum(crps_scores < crps_climatology_scores) / len(crps_scores)
    print(f"Proportion of forecast CRPS scores that are better than climatology: {round(better_than_climatology, 2)}%")

    # Plot CRPS comparison Histogram:
    num_bins = 50
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(crps_scores, bins = num_bins, alpha = 0.4, color ='#26828e', label = f'Forecast CRPS Average: {CRPS_forecast} ')
    ax.hist(crps_climatology_scores,bins = num_bins, alpha = 0.4, color ='#f5a962', label = f'Climatology CRPS Average: {CRPS_climatology} ')
    ax.set_title("CRPS Comparison")
    ax.set_xlabel("CRPS Score")
    ax.set_ylabel("Frequency")
    ax.legend(markerscale = 9)
    plt.savefig(str(config["perlmutter_figure_dir"]) + "/" + str(exp) + "/CRPS_comparative_histogram.png", format='png', bbox_inches ='tight', dpi = 300)

    # return CRPS_forecast, CRPS_climatology


def calculateCRPS(output, target, x, config, climatology = None):
    """
    Comprehensive calculation of CRPS scores for the forecast output and climatology.

    """
    if climatology is not None:

        climatology_array, climatology_pdf = climatologyCDF(target, x, climatology)
        tol = config["databuilder"]["CRPS_tolerance"]

        cumulative_sum = CumulativeSum(axis=1)
        cdf_clima = cumulative_sum(climatology_pdf, target, x)

        bounds = _discover_bounds(climatology_array, x[:len(climatology_array)], tolerance =tol)

        print(len(target), climatology_array.shape[1], bounds.shape[0])

        CRPS_clima = np.zeros(len(target))
        # for i in range(len(target)+1):
        #     CRPS_clima[i] = _crps_single(target[i], cdf_clima, climatology_array[:,i], x, xmin = bounds[i,0], xmax= bounds[i,1], tol=tol)

        return CRPS_clima
    
    elif climatology == None:
        dist = Shash(output)
        p = dist.prob(x).numpy()
        print(f"Shape of all network outputs as PDFs: {p.shape}")

        # Instantiate CumulativeSum class to cread CDF ------------------
        cumulative_sum = CumulativeSum(axis=1)
        cdf = cumulative_sum(p, target = target, x_values=x)
        assert callable(cdf)

        cdf_array = CumulativeSum.cdf_array_output(p)

        # Discover Bounds for CDF distributions: -------------------------
        tol = config["databuilder"]["CRPS_tolerance"]
        # print(f"tolerance: {tol}")

        bounds = _discover_bounds(cdf_array, x, tolerance =tol)

        # Calculate Prediction CRPS scores: ----------------------------------------
        CRPS = np.zeros(len(target))
        for i in range(len(target)):
            CRPS[i] = _crps_single(target[i], cdf, cdf_array[:,i], x, xmin = bounds[i,0], xmax= bounds[i,1], tol=tol)

        plt.figure()
        plt.plot(CRPS)
        plt.xlabel('Time (Samples in Chronological Order)')
        plt.ylabel('CRPS Score')
        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/CRPS_score_time_series_all_sampes.png', format = 'png', dpi = 300)   
        return CRPS
