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


def crps_basic_numeric(pred, target_y, bins, single_cdf=False):
    # see alternative formulation in ``crps_sample_score``
    
    crps = np.zeros((len(target_y),))
    for isample in np.arange(0, len(target_y)):
        ibin = np.argmin(np.abs(bins[:-1] - target_y[isample]))

        pdf, __ = np.histogram(pred[:, isample], bins, density=True)  # PDF = shash.dist()
        pdf = pdf / (np.sum(pdf) ) #* np.diff(bins)[0]
        cdf = np.cumsum(pdf) #/ np.sum(pdf)
        
        
        cdf_sample = cdf
        term_1 = np.sum((cdf_sample[:ibin]) ** 2)
        term_2 = np.sum((cdf_sample[ibin:] - 1) ** 2)
        crps[isample] = (term_1 + term_2)  * np.diff(bins)[0]

    return crps



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
    
    print(f"The shape of the upper and lower distribution limits array is {bounds.shape}")
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

    # print("Inside _crps_cdf_single:")
    # print("Type of cdf:", type(cdf))
    # print("Is cdf callable?:", callable(cdf))

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
        # Ensure the input PDF array is normalized
        p_sum = np.sum(p, axis=0, keepdims=True)
        if not np.allclose(p_sum, 1):
            print("Normalizing input PDF array")
            p = p / p_sum

        cdf_array = np.cumsum(p, axis=0)
        cdf_array = cdf_array / cdf_array[-1, :]
        plt.figure()
        for i in range(cdf_array.shape[1]):
            plt.plot(x_values, cdf_array[:, i], linewidth = 1)      
        print(f"cdf_array shape within CumulativeSum: {cdf_array.shape}")
        print(f"cdf_array last row: {cdf_array[-1,:]}")  # Debug print to check normalization

        # Interpolate to compute the CDF at the target values for all samples (PRINTING CHECK PURPOSES ONLY)
        calculated_cdf_values = np.zeros(len(target))
        for i, truth in enumerate(target):
            x1 = np.where(x_values <= truth)[0][-1]
            x2 = np.where(x_values >= truth)[0][0]
            cdf1 = cdf_array[x1, i]
            cdf2 = cdf_array[x2, i]
            cdfs = cdf1 + (cdf2 - cdf1) * (truth - x_values[x1]) / (x_values[x2] - x_values[x1])
            calculated_cdf_values[i] = np.round(cdfs, 6)
            #print(f"cdf for sample {i} (target = {np.round(truth, 3)}) : {calculated_cdf_values[i]}")
        
        def cdf_function(target, cdf1D, x_values):
            # Interpolate for each row of the 1-dimensional CDF array (single sample at a time)
            x1 = np.where(x_values <= target)[0][-1]
            x2 = np.where(x_values >= target)[0][0]
            cdf1 = cdf1D[x1]
            cdf2 = cdf1D[x2]
            cdf = cdf1 + (cdf2 - cdf1) * (target - x_values[x1]) / (x_values[x2] - x_values[x1])
            calculated_cdf_value = np.round(cdf, 6)
            #print(f"cdf value at target ({target}) : {calculated_cdf_value}")
            return calculated_cdf_value

        return cdf_function
    
    def cdf_array_output(p):
        p_sum = np.sum(p, axis=0, keepdims=True)
        if not np.allclose(p_sum, 1):
            p = p / p_sum

        cdf_array = np.cumsum(p, axis=0)
        cdf_array = cdf_array / cdf_array[-1, :]
        return cdf_array