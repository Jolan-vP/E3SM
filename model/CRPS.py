"""
CRPS Module: 

Functions: -------------------
    crps_basic_numeric

(Proper Scoring Functions:)
    crps_gaussian
    crps_discover_bounds
    crps_cdf_single
    crps_quadrature

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
from scipy.stats import norm


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


def crps_gaussian(x, mu, sig, grad=False):
    """
    Computes the CRPS of observations x relative to normally distributed
    forecasts with mean, mu, and standard deviation, sig.

    CRPS(N(mu, sig^2); x)

    Formula taken from Equation (5):

    Calibrated Probablistic Forecasting Using Ensemble Model Output
    Statistics and Minimum CRPS Estimation. Gneiting, Raftery,
    Westveld, Goldman. Monthly Weather Review 2004

    http://journals.ametsoc.org/doi/pdf/10.1175/MWR2904.1

    Parameters
    ----------
    x : scalar or np.ndarray
        The observation or set of observations.
    mu : scalar or np.ndarray
        The mean of the forecast normal distribution
    sig : scalar or np.ndarray
        The standard deviation of the forecast distribution
    grad : boolean
        If True the gradient of the CRPS w.r.t. mu and sig
        is returned along with the CRPS.

    Returns
    -------
    crps : scalar or np.ndarray or tuple of
        The CRPS of each observation x relative to mu and sig.
        The shape of the output array is determined by numpy
        broadcasting rules.
    crps_grad : np.ndarray (optional)
        If grad=True the gradient of the crps is returned as
        a numpy array [grad_wrt_mu, grad_wrt_sig].  The
        same broadcasting rules apply.
    """
    x = np.asarray(x)
    mu = np.asarray(mu)
    sig = np.asarray(sig)
    # standadized x
    sx = (x - mu) / sig
    # some precomputations to speed up the gradient
    pdf = _normpdf(sx)
    cdf = _normcdf(sx)
    pi_inv = 1. / np.sqrt(np.pi)
    # the actual crps
    crps = sig * (sx * (2 * cdf - 1) + 2 * pdf - pi_inv)
    if grad:
        dmu = 1 - 2 * cdf
        dsig = 2 * pdf - pi_inv
        return crps, np.array([dmu, dsig])
    else:
        return crps
    

def _discover_bounds(cdf, tol=1e-7):
    """
    Uses scipy's general continuous distribution methods
    which compute the ppf from the cdf, then use the ppf
    to find the lower and upper limits of the distribution.
    """
    class DistFromCDF(stats.distributions.rv_continuous):
        def cdf(self, x):
            return cdf(x)
    dist = DistFromCDF()
    # the ppf is the inverse cdf
    lower = dist.ppf(tol)
    upper = dist.ppf(1. - tol)
    return lower, upper


def _crps_cdf_single(x, cdf_or_dist, xmin=None, xmax=None, tol=1e-6):
    """
    See crps_cdf for docs.
    """
    # TODO: this function is pretty slow.  Look for clever ways to speed it up.

    # allow for directly passing in scipy.stats distribution objects.
    cdf = getattr(cdf_or_dist, 'cdf', cdf_or_dist)
    assert callable(cdf)

    # if bounds aren't given, discover them
    if xmin is None or xmax is None:
        # Note that infinite values for xmin and xmax are valid, but
        # it slows down the resulting quadrature significantly.
        xmin, xmax = _discover_bounds(cdf)

    # make sure the bounds haven't clipped the cdf.
    if (tol is not None) and (cdf(xmin) >= tol) or (cdf(xmax) <= (1. - tol)):
        raise ValueError('CDF does not meet tolerance requirements at %s '
                         'extreme(s)! Consider using function defaults '
                         'or using infinities at the bounds. '
                         % ('lower' if cdf(xmin) >= tol else 'upper'))

    # CRPS = int_-inf^inf (F(y) - H(x))**2 dy
    #      = int_-inf^x F(y)**2 dy + int_x^inf (1 - F(y))**2 dy
    def lhs(y):
        # left hand side of CRPS integral
        return np.square(cdf(y))
    # use quadrature to integrate the lhs
    lhs_int, lhs_tol = integrate.quad(lhs, xmin, x)
    # make sure the resulting CRPS will be with tolerance
    if (tol is not None) and (lhs_tol >= 0.5 * tol):
        raise ValueError('Lower integral did not evaluate to within tolerance! '
                         'Tolerance achieved: %f , Value of integral: %f \n'
                         'Consider setting the lower bound to -np.inf.' %
                         (lhs_tol, lhs_int))

    def rhs(y):
        # right hand side of CRPS integral
        return np.square(1. - cdf(y))
    rhs_int, rhs_tol = integrate.quad(rhs, x, xmax)
    # make sure the resulting CRPS will be with tolerance
    if (tol is not None) and (rhs_tol >= 0.5 * tol):
        raise ValueError('Upper integral did not evaluate to within tolerance! \n'
                         'Tolerance achieved: %f , Value of integral: %f \n'
                         'Consider setting the upper bound to np.inf or if '
                         'you already have, set warn_level to `ignore`.' %
                         (rhs_tol, rhs_int))

    return lhs_int + rhs_int

_crps_cdf = np.vectorize(_crps_cdf_single)


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
    return _crps_cdf(x, cdf_or_dist, xmin, xmax, tol)
