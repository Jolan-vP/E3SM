"""Metrics for training and evaluation.

Functions
---------
custom_mae(output, target)
iqr_capture(output, target)
sign_test(output, target)
pit(output, target)

"""

import torch
from shash.shash_torch import Shash
import numpy as np


def custom_mae(output, target):
    """Compute the prediction mean absolute error.
    The "predicted value" is the median of the conditional distribution.

    """
    with torch.no_grad():

        assert len(output[:, 0]) == len(target)

        dist = Shash(output)
        return torch.mean(torch.abs(dist.median() - target)).item()


def iqr_capture(output, target):
    """Compute the fraction of true values between the 25 and 75 percentiles
    (i.e. the interquartile range).

    """
    with torch.no_grad():
        assert len(output[:, 0]) == len(target)

        dist = Shash(output)
        lower = dist.quantile(torch.tensor(0.25))
        upper = dist.quantile(torch.tensor(0.75))
        count = torch.sum(
            torch.logical_and(torch.greater(target, lower), torch.less(target, upper))
        ).item()

        return count / len(target)


def sign_test(output, target):
    """Compute the fraction of true values above the median."""
    with torch.no_grad():
        assert len(output[:, 0]) == len(target)

        dist = Shash(output)
        median = dist.quantile(torch.tensor(0.50))
        count = torch.sum(torch.greater(target, median)).item()

        return count / len(target)


def pit(output, target):
    """Compute the PIT (Probability Integral Transform) histogram."""
    bins = np.linspace(0, 1, 11)

    dist = Shash(output)
    F = dist.cdf(target)
    pit_hist = np.histogram(
        F,
        bins,
        weights=np.ones_like(F) / float(len(F)),
    )

    B = len(pit_hist[0])
    D = np.sqrt(1 / B * np.sum((pit_hist[0] - 1 / B) ** 2))
    EDp = np.sqrt((1.0 - 1 / B) / (target.shape[0] * B))

    return bins, pit_hist, D, EDp


# def crps(cdf, y, bins, single_cdf=False):
#     # see alternative formulation in ``crps_sample_score``

#     crps = np.zeros((len(y),))
#     for isample in np.arange(0, len(y)):
#         ibin = np.argmin(np.abs(bins[:-1] - y[isample]))
#         if single_cdf:
#             cdf_sample = cdf
#         else:
#             cdf_sample = cdf[isample, :]
#         term_1 = np.sum((cdf_sample[:ibin]) ** 2)
#         term_2 = np.sum((cdf_sample[ibin:] - 1) ** 2)
#         crps[isample] = term_1 + term_2

#     return crps * np.diff(bins)[0]


# def compute_crps(pred, y, bins, climatology=False, parametric=True):
#     if climatology:
#         pdf, __ = np.histogram(pred, bins, density=True)
#         pdf = pdf / (np.sum(pdf) * np.diff(bins)[0])
#         cdf_base = np.cumsum(pdf) / np.sum(pdf)
#         return crps(cdf_base, y, bins, climatology)

#     elif parametric:
#         return crps(pred, y, bins)

#     else:
#         crps_out = np.zeros((len(y),))
#         for isample in range(len(y)):
#             pdf, __ = np.histogram(pred[isample, :], bins, density=True)  # PDF = shash.dist()
#             pdf = pdf / (np.sum(pdf) * np.diff(bins)[0])
#             cdf = np.cumsum(pdf) / np.sum(pdf)
#             crps_out[isample] = crps(cdf[np.newaxis, :], y[isample], bins)

#         return crps_out


def computeCRPS(x, output):
    """
    Inputs: 
    - SHASH parameters [mu,   sigma,   tau,   gamma]
    - Target parameter [ x ]

    Outputs: 
    - CRPS value: area between distribution CDF and discreet CDF curves

    """

    crps_lower = Shash(output).cdf(x) # area under SHASH CDF up until target value

                # heaviside - (1- crps_lower) = (x_inf - x) - (1 - crps_lower) = x_delta - (1 - crps_lower)
    crps_upper =  x_delta - (1 - crps_lower) # area between heaviside function and SHASH CDF 

    CRPS =  (crps_lower + crps_upper) ** 2

    return CRPS