"""
Functions: ------------------
crps_climatology(test_target, x, preprocessed_data_file)

"""

import gzip
import pickle
import numpy as np


def crps_climatology(test_target, x, preprocessed_data_file):
    """
    test_target: 2D array of target values
    x: 1D array of bin edges
    """
    # Create pdf distribution of climatology, calculate CDF, and repeat same CDF across all samples
    exp006_processeddata = preprocessed_data_file
    with gzip.open(exp006_processeddata, "rb") as obj1:
        data = pickle.load(obj1)
        climatology = data["y"] # pulling all target values from processed data
        print(f"Climatologial Mean = {np.mean(climatology)}")

    pdf, __ = np.histogram(climatology, bins = x, density=True)
    pdf = pdf / (np.sum(pdf) * np.diff(x)[0])
    cdf_base = np.cumsum(pdf) / np.sum(pdf)
    cdf_base = cdf_base[:len(x-1)]

    # Echo cdf_base across the depth of all samples: 
    climatology_array = np.tile(cdf_base, (len(test_target), 1))
    # Flip climatology array to match the shape of the cdf_array
    climatology_array = np.transpose(climatology_array)
    print(climatology_array.shape)

    return climatology_array
