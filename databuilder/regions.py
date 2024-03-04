"""
Region Definitions

Functions: -------------------
compute_global_mean(da)
extract_region(data, region, dir, land_only=False, lat=None, lon=None)

"""

import xarray as xr
import numpy as np
import pandas as pd
import regionmask
import matplotlib.pyplot as plt


def extract_region(data, region=None, dir, lat=None, lon=None, drop=True):
    if region is None: 
        min_lon, max_lon = [0, 360]
        min_lat, max_lat = [-90, 90]
    else: 
        min_lon, max_lon = region[:2]
        min_lon, max_lon = region[:2]

    if isinstance(data, xr.DataArray):
        mask_lon = (data.lon >= min_lon) & (data.lon <= max_lon)
        mask_lat = (data.lat >= min_lat) & (data.lat <= max_lat)
        data_masked = data.where(mask_lon & mask_lat, drop=True)
        return(
            data_masked, 
            data_masked["lat"]
            data_masked["lon"] #TODO: Need to include .to_numpy() and astype(float32) anymore due to added flexibility in pytorch input types??
        )
    else: 
        raise NotImplementedError #need this anymore?