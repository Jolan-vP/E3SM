#!/usr/bin/env python3.10

import sys
# sys.path.insert(0, '/pscratch/sd/p/plutzner/E3SM')
sys.path.insert(0, '/Users/C830793391/Documents/Research/E3SM')
import os
os.environ['PROJ_DATA'] = "/pscratch/sd/p/plutzner/proj_data"
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
import pandas as pd

# import utils
import utils.filemethods as filemethods
import databuilder.data_loader as data_loader
from databuilder.data_loader import universaldataloader
import databuilder.data_generator as data_generator
import model.loss as module_loss
import model.metric as module_metric
import databuilder.data_loader as data_loader
from utils.filemethods import open_data_file

# from utils import utils
from shash.shash_torch import Shash
import analysis.calc_climatology as calc_climatology
from analysis import analysis_metrics
from utils.utils import filter_months
import analysis
from analysis import CRPS
from analysis import ENSO_indices_calculator
from analysis.calc_climatology import precip_regime
from analysis.ENSO_indices_calculator import idealENSOphases

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"xarray version = {xr.__version__}")
print(f"pytorch version = {torch.__version__}")

# ------------------------------------------------------------
# Determine Ideal ENSO Phases for Column Model Runs: 

# Open Observational Nino34 Index: 
nino_indices = open_data_file('/pscratch/sd/p/plutzner/E3SM/bigdata/NOAA_CPC_SST_NinoInidces.txt')


# Rename columns for compatibility with pd.to_datetime
nino_indices = nino_indices.rename(columns={'YR': 'year', 'MON': 'month'})

# Create a time coordinate using the year and month columns
time = pd.to_datetime(nino_indices[['year', 'month']].assign(day=1))

# Extract the 'ANOM.3' column and convert it to an xarray DataArray
nino34_obs = nino_indices['ANOM.3']
nino34_obs_xr = xr.DataArray(nino34_obs.values, coords=[time], dims=['time'])

# saveplot = '/pscratch/sd/p/plutzner/E3SM/COLUMN MODEL RUN PROJECT/'
saveplot = '/pscratch/sd/p/plutzner/E3SM/COLUMN MODEL RUN PROJECT/'
idealENSOphases(nino34_obs_xr, strength_threshold = 1.25, ens = 'OBS', percentile = 70, numberofeachphase= 1, plotfn = saveplot )

# ------------------------------------------------------------
# Open ERA5 Precip Data: 
era5_target_test_data = analysis_metrics.load_pickle('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp072_d_test.pkl')
era5_target_test_data = era5_target_test_data['y']

era5_training_target_data = analysis_metrics.load_pickle('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp072_d_train.pkl')
era5_training_target_data = era5_training_target_data['y']

era5_validation_target_data = analysis_metrics.load_pickle('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp072_d_val.pkl')
era5_validation_target_data = era5_validation_target_data['y']

total_era5_target_data = xr.concat([era5_training_target_data, era5_validation_target_data, era5_target_test_data], dim='time')
print(f"era5 total target precip shape: {total_era5_target_data.shape}")

# # Daily standardized precip data: 
# era5_precip_standard = total_era5_target_data / total_era5_target_data.std(dim='time')

# plt.figure(figsize=(10, 5))
# plt.plot(total_era5_target_data.time, era5_precip_standard, label='Standardized Precipitation')
# plt.title('Standardized Daily Precipitation Anomalies')
# plt.xlabel('Time')
# plt.ylabel('Standardized Precipitation Anomalies (mm/day)')
# # plt.legend()
# plt.savefig('/pscratch/sd/p/plutzner/E3SM/COLUMN MODEL RUN PROJECT/ERA5_precip_standardized_timeseries.png', format='png', dpi=250)

# daily_std_per_year = total_era5_target_data.groupby('time.year').std(dim='time')
# print(f"shape of daily std per year: {daily_std_per_year.shape}")

# Identify precipitation variance on monthly timescales: "Is each month dry relative to all months?"
monthly_precip_anomalies = total_era5_target_data.resample(time = 'ME').mean(dim='time')
analysis_metrics.save_pickle(monthly_precip_anomalies, '/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp072_monthly_precip_anomalies.pkl')
monthly_precip_anomalies = analysis_metrics.load_pickle('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp072_monthly_precip_anomalies.pkl')

monthly_std_per_year = monthly_precip_anomalies.groupby('time.year').std(dim='time')
print(f"shape of monthly std per year: {monthly_std_per_year.shape}")

# plt.figure(figsize=(10, 5))
# plt.plot(monthly_std_per_year['year'], monthly_std_per_year, label='Monthly Standard Deviation', color = '#3b528b')
# plt.axhline(y = np.mean(monthly_std_per_year), color = '#3b528b', linestyle='--', label='Mean Monthly Standard Deviation')
# plt.plot(daily_std_per_year['year'], daily_std_per_year, label='Daily Standard Deviation', color = '#21918c')
# plt.axhline(y = np.mean(daily_std_per_year), color = '#21918c', linestyle='--', label='Mean Daily Standard Deviation')

# # axvspan in brown for dates of monthly precip anomalies below 25th percentile
# for i in range(len(monthly_std_per_year['year'])):
#     if monthly_precip_anomalies.values[i] < np.percentile(monthly_precip_anomalies.values, 25):
#         if i == 5:
#             plt.axvspan(monthly_std_per_year['year'][i], monthly_std_per_year['year'][i+1], color='#bb8e3d', alpha=0.3, label = 'Monthly Precip Anomalies < 25th Percentile')
#         else: 
#             plt.axvspan(monthly_std_per_year['year'][i], monthly_std_per_year['year'][i+1], color='#bb8e3d', alpha=0.3)
#     elif monthly_precip_anomalies.values[i] > np.percentile(monthly_precip_anomalies.values, 75):
#         if i == 1:
#             plt.axvspan(monthly_std_per_year['year'][i], monthly_std_per_year['year'][i+1], color='#299b7d', alpha=0.3, label = 'Monthly Precip Anomalies > 75th Percentile')
#         else:
#             plt.axvspan(monthly_std_per_year['year'][i], monthly_std_per_year['year'][i+1], color='#299b7d', alpha=0.3)


# plt.title('Daily and Monthly Precipitation Anomaly Standard Deviations')
# plt.xlabel('Time')
# plt.ylabel('Standard Deviation (mm/day)')
# plt.ylim([-0.1, 2.3])
# years = monthly_std_per_year['year'].values
# plt.xticks(ticks=years[::5])  # show every 5th year
# plt.legend(fontsize = 8)
# plt.tight_layout()
# plt.savefig('/pscratch/sd/p/plutzner/E3SM/COLUMN MODEL RUN PROJECT/ERA5_precip_daily_monthly_STD_timeseries_shaded.png', format='png', dpi=250)



# # ------------------------------------------------------------

enso_dates_dict = {
    "El Nino": pd.to_datetime(['1982-12-01', '1992-01-01', '2009-12-01']),
    "La Nina": pd.to_datetime(['1998-12-01', '2008-02-01', '2007-11-01']),
    "Neutral": pd.to_datetime(['1990-02-01', '1994-07-01', '2004-05-01'])
}

# Print the value of the enso index for each date in the dictionary
for event_type, dates in enso_dates_dict.items():
    for date in dates:
        print(f"{event_type} event on {date.strftime('%Y-%m-%d')}: {nino34_obs_xr.sel(time=date, method='nearest').values:.2f}")
        # print(f"{event_type} event on {date.strftime('%Y-%m-%d')}: {total_era5_target_data.sel(time=date).values:.2f}")
# Print the value of the precipitaion anomaly for each date in the dictionary
for event_type, dates in enso_dates_dict.items():
    for date in dates:
        print(f"{event_type} event on {date.strftime('%Y-%m-%d')}: {monthly_precip_anomalies.sel(time=date, method='nearest').values:.2f} mm/day")
        # print(f"{event_type} event on {date.strftime('%Y-%m-%d')}: {total_era5_target_data.sel(time=date).values:.2f}")

# enso_dates_dict = {
#     "El Nino": pd.to_datetime(['1982-12-01', '1997-11-01', '2015-11-01', '1992-01-01', '2023-12-01', '2002-12-01', '1987-09-01', '2009-12-01']),
#     "La Nina": pd.to_datetime(['1988-11-01', '1998-12-01', '2008-01-01', '1999-02-01', '1984-12-01', '2011-01-01', '1985-01-01', '1984-11-01', '2008-02-01', '2007-11-01', '2010-10-01']),
#     "Neutral": pd.to_datetime(['1994-01-01', '1990-02-01', '1990-09-01', '2004-04-01', '2001-08-01', '1993-08-01', '1994-07-01', '2004-06-01', '2004-05-01', '1993-09-01', '1993-10-01', '1991-01-01', '1991-02-01', '1991-04-01'])
# }

low_thresh = np.percentile(monthly_precip_anomalies, 25)
high_thresh = np.percentile(monthly_precip_anomalies, 75)

monthly_precip_anomalies_smoothed = monthly_precip_anomalies.rolling(time=5, center=True).mean()
print(f"monthly precip anomalies time: {monthly_precip_anomalies.time}")

plt.figure(figsize=(12, 5))
plt.plot(monthly_precip_anomalies.time, monthly_precip_anomalies_smoothed, label='Monthly Mean Precip', color='#3b528b')
plt.axhline(low_thresh, color='#bb8e3d', linestyle='--', label='25th Percentile')
plt.axhline(high_thresh, color='#299b7d', linestyle='--', label='75th Percentile')
# plt.scatter(enso_dates_dict['El Nino'], total_era5_target_data.sel(time=enso_dates_dict['El Nino']), color='red', label='El Nino Events', marker='*')
# plt.scatter(enso_dates_dict['La Nina'], total_era5_target_data.sel(time=enso_dates_dict['La Nina']), color='blue', label='La Nina Events', marker='*')
# plt.scatter(enso_dates_dict['Neutral'], total_era5_target_data.sel(time=enso_dates_dict['Neutral']), color='grey', label='Neutral Events', marker='*')
plt.scatter(enso_dates_dict['El Nino'], monthly_precip_anomalies.sel(time=enso_dates_dict['El Nino'], method = 'nearest'), color='red', label='El Nino Events', marker='o')
plt.scatter(enso_dates_dict['La Nina'], monthly_precip_anomalies.sel(time=enso_dates_dict['La Nina'], method = 'nearest'), color='blue', label='La Nina Events', marker='o')
plt.scatter(enso_dates_dict['Neutral'], monthly_precip_anomalies.sel(time=enso_dates_dict['Neutral'], method = 'nearest'), color='grey', label='Neutral Events', marker='o')
# label each point with the date
for event_type, dates in enso_dates_dict.items():
    for date in dates:
        plt.annotate(date.strftime('%Y-%m'), (date, monthly_precip_anomalies.sel(time=date, method='nearest').values), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
start_date = pd.to_datetime('1980-01-01')
end_date = pd.to_datetime('2023-01-01')
plt.xlim(start_date, end_date)
# # Shade areas below 25th and above 75th percentile
# for t, val in zip(monthly_precip_anomalies.time.values, monthly_precip_anomalies.values):
#     if val < low_thresh:
#         plt.axvspan(t - np.timedelta64(15, 'D'), t + np.timedelta64(15, 'D'), color='#bb8e3d', alpha=0.3)
#     elif val > high_thresh:
#         plt.axvspan(t - np.timedelta64(15, 'D'), t + np.timedelta64(15, 'D'), color='#299b7d', alpha=0.3)

plt.title('Monthly Precipitation Anomalies')
plt.xlabel('Time')
plt.ylabel('Precipitation Anomaly (mm/day)')
plt.legend()
plt.tight_layout()
plt.savefig('/pscratch/sd/p/plutzner/E3SM/COLUMN MODEL RUN PROJECT/ERA5_precip_monthly_anomalies_DryWetNeutralDates.png', format='png', dpi=250)








# # ------------------------------------------------------------

# # Calculate std for all Januarys, all Februarys, etc.
# monthly_std = np.empty([12])
# for month in range(1, 13):
#     monthly_std[month-1] = total_era5_target_data.sel(time=total_era5_target_data['time.month'] == month).std(dim='time')

# plt.figure(figsize=(10, 5))
# plt.plot(range(1, 13), monthly_std, marker='o', label='Monthly Standard Deviation', color = '#3b528b')
# plt.axhline(y = np.mean(monthly_std), color = '#3b528b', linestyle='--', label='Mean Monthly Standard Deviation')
# plt.title('Monthly Precipitation Anomaly Standard Deviations')
# plt.xlabel('Month')
# plt.ylabel('Standard Deviation (mm/day)')
# plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
# plt.tight_layout()
# plt.savefig('/pscratch/sd/p/plutzner/E3SM/COLUMN MODEL RUN PROJECT/ERA5_precip_monthly_STD_timeseries.png', format='png', dpi=250)

# ------------------------------------------------------------
# ENSO_ens1 = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/ENSO_ne30pg2_HighRes/nino.member0101.nc')
# ENSO_ens2 = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/ENSO_ne30pg2_HighRes/nino.member0151.nc')
# ENSO_ens3 = xr.open_dataset('/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/ENSO_ne30pg2_HighRes/nino.member0201.nc')

# nino34_ens1 = ENSO_ens1['nino34']
# nino34_ens2 = ENSO_ens2['nino34']
# nino34_ens3 = ENSO_ens3['nino34']






# low_thresh = np.percentile(monthly_precip_anomalies, 25)
# high_thresh = np.percentile(monthly_precip_anomalies, 75)

# plt.figure(figsize=(12, 5))
# plt.plot(monthly_precip_anomalies.time, monthly_precip_anomalies, label='Monthly Mean Precip', color='#3b528b')
# plt.axhline(low_thresh, color='#bb8e3d', linestyle='--', label='25th Percentile')
# plt.axhline(high_thresh, color='#299b7d', linestyle='--', label='75th Percentile')

# # # Shade areas below 25th and above 75th percentile
# # for t, val in zip(monthly_precip_anomalies.time.values, monthly_precip_anomalies.values):
# #     if val < low_thresh:
# #         plt.axvspan(t - np.timedelta64(15, 'D'), t + np.timedelta64(15, 'D'), color='#bb8e3d', alpha=0.3)
# #     elif val > high_thresh:
# #         plt.axvspan(t - np.timedelta64(15, 'D'), t + np.timedelta64(15, 'D'), color='#299b7d', alpha=0.3)

# plt.title('Monthly Precipitation Anomalies with Percentile Shading')
# plt.xlabel('Time')
# plt.ylabel('Precipitation (mm/day)')
# plt.legend()
# plt.tight_layout()
# plt.savefig('/pscratch/sd/p/plutzner/E3SM/COLUMN MODEL RUN PROJECT/ERA5_precip_monthly_anomalies_shaded.png', format='png', dpi=250)