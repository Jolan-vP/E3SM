"""
Functions: ------------------
iENSO(SST_fn)

"""

import matplotlib.pyplot as plt
import numpy as np  
from shash.shash_torch import Shash
from utils import filemethods
import xarray as xr
import gzip
import pickle
from databuilder.data_loader import universaldataloader
from analysis.analysis_metrics import save_pickle, load_pickle
import cartopy
from cartopy.crs import PlateCarree
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cftime
import pandas as pd
from datetime import datetime

def identify_nino_phases(nino34_filename, config, threshold=0.4, window=6, lagtime = None, smoothing_length = None):
    """
    Function to identify El Niño, La Niña, and Neutral phases based on Nino 3.4 SST index.
    The Niño 3.4 index typically uses a 5-month running mean, and El Niño or La  Niña events are defined when the  
    Niño 3.4 SSTs exceed +/- 0.4C for a period of six months or more." 
    https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni

    Parameters:
    - nino34_index (numpy array): Time series of DAILY Nino 3.4 SST index values.
    - threshold (float): Threshold for El Niño/La Niña classification.
    - window (int): Number of consecutive months for classification (default 6 months).
    
    Returns:
    - phase_array (numpy array): Array with 3 columns (El Niño, La Niña, Neutral) and rows corresponding to time steps.
    """
    if config["data_source"] == "ERA5": # ERA5 ENSO data is monthly 
        print("Data source is ERA5 - daily ENSO data")
        Nino34 = xr.open_dataset(nino34_filename)
        Nino34 = Nino34.value
        nino34_index = Nino34.sel(time = slice(str(config["databuilder"]["input_years"][0]), str(config["databuilder"]["input_years"][1])))
        window_length = window * 30 # 6 months of daily data

    elif config["data_source"] == "E3SM":
        print("Data source is E3SM - daily ENSO data")
        Nino34 = xr.open_dataset(nino34_filename)
        Nino34 = Nino34.nino34
        # Create desired full time range (e.g., 1850-01-01 to 2014-12-31, daily)
        full_time = xr.cftime_range(start='1850-01-01', end='2014-12-31', freq='D', calendar = 'noleap')
        # Reindex to full time range, filling the first and last two weeks of the entire dataset with NANs. Due to monthly -> daily interpolation of centered monthly means, the first and last two weeks are missing. 
        Nino34 = Nino34.reindex(time=full_time)
        nino34_index = Nino34.sel(time = slice(str(config["databuilder"]["input_years"][0]), str(config["databuilder"]["input_years"][1])))

        n = nino34_index.values.shape[0]  # Number of time steps

        if n > 25000: #E3SM ENSO data is daily
           window_length = 6 * 30
        elif n < 15000:
            window_length = window
            print(f"Caution, seems like Monthly E3SM data is being used, double check that!")
 
    n = nino34_index.values.shape[0]  # Number of time steps
    # Initialize array to hold the phase classifications
    phase_array = np.zeros((n, 3), dtype=int)  # Columns: [El Niño, La Niña, Neutral]

    # Loop through the Nino3.4 index using a sliding window
    for i in range(n - window_length + 1):
        window_slice = nino34_index[i:i + window_length]
        
        if np.all(window_slice > threshold):
            # Mark El Niño (6-month period all > threshold)
            phase_array[i:i + window_length, 0] = 1
        elif np.all(window_slice < -threshold):
            # Mark La Niña (6-month period all < -threshold)
            phase_array[i:i + window_length, 1] = 1
        else:
            # Mark neutral for all other periods
            phase_array[i:i + window_length, 2] = 1
    
    index_array_daily = phase_array.copy()

    # save index array daily as an xarray dataset with a time component (same as the original ENSO dataset)
    # Assign daily index array the same time coordinate from the original dataset
    index_array_dailyXR = xr.DataArray(
        index_array_daily, 
        dims=["time", "variables"],  # Specify the dimensions
        coords={
            "time": nino34_index.coords["time"],  # Use the 'time' from ENSO index
            "variables": ["El Nino Phase", "La Nina Phase", "Neutral Phase"] 
        },
        attrs = {"description" : "ENSO phases by each day carrying the time coord"}
    )
    # print(f"index_array_dailyXR shape: {index_array_dailyXR.shape}")
    # print(f"index_array_dailyXR: {index_array_dailyXR}")

    # filter indices by target months, cut leads/lags, smoothing length, etc. 
    filtered_daily_indicesXR = universaldataloader(index_array_dailyXR, config, target_only = True) 

    # initialize dict to hold timestamps for each ENSO phase
    ENSO_dates_dict = {
        "El Nino" : [],
        "La Nina" : [], 
        "Neutral" : []
    }

    for col in range(filtered_daily_indicesXR.shape[1]):
        for row in range(filtered_daily_indicesXR.shape[0]):
            if filtered_daily_indicesXR[row, col] != 0:
                date = filtered_daily_indicesXR.time.isel(time = row)
                if col == 0:
                    ENSO_dates_dict['El Nino'].append(date.values)
                elif col == 1:
                    ENSO_dates_dict['La Nina'].append(date.values)
                elif col == 2:
                    ENSO_dates_dict['Neutral'].append(date.values)

    return ENSO_dates_dict


def ENSO_CRPS(daily_enso_dates, crps_scores, target_time, config): 

    # Ensure that EN, LN, N dates all fall within the test set period: 
    start_year = config["databuilder"]["test_years"][0]
    end_year = config["databuilder"]["test_years"][1]

    # Convert to pandas
    elnino_dates = [pd.Timestamp(d) for d in daily_enso_dates["El Nino"] if start_year <= pd.Timestamp(d).year <= end_year]
    lanina_dates = [pd.Timestamp(d) for d in daily_enso_dates["La Nina"] if start_year <= pd.Timestamp(d).year <= end_year]
    neutral_dates = [pd.Timestamp(d) for d in daily_enso_dates["Neutral"] if start_year <= pd.Timestamp(d).year <= end_year]

    # Convert to numpy datetime64
    lanina_dates_np = np.array([np.datetime64(date) for date in lanina_dates])
    elnino_dates_np = np.array([np.datetime64(date) for date in elnino_dates])
    neutral_dates_np = np.array([np.datetime64(date) for date in neutral_dates])

    ENSO_dict = {
        "elnino" : elnino_dates_np, 
        "lanina" : lanina_dates_np, 
        "neutral": neutral_dates_np, 
        "CRPS":    crps_scores
    }
    save_pickle(ENSO_dict, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "ENSO_indices_CRPS.pkl")

    # Create crps xarray object with target time coordinate 
    crps_scores = xr.DataArray(crps_scores, coords=[target_time.values], dims=["time"], attrs={"description": "CRPS scores"})

    # Find intersections between dates and target times
    elnino_common = np.intersect1d(elnino_dates_np, target_time.values)
    lanina_common = np.intersect1d(lanina_dates_np, target_time.values)
    neutral_common = np.intersect1d(neutral_dates_np, target_time.values)

    CRPS_elnino = round(crps_scores.sel(time = elnino_common).values.mean(), 5)
    CRPS_lanina = round(crps_scores.sel(time = lanina_common).values.mean(), 5)
    CRPS_neutral = round(crps_scores.sel(time = neutral_common).values.mean(), 5)

    print(f"El Nino average CRPS across all samples: {np.round(CRPS_elnino, 4)}")
    print(f"La Nina average CRPS across all samples: {np.round(CRPS_lanina, 4)}")
    print(f"Neutral average CRPS across all samples: {np.round(CRPS_neutral, 4)}")


def idealENSOphases(nino34index, strength_threshold = 2, ens = None, percentile = None, numberofeachphase = None, plotfn = None):
    """
    Function to find the dates corresponding to 'idealized' ENSO phases for OBSERVATIONAL Nino 3.4 index.
    """
    threshold = 0.4  # Threshold for El Niño/La Niña classification
    window = 6  # 6 months consecutively 

    n = len(nino34index)

    # Mask out dates if there are fewer than 6 consecutive values in a row 
    # Loop through the Nino3.4 index using a sliding window
    threshold_mask = np.zeros(n, dtype=int)
    for i in range(n - window + 1):
        window_slice = nino34index[i:i + window]
        if np.all(window_slice > threshold) or np.all(window_slice < -threshold):
            threshold_mask[i:i + window] = 1

    # Apply threshold_mask to nino34 index: 
    nino34_threshold_masked = np.ma.masked_where(threshold_mask == 0, nino34index)

    # Replace masked values with zero
    nino34_filled = nino34_threshold_masked.filled(0)

    # Convert the filled array back to an xarray DataArray
    nino34_filled_xr = xr.DataArray(nino34_filled, coords=nino34index.coords, dims=nino34index.dims)

    # Now there are lots of values separated by masked values
    # For every block of consecutive values, find the maximum value in that block 
    # store the values as an xarray with the same time coordinate as the original dataset

    current_block = []
    max_values = []
    max_dates = []
   # Iterate through each index in the range of n
    for i in range(n):
        if threshold_mask[i] == 1:
            # If threshold_mask is 1, add the current value to the current block
            current_block.append((nino34_filled_xr[i], nino34index.time.values[i]))
        else:
            if current_block:
                # Filter values within the block that are greater than 2 or less than -2
                filtered_block = [(value, date) for value, date in current_block if value > strength_threshold or value < -strength_threshold]
                if filtered_block:
                    # If the filtered block is not empty, calculate the three highest values and their corresponding dates
                    top_three = sorted(filtered_block, key=lambda x: np.abs(x[0]), reverse=True)[:3]
                    for value, date in top_three:
                        max_values.append(value)
                        max_dates.append(date)
                    
                    # max_value, max_date = max(filtered_block, key=lambda x: np.abs(x[0]))
                    # max_values.append(max_value)
                    # max_dates.append(max_date)
                    # If the max value switches from positive to negative or vice versa within the block, record three max values and three dates:
                    if len(filtered_block) > 1:
                        for j in range(1, len(filtered_block)):
                            if np.sign(filtered_block[j][0]) != np.sign(filtered_block[j-1][0]):
                                max_values.append(filtered_block[j][0])
                                max_dates.append(filtered_block[j][1])
                                
                # Reset the current block for the next contiguous block
                current_block = []

    # Handle the case where the last block extends to the end of the array
    if current_block:
        filtered_block = [(value, date) for value, date in current_block if value >  strength_threshold or value < - strength_threshold]
        if filtered_block:
            max_value, max_date = max(filtered_block, key=lambda x: np.abs(x[0]))
            max_values.append(max_value)
            max_dates.append(max_date)

    max_values = np.array(max_values)
    max_dates = np.array(max_dates)

    # Print max dates and values separately for positive and negative values
    positive_max_dates = max_dates[max_values > 0]
    positive_max_values = max_values[max_values > 0]
    negative_max_dates = max_dates[max_values < 0]
    negative_max_values = max_values[max_values < 0]
    # print(f"The maximum values of the Nino 3.4 index that are greater than {strength_threshold} are: {positive_max_values}")
    # print(f"The dates for the maximum values of the Nino 3.4 index that are greater than {strength_threshold} are: {positive_max_dates}")
    # print(f"The maximum values of the Nino 3.4 index that are less than {-strength_threshold} are: {negative_max_values}")
    # print(f"The dates for the maximum values of the Nino 3.4 index that are less than {-strength_threshold} are: {negative_max_dates}")

    # print(f"The maximum values of the Nino 3.4 index that are greater than {strength_threshold} or less than {-strength_threshold} are: {max_values}")
    # print(f"The dates for the maximum values of the Nino 3.4 index that are greater than {strength_threshold} or less than {-strength_threshold} are: {max_dates}")
    ## FIND NEUTRAL PHASES: ---------------------------------------------------------------------------------

    filtered_neutral_values = []
    filtered_neutral_dates = []
    neutral_threshold = 0.3

    # Find blocks of consecutive dates where the enso index is -0.3 to 0.3, without using the threshold mask
    current_block_dates_neutral = []
    current_block_values_neutral = []
    for i in range(n):
        if nino34index[i] > -neutral_threshold and nino34index[i] < neutral_threshold:
            current_block_dates_neutral.append(nino34index.time.values[i])
            current_block_values_neutral.append(nino34index[i].values)
        else:
            if current_block_dates_neutral:
                filtered_neutral_dates.append(current_block_dates_neutral)
                filtered_neutral_values.append(current_block_values_neutral)
                current_block_dates_neutral = []
                current_block_values_neutral = []
    if current_block_dates_neutral:
        filtered_neutral_dates.append(current_block_dates_neutral)
        filtered_neutral_values.append(current_block_values_neutral)

        # Convert cftime.DatetimeNoLeap and numpy.datetime64 objects to datetime objects
    def convert_to_datetime(date):
        if isinstance(date, cftime.DatetimeNoLeap):
            return datetime(date.year, date.month, date.day)
        elif isinstance(date, np.datetime64):
            return pd.to_datetime(date).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
        return date

    number_of_blocks = 5
    # Convert dates in filtered_neutral_dates to datetime objects
    filtered_neutral_dates = [[convert_to_datetime(date) for date in block] for block in filtered_neutral_dates]

    # print(f"The longest block of consecutive dates with Nino 3.4 index between -{neutral_threshold} and {neutral_threshold} is {max(map(len, filtered_neutral_dates))} dates long.")
    # print(f"The top 5 longest blocks of consecutive dates with Nino 3.4 index between -{neutral_threshold} and {neutral_threshold} are {sorted(map(len, filtered_neutral_dates), reverse=True)[:number_of_blocks]} dates long.")
    
    # Save these top five blocks of dates in a variable
    consecutive_neutral_dates = sorted(filtered_neutral_dates, key=len, reverse=True)[:number_of_blocks]
    
    neutral_dates_array = np.array(np.concatenate(consecutive_neutral_dates), dtype='datetime64[ns]')
    # save dictionary of dates for each phase
    ENSO_dates_dict = {
        "El Nino" : positive_max_dates,
        "La Nina" : negative_max_dates, 
        "Neutral" : neutral_dates_array
    }
    # print(f"ENSO dates dictionary: {ENSO_dates_dict}")


    # print(f"The dates for all the top 5 longest blocks of consecutive dates with Nino 3.4 index between -{neutral_threshold} and {neutral_threshold} are:")
    # for i, block in enumerate(sorted(filtered_neutral_dates, key=len, reverse=True)[:number_of_blocks]):
    #     print(f"Block {i + 1}: {block[0]} to {block[-1]}")

    # # print(f"ALL of the actual values of the Nino 3.4 index for the top 5 longest blocks of consecutive dates with Nino 3.4 index between -{neutral_threshold} and {neutral_threshold} are:")
    # for i, block in enumerate(sorted(filtered_neutral_values, key=len, reverse=True)[:number_of_blocks]):
    #     print(f"Block {i + 1}: {np.array(block)} \n")
        # print(f"Block {i + 1}: {block[0]} to {block[-1]}")

    # # Convert cftime.DatetimeNoLeap objects to datetime objects
    # def convert_to_datetime(date):
    #     if isinstance(date, cftime.DatetimeNoLeap):
    #         return datetime(date.year, date.month, date.day, date.hour, date.minute, date.second)
    #     return date

    consecutive_neutral_dates = [np.array([convert_to_datetime(date) for date in block]) for block in consecutive_neutral_dates]
    flat_consecutive_neutral_dates = np.concatenate(consecutive_neutral_dates)

    # plot these zero dates as scatter points over the total nino34 index time series
    x_times = nino34index.time
     
    # plt.figure(figsize=(12, 7), dpi=200)
    # plt.plot(x_times, nino34index, color = '#26828e', label = 'Nino 3.4 Index MASKED')
    # plt.scatter(consecutive_neutral_dates, np.zeros(len(consecutive_neutral_dates)), color='r', s = 12)
    # plt.savefig(plotfn + 'ENSO_total_index_w_zeros_' + str(ens) + '.png', format='png', bbox_inches ='tight', dpi = 200)

    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(x_times, nino34index, color='#26828e')
    plt.scatter(flat_consecutive_neutral_dates, np.zeros(len(flat_consecutive_neutral_dates)), color='r', s=12, label = 'Consecutive Months of -0.2 < Index < 0.2')
    plt.title(f"{str(ens)} : Nino 3.4 Index - Consecutive Neutral-Phase Months")
    plt.ylabel('Nino 3.4 Index')
    plt.legend()
    plt.savefig(plotfn + 'ENSO_total_index_w_zeros_' + str(ens) + '.png', format='png', bbox_inches='tight', dpi=200)
    # plt.show()
    

    


    # Scatter Plot ----------------------------------------------------
    x_times = nino34index.time

    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(x_times, nino34_threshold_masked, color = '#26828e', label = 'Nino 3.4 Index MASKED')
    plt.scatter(max_dates, max_values, color='r', s = 12)

    # Convert max_dates to datetime before annotating
    max_dates = [convert_to_datetime(date) for date in max_dates]
    # add labels of each dates for each point (without time values)
    for i, txt in enumerate(max_dates):
        plt.annotate(pd.to_datetime(txt).strftime('%Y-%m-%d'), (max_dates[i], max_values[i]), fontsize=10)
    plt.title(f"{str(ens)} : Nino 3.4 Index with Events Greater than {strength_threshold}")
    plt.ylabel('Nino 3.4 Index')
    plt.xlabel('Time')
    # plt.show()
    plt.savefig(plotfn + 'ENSO_index_' + str(ens) + '.png', format='png', bbox_inches ='tight', dpi = 200)

    # Map Plots for E3SM DATA -------------------------------------------------------
    # print(type(ens2))
    # print(ens2.time)

    # date_to_select1 = cftime.DatetimeNoLeap(1927, 2, 1)

    # # Plot TS for 1927-02-01 ENS2 : EL NINO
    # ens2_ELNINO = ens2.sel(time = date_to_select1, method = 'nearest')

    # fig, ax = plt.subplots(1, 1, figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=200)
    # ens2_ELNINO.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', add_colorbar=False)
    # ax.coastlines()
    # ax.tick_params(axis='both', labelsize=8)
    # ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    # plt.savefig(plotfn + 'ENSO_index_' + str(ens) + 'ELNINO_1927-02-01.png', format='png', bbox_inches ='tight', dpi = 200)

    # # Plot TS for 1943-12-01 ENS2 : LA NINA
    # date_to_select2 = cftime.DatetimeNoLeap(1943, 12, 1)
    # ens2_LANINA = ens2.sel(time = date_to_select2, method = 'nearest')

    # fig, ax = plt.subplots(1, 1, figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=200)
    # ens2_LANINA.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', add_colorbar=False)
    # ax.coastlines()
    # ax.tick_params(axis='both', labelsize=8)
    # ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    # plt.savefig(plotfn + 'ENSO_index_' + str(ens) + 'LANINA_1927-02-01.png', format='png', bbox_inches ='tight', dpi = 200)

    return ENSO_dates_dict