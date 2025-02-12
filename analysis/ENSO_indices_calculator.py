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

def identify_nino_phases(nino34_index, config, threshold=0.4, window=6, lagtime = None, smoothing_length = None):
    """
    Function to identify El Niño, La Niña, and Neutral phases based on Nino 3.4 SST index.
    The Niño 3.4 index typically uses a 5-month running mean, and El Niño or La  Niña events are defined when the  
    Niño 3.4 SSTs exceed +/- 0.4C for a period of six months or more." 
    https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni

    Parameters:
    - nino34_index (numpy array): Time series of Nino 3.4 SST index values.
    - threshold (float): Threshold for El Niño/La Niña classification.
    - window (int): Number of consecutive months for classification (default 6 months).
    
    Returns:
    - phase_array (numpy array): Array with 3 columns (El Niño, La Niña, Neutral) and rows corresponding to time steps.
    """
    n = len(nino34_index)
    # Initialize array to hold the phase classifications
    phase_array = np.zeros((n, 3), dtype=int)  # Columns: [El Niño, La Niña, Neutral]
    
    # Loop through the Nino3.4 index using a sliding window
    for i in range(n - window + 1):
        window_slice = nino34_index[i:i + window]
        
        if np.all(window_slice > threshold):
            # Mark El Niño (6-month period all > threshold)
            phase_array[i:i + window, 0] = 1
        elif np.all(window_slice < -threshold):
            # Mark La Niña (6-month period all < -threshold)
            phase_array[i:i + window, 1] = 1
        else:
            # Mark neutral for all other periods
            phase_array[i:i + window, 2] = 1

    # open original training dataset: 
    train_ds = filemethods.get_netcdf_da(str(config['perlmutter_data_dir']) + "/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc")
    # train_ds = filemethods.get_netcdf_da(str(config['perlmutter_data_dir']) + "ens1/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc")
    train_ds = train_ds.sel(time = slice("1850", "2014"))
    # train_ds = train_ds.sel(time = slice("str(config["databuilder"]["input_years"][0]"), str(config["databuilder"]["input_years"][1])))

    days_in_month = [28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31] 
    days_in_month_array = np.tile(days_in_month, 165) # repeat days in month pattern to match SST data
    index_array_daily = np.full([len(train_ds.time), 3], np.nan, dtype=float)

    current_day = 0
    # Interpolate each column of 'ones' and 'zeros' from monthly to daily according to the 355-no leap calendar
    for row in range(phase_array.shape[0]):
            for col in range(phase_array.shape[1]):
                month_chunk = np.repeat(phase_array[row, col], days_in_month_array[row])
                index_array_daily[current_day : current_day + days_in_month_array[row], col] = month_chunk
            current_day += days_in_month_array[row]

    # save index array daily as an xarray dataset with a time component (same as the original dataset)
    # Assign daily index array the same time coordinate from the original dataset
    index_array_dailyXR = xr.DataArray(
        index_array_daily, 
        dims=["time", "variables"],  # Specify the dimensions
        coords={
            "time": train_ds.coords["time"],  # Use the 'time' from 'y'
            "variables": ["El Nino Phase", "La Nina Phase", "Neutral Phase"] 
        },
        attrs = {"description" : "ENSO phases by each day carrying the time coord"}
    )

    # filter indices by target months, cut leads/lags, smoothing length, etc. 
    filtered_daily_indicesXR = universaldataloader(index_array_dailyXR, config, target_only = True) 

    # Multiply the index_array_daily by the row number to recover the index of each day
    non_zero_indices = np.full_like(filtered_daily_indicesXR, np.nan)

    for col in range(filtered_daily_indicesXR.shape[1]):
        for row in range(filtered_daily_indicesXR.shape[0]):
            if filtered_daily_indicesXR[row, col] != 0:
                filtered_daily_indicesXR[row, col] = filtered_daily_indicesXR[row, col] * row
        # Remove all zeros from each column so that only non-zero values remain
        non_zero_values = filtered_daily_indicesXR[:, col][filtered_daily_indicesXR[:, col] != 0]
        non_zero_indices[:len(non_zero_values), col] = non_zero_values

    # convert all remaining nans to zeros:
    non_zero_indices[np.isnan(non_zero_indices)] = 0

    return non_zero_indices.astype(int)  #index_array_daily.astype(int),



def ENSO_CRPS(enso_indices_daily, crps_scores, climatology, x_values, output, config): 
    # Isolate non-zero indices for each ENSO phase
    # Calculate index of first non-zero value when counting from back to front
    nino_indices = np.where(enso_indices_daily[:,0] != 0)[0]
    nina_indices = np.where(enso_indices_daily[:,1] != 0)[0]

    if nino_indices.size == 0:
        raise ValueError("No non-zero elements found in enso_indices_daily[:,0].")
    if nina_indices.size == 0:
        raise ValueError("No non-zero elements found in enso_indices_daily[:,1].")
    
    # Calculate index of first non-zero value when counting from back to front
    maxnino = max(np.where(enso_indices_daily[:,0] != 0)[0])
    maxnina = max(np.where(enso_indices_daily[:,1] != 0)[0])

    # print(f"maxnino: {maxnino}")
    # print(f"maxnina: {maxnina}")

    elnino = enso_indices_daily[:maxnino, 0]
    lanina = enso_indices_daily[:maxnina, 1]

    # print(f"len of elnino: {len(elnino)}")
    # print(f"len of lanina: {len(lanina)}")
    # print(f"len of crps_scores: {len(crps_scores)}")

    non_neutral = np.concatenate((elnino, lanina))
    neutral_total = (len(crps_scores) - (len(elnino) + len(lanina)))
    neutral = np.setdiff1d(np.arange(0, 60225), non_neutral)[:neutral_total]

    CRPS_elnino = round(crps_scores[elnino].mean(), 5)
    CRPS_lanina = round(crps_scores[lanina].mean(), 5)
    CRPS_neutral = round(crps_scores[neutral].mean(), 5)

    ENSO_dict = {
        "elnino" : [elnino], 
        "lanina" : [lanina], 
        "neutral": [neutral], 
        "CRPS":    [crps_scores]
    }

    save_pickle(ENSO_dict, str(config["perlmutter_output_dir"]) + str(config["expname"]) + "/" + str(config["expname"]) + "ENSO_indices_CRPS.pkl")

    # Plot CRPS by ENSO index
    # create a subplot with three columns and one row
    fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharey=True)
    ax[0].scatter(elnino, crps_scores[elnino], s=0.4, color ='#26828e', label = f'CRPS Average: {CRPS_elnino} ')
    ax[0].set_title('El Nino')
    ax[0].set_ylabel('CRPS')
    ax[1].scatter(lanina, crps_scores[lanina], s=0.4, color = '#26828e', label = f'CRPS Average: {CRPS_lanina}')
    ax[1].set_title('La Nina')
    ax[1].set_ylabel('CRPS')
    ax[2].scatter(neutral, crps_scores[neutral], s=0.4, color = '#26828e', label = f'CRPS Average: {CRPS_neutral}')
    ax[2].set_title('Neutral')
    ax[2].set_xlabel('Time (Samples in Chronological Order)')
    ax[2].set_ylabel('CRPS')
    ax[0].legend(loc = 'upper right')
    ax[1].legend(loc = 'upper right') 
    ax[2].legend(loc = 'upper right')
    plt.subplots_adjust(hspace=0.3)

    # Create plot with climatology histogram in the background and 100 random ENSO phase distributions on top
    # select 100 random samples each from elnino, lanina, and neutral
    np.random.seed(config["seed_list"][0])
    num_samples = 300
    rand_samps_elnino = np.random.choice(len(elnino), num_samples)
    rand_samps_lanina = np.random.choice(len(lanina), num_samples)
    rand_samps_neutral = np.random.choice(len(neutral), num_samples)

    dist_elnino = Shash(output[rand_samps_elnino])
    dist_lanina = Shash(output[rand_samps_lanina])
    dist_neutral = Shash(output[rand_samps_neutral])

    p_elnino = dist_elnino.prob(x_values).numpy()
    p_lanina = dist_lanina.prob(x_values).numpy()
    p_neutral = dist_neutral.prob(x_values).numpy()

    # print(f"shape of p_elnino: {p_elnino.shape}")
    # print(f"x_values shape: {x_values.shape}")  

    plt.figure(figsize=(12, 7), dpi=200)
    plt.hist(
        climatology, x_values, density=True, color="silver", alpha=0.75, label="climatology"
    )
     # Plot the first curve with a label
    plt.plot(x_values, p_elnino[:,0], alpha=0.1, color='#648FFF', linewidth=0.9, label=f'{num_samples} Random Predictions (El Nino)')
    plt.plot(x_values, p_elnino, alpha=0.1, color='#648FFF', linewidth=0.9, label=None)

    # Plot the first curve with a label
    plt.plot(x_values, p_lanina[:,0], alpha=0.1, color='#FE6100', linewidth=0.9, label=f'{num_samples} Random Predictions (La Nina)')
    plt.plot(x_values, p_lanina, alpha=0.1, color='#FE6100', linewidth=0.9, label=None)

    # # Plot the first curve with a label
    # plt.plot(x_values, p_neutral[:,0], alpha=0.2, color='#646363', linewidth=0.7, label=f'{num_samples} Random Predictions (Neutral ENSO)')
    # plt.plot(x_values, p_neutral, alpha=0.2, color='#646363', linewidth=0.7, label=None)

    # plt.plot(x_values, p_elnino, alpha = 0.2, color = '#648FFF', linewidth = 0.5 , label = f'{num_samples} Random Predictions during El Nino') 
    # plt.plot(x_values, p_lanina, alpha = 0.2, color = '#FFB000' , linewidth = 0.5, label = f'{num_samples} Random Predictions during La Nina') 
    # plt.plot(x_values, p_neutral, alpha = 0.2, color = '#b0b0b0', linewidth = 0.5, label = f'{num_samples} Random Predictions during Neutral ENSO') 
    plt.xlabel("Precipitation Anomalies (mm/day)")
    plt.ylabel("Probability Density")
    plt.title("Network Shash Prediction")
    plt.xlim([-10, 12])
    plt.ylim([0, 0.35])
    # plt.axvline(valset[:len(output)], color='r', linestyle='dashed', linewidth=1)
    lege = plt.legend(loc = 'upper left', )
    for lh in lege.legendHandles: 
        lh.set_alpha(1)
    plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(config["expname"]) + '_ENSO_phase_predictions_w_climatology.png', format='png', bbox_inches ='tight', dpi = 300)
   

    print(f"El Nino average CRPS across all samples: {np.round(crps_scores[elnino].mean(), 4)}")
    print(f"La Nina average CRPS across all samples: {np.round(crps_scores[lanina].mean(), 4)}")
    print(f"Neutral average CRPS across all samples: {np.round(crps_scores[neutral].mean(), 4)}")

    # plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/CRPS_vs_ENSO_Phases_' + str(config["expname"]) + '.png', format='png', bbox_inches ='tight', dpi = 300)

    return elnino, lanina, neutral, CRPS_elnino, CRPS_lanina, CRPS_neutral


def idealENSOphases(nino34index, ens2, ens = None, percentile = None, numberofeachphase = None, plotfn = None):
    """
    Function to find the dates corresponding to 'idealized' ENSO phases. Track dates by keeping xarray time coordinate
    """
    threshold = 0.4  # Threshold for El Niño/La Niña classification
    window = 6  # 6 months consecutively 


    # Mask out dates if there are fewer than 6 consecutive values in a row 
    # Loop through the Nino3.4 index using a sliding window
    n = len(nino34index)
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
                filtered_block = [(value, date) for value, date in current_block if value > 2 or value < -2]
                if filtered_block:
                    # If the filtered block is not empty, calculate the max value and record the date
                    max_value, max_date = max(filtered_block, key=lambda x: np.abs(x[0]))
                    max_values.append(max_value)
                    max_dates.append(max_date)
                # Reset the current block for the next contiguous block
                current_block = []

    # Handle the case where the last block extends to the end of the array
    if current_block:
        filtered_block = [(value, date) for value, date in current_block if value > 2 or value < -2]
        if filtered_block:
            max_value, max_date = max(filtered_block, key=lambda x: np.abs(x[0]))
            max_values.append(max_value)
            max_dates.append(max_date)

    # Find the 10th and 90th percentile values of those maximum values and return their corresponding dates
    max_values = np.array(max_values)
    max_dates = np.array(max_dates)

    filtered_zero_values = []
    filtered_zero_dates = []

    # Iterate through each index in the range of n
    for i in range(n):
        value = nino34_filled_xr[i]
        date = nino34index.time.values[i]
        # Filter values that are between -0.05 and 0.05
        if -0.05 < value < 0.05:
            filtered_zero_values.append(value)
            filtered_zero_dates.append(date)

    # Convert lists to numpy arrays
    filtered_zero_values = np.array(filtered_zero_values)
    filtered_zero_dates = np.array(filtered_zero_dates)

    print(f"Number of filtered dates: {len(filtered_zero_dates)}")

    # plot these zero dates as scatter points over the total nino34 index time series
    x_times = nino34index.time
     
    plt.figure(figsize=(12, 7), dpi=200)
    plt.plot(x_times, nino34index, color = '#26828e', label = 'Nino 3.4 Index MASKED')
    plt.scatter(filtered_zero_dates, filtered_zero_values, color='r', s = 7)
    plt.savefig(plotfn + 'ENSO_total_index_w_zeros_' + str(ens) + '.png', format='png', bbox_inches ='tight', dpi = 200)


    # General Index Time Series Plot -------------------------------------

    # plt.figure(figsize=(12, 7), dpi=200)
    # plt.plot(x_times, nino34index, color = '#26828e', label = 'Nino 3.4 Index MASKED')
    # plt.scatter(max_dates, max_values, color='r', s = 12)
    # # add labels of each dates for each point (without time values)
    # for i, txt in enumerate(max_dates):
    #     plt.annotate(txt.strftime('%Y-%m-%d'), (max_dates[i], max_values[i]), fontsize=8)
    # plt.title(f"{str(ens)} : Nino 3.4 Index with Events Greater than 2.0")
    # plt.show()
    # plt.savefig(plotfn + 'ENSO_total_index_time_series_' + str(ens) + '.png', format='png', bbox_inches ='tight', dpi = 200)

    # Scatter Plot ----------------------------------------------------
    x_times = nino34index.time

    plt.figure(figsize=(12, 7), dpi=200)
    plt.plot(x_times, nino34_threshold_masked, color = '#26828e', label = 'Nino 3.4 Index MASKED')
    plt.scatter(max_dates, max_values, color='r', s = 12)
    # add labels of each dates for each point (without time values)
    for i, txt in enumerate(max_dates):
        plt.annotate(txt.strftime('%Y-%m-%d'), (max_dates[i], max_values[i]), fontsize=8)
    plt.title(f"{str(ens)} : Nino 3.4 Index with Events Greater than 2.0")
    plt.ylabel('Nino 3.4 Index')
    plt.xlabel('Time')
    plt.show()
    plt.savefig(plotfn + 'ENSO_index_' + str(ens) + '.png', format='png', bbox_inches ='tight', dpi = 200)

    # Map Plots -------------------------------------------------------
    print(type(ens2))
    print(ens2.time)

    date_to_select1 = cftime.DatetimeNoLeap(1927, 2, 1)

    # Plot TS for 1927-02-01 ENS2 : EL NINO
    ens2_ELNINO = ens2.sel(time = date_to_select1, method = 'nearest')

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=200)
    ens2_ELNINO.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', add_colorbar=False)
    ax.coastlines()
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    plt.savefig(plotfn + 'ENSO_index_' + str(ens) + 'ELNINO_1927-02-01.png', format='png', bbox_inches ='tight', dpi = 200)

    # Plot TS for 1943-12-01 ENS2 : LA NINA
    date_to_select2 = cftime.DatetimeNoLeap(1943, 12, 1)
    ens2_LANINA = ens2.sel(time = date_to_select2, method = 'nearest')

    fig, ax = plt.subplots(1, 1, figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()}, dpi=200)
    ens2_LANINA.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='coolwarm', add_colorbar=False)
    ax.coastlines()
    ax.tick_params(axis='both', labelsize=8)
    ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    plt.savefig(plotfn + 'ENSO_index_' + str(ens) + 'LANINA_1927-02-01.png', format='png', bbox_inches ='tight', dpi = 200)