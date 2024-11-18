"""
Functions: ------------------
iENSO(SST_fn)

"""

import matplotlib.pyplot as plt
import numpy as np  
from shash.shash_torch import Shash

def identify_nino_phases(nino34_index, threshold=0.4, window=6, lagtime = None, smoothing_length = None):
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


    days_in_month = [28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31]
    days_in_month_array = np.tile(days_in_month, 165) # repeat days in month pattern to match SST data
    index_array_daily = np.full([60225,3], np.nan)

    current_day = 0
    # Interpolate each column of 'ones' and 'zeros' from monthly to daily according to the 355-no leap calendar
    for row in range(phase_array.shape[0]):
            for col in range(phase_array.shape[1]):
                month_chunk = np.repeat(phase_array[row, col], days_in_month_array[row])
                index_array_daily[current_day : current_day + days_in_month_array[row], col] = month_chunk
            current_day += days_in_month_array[row]

    # Chop front and back according to neural network input size: 
    index_array_daily = index_array_daily[:-lagtime, :]
    index_array_daily = index_array_daily[smoothing_length:]

    # Multiply the index_array_daily by the row number to recover the index of each day
    
    non_zero_indices = np.full_like(index_array_daily, np.nan)
    for col in range(index_array_daily.shape[1]):
        for row in range(index_array_daily.shape[0]):
            index_array_daily[row, col] = index_array_daily[row, col] * (row)
        # Remove all zeros from each column so that only non-zero values remain
        non_zero_values = index_array_daily[:, col][index_array_daily[:, col] != 0]
        non_zero_indices[:len(non_zero_values), col] = non_zero_values

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

    print(f"shape of p_elnino: {p_elnino.shape}")
    print(f"x_values shape: {x_values.shape}")  

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
    plt.ylim([0, 0.75])
    # plt.axvline(valset[:len(output)], color='r', linestyle='dashed', linewidth=1)
    lege = plt.legend(loc = 'upper left', )
    for lh in lege.legendHandles: 
        lh.set_alpha(1)
    plt.savefig('/Users/C830793391/Documents/Research/E3SM/saved/figures/' + str(config["expname"]) + '/' + str(config["expname"]) + '_ENSO_phase_predictions_w_climatology.png', format='png', bbox_inches ='tight', dpi = 300)
   



    print(f"El Nino average CRPS across all samples: {np.round(crps_scores[elnino].mean(), 4)}")
    print(f"La Nina average CRPS across all samples: {np.round(crps_scores[lanina].mean(), 4)}")
    print(f"Neutral average CRPS across all samples: {np.round(crps_scores[neutral].mean(), 4)}")

    plt.savefig('/Users/C830793391/Documents/Research/E3SM/saved/figures/' + str(config["expname"]) + '/CRPS_vs_ENSO_Phases_' + str(config["expname"]) + '.png', format='png', bbox_inches ='tight', dpi = 300)

    return elnino, lanina, neutral, CRPS_elnino, CRPS_lanina, CRPS_neutral