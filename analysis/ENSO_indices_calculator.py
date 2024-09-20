"""
Functions: ------------------
iENSO(SST_fn)

"""

import matplotlib.pyplot as plt
import numpy as np  

def identify_nino_phases(nino34_index, threshold=0.4, window=6, front_cutoff=0, back_cutoff=0):
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

    # Chop front and back according to nerual network input size: 
    index_array_daily = index_array_daily[front_cutoff : -back_cutoff, :]

    # Multiply the index_array_daily by the row number to recover the index of each day
    
    non_zero_indices = np.full_like(index_array_daily, np.nan)
    for col in range(index_array_daily.shape[1]):
        for row in range(index_array_daily.shape[0]):
            index_array_daily[row, col] = index_array_daily[row, col] * (row)
        # Remove all zeros from each column so that only non-zero values remain
        non_zero_values = index_array_daily[:, col][index_array_daily[:, col] != 0]
        non_zero_indices[:len(non_zero_values), col] = non_zero_values

    return non_zero_indices.astype(int)  #index_array_daily.astype(int),



def ENSO_CRPS(enso_indices_daily, crps_scores, config): 
    # Isolate non-zero indices for each ENSO phase
    
    # Calculate index of first non-zero value when counting from back to front
    maxnino = max(np.where(enso_indices_daily[:,0] != 0)[0])
    maxnina = max(np.where(enso_indices_daily[:,1] != 0)[0])

    elnino = enso_indices_daily[:maxnino, 0]
    lanina = enso_indices_daily[:maxnina, 1]
    non_neutral = np.concatenate((elnino, lanina))
    neutral_total = (len(crps_scores) - (len(elnino) + len(lanina)))
    neutral = np.setdiff1d(np.arange(0, 60225), non_neutral)[:neutral_total]

    CRPS_elnino = np.round(crps_scores[elnino].mean(), 4)
    CRPS_lanina = np.round(crps_scores[lanina].mean(), 4)
    CRPS_neutral = np.round(crps_scores[neutral].mean(), 4)

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

    print(f"El Nino average CRPS across all samples: {np.round(crps_scores[elnino].mean(), 4)}")
    print(f"La Nina average CRPS across all samples: {np.round(crps_scores[lanina].mean(), 4)}")
    print(f"Neutral average CRPS across all samples: {np.round(crps_scores[neutral].mean(), 4)}")

    plt.savefig('/Users/C830793391/Documents/Research/E3SM/visuals/' + str(config["expname"]) + '/CRPS_vs_ENSO_Phases_' + str(config["expname"]) + '_allsamples.png', format='png', bbox_inches ='tight', dpi = 300)

    return elnino, lanina, neutral, CRPS_elnino, CRPS_lanina, CRPS_neutral