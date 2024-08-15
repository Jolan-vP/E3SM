"""
Functions: ------------------
iENSO(SST_fn)

"""

import matplotlib.pyplot as plt
import numpy as np  

def iENSO(SSTdata):
    """
    Given a 1D array of monthly SST values, calculate the Niño 3.4 (5N-5S, 170W-120W) index

    "The  Niño 3.4 anomalies may be thought of as representing the average equatorial SSTs across the Pacific from about the dateline to the South American coast.  The Niño 3.4 index typically uses a 5-month running mean, and El Niño or La  Niña events are defined when the  Niño 3.4 SSTs exceed +/- 0.4C for a period of six months or more." - https://climatedataguide.ucar.edu/climate-data/nino-sst-indices-nino-12-3-34-4-oni-and-tni

    Input: 
    SSTseries (1D array of monthly SST values)
    """
    
    plt.figure()
    plt.plot(SSTdata)

    # If the SST values maintain 0.4C or above for 6 months or more, append the index of each month of that time period in a list
    # Allow each row to be a sample. Column 0 = El Nino, Column 1 = La Nina, Column 2 = Neutral

    # The SST data is monthly data
    # We ultimately need daily indices however

    # Creat an array where colum 0 is El Nino, Column 1 is La Nina, Column 2 is Neutral
    # Each row is a sample

    index_array = np.full([np.size(SSTdata), 3], np.nan)
    print(f"Index array shape: {index_array.shape}")

    # Put 1 if the sample is one of the three phase options, 0 if not
    # If the sample is not one of the three phase options, put a nan

    count_elnino = 0
    count_lanina = 0

    for i, daily_sst in enumerate(SSTdata):
        # Identify when the SST values exceed 0.4C for greater than or equal to 6 months
        if daily_sst >= 0.4:
            count_elnino += 1 
            count_lanina = 0 # Reset counter
            if count_elnino >= 6:
                index_array[i - count_elnino + 1: i + 1, 0] = np.arange(i-count_elnino+1, i+1)
                index_array[i - count_elnino + 1: i + 1, 2] = np.nan  # Clear neutral column
            elif count_elnino < 6:
                index_array[i, 2] = i

        elif daily_sst <= -0.4:
            count_lanina += 1 
            count_elnino = 0  # Reset counter
            if count_lanina >= 6:
                index_array[i - count_lanina + 1 : i + 1, 1] = np.arange(i-count_lanina+1, i+1)
                index_array[i - count_lanina + 1 : i + 1, 2] = np.nan  # Clear neutral column
            elif count_lanina < 6:
                index_array[i, 2] = i

        elif -0.4 < daily_sst < 0.4:
            count_elnino = 0
            count_lanina = 0
            index_array[i, 2] = i

    # Replace all the non-nan values with 1 and the nan values with 0
    index_array_binary = np.where(~np.isnan(index_array), 1, 0)

    index_array_daily = np.full([60225,3], np.nan)
    # # Interpolate each column of 'ones' and 'zeros' from monthly to daily
    for col in range(index_array_binary.shape[1]):
        for sampleindex, binaryvalue in enumerate(index_array_daily[:, col]):
            #convert monthly to daily according to the true number of days in each month (assuming no leap years) starting Feb 2nd 1850
            days_in_month = [28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31]
            # repeat days in month 165 times to match the number of days in the SST data
            days_in_month_array = np.tile(days_in_month, 165)

            index_array_daily[sampleindex : sampleindex + days_in_month_array[sampleindex], col] = np.repeat(index_array_binary[sampleindex, col], days_in_month_array[sampleindex])

# ^^ this is too complicated
    # for every row in the new array that you want to make
    # repeat the old array's row by the number of days in the month
    # approximately just repeating every old row so that each old row shows up 30 tiems in the new array

        # # eliminate nans from all columns such that there are only indices remaining
        # filtered_columns = [index_array[:, col][~np.isnan(index_array[:, col])] for col in range(index_array.shape[1])]
        # max_length = max(len(col) for col in filtered_columns)
        # index_array_condensed = np.array([np.pad(col, (0, max_length - len(col)), constant_values=np.nan) for col in filtered_columns]).T
        # # for the non-nan values, ensure they are of type int
        # index_array_condensed = index_array_condensed.astype(int)

    return index_array_binary, index_array_daily