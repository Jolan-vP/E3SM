"""
Indices Compositor: Calculate composites of data according to MJO/ENSO Indices

Functions: -----------------------
compositeindices

"""
import configs
import json
import pickle
import numpy as np


def compositeindices(config, timeseriesdata): 
    """
    Inputs: 
    - Realtime Mulitvariate MJO Indices (time series of RMM1, RMM2, ... RMMn)
    - Time Series Data (PRECT, TS)

    Outputs: 
    - Composite graphs of each phase (1-9) for each variable (PRECT, TS..)

    """
    # Open MJO RMM1 + RMM2 files: 
    for iens, ens in enumerate(config['databuilder']['ensembles']):
        with open(config['data_dir'] + ens + '/MJO_historical_' + config['databuilder']['ensemble_codes'][iens] + '_1850-2014.pkl', 'rb') as MJO_file:
            if ens == "ens1":
                MJOens1 = np.load(MJO_file, allow_pickle=True)
                MJOens1 = np.asarray(MJOens1)
            if ens == "ens2":
                MJOens2 = np.load(MJO_file, allow_pickle=True)
                MJOens2 = np.asarray(MJOens2)
            if ens == "ens3":
                MJOens3 = np.load(MJO_file, allow_pickle=True)
                MJOens3 = np.asarray(MJOens3)
            ## TODO: THERE are 120 missing days from these MJO files?? 
    # Combine indices for easier looping: 
    MJOindices = np.array([MJOens1, MJOens2, MJOens3])

    # Desired phase number output array: 
    phases = np.zeros([len(MJOindices[1]), np.size(config["databuilder"]["ensembles"])])
    print(phases.shape)
    # FIRST: Identify which phase of MJO each datapoint is in: 
    for iens, ens in enumerate(config["databuilder"]["ensembles"]):
        for samplecoord in np.arange(0, len(MJOindices[0, :,0])):

            # calculate coordinate angle: 
            RMM1 = MJOindices[iens, samplecoord, 2]
            RMM2 = MJOindices[iens, samplecoord, 3]

            dY = RMM2
            dX = RMM1

            angle_deg = np.abs(np.arctan(dY/dX) * 180 * np.pi)
            if angle_deg > 360: 
                angle_deg = angle_deg - 360 #TODO: Confirm this choice

            magnitude = np.sqrt(RMM1**2 + RMM2**2)

            # (1) If the magnitude of the line of coord (RMM1, RMM2) < 0 - phase 0 (Non-phase)
            if magnitude <= 1: 
                phases[samplecoord, iens] = 0
                # print(phases[samplecoord, iens])
            # If the coordinate point of (RMM1, RMM2) angle with (0,0) is [0,45] = phase 5... etc. 
            elif magnitude > 1 and angle_deg >= 0 and angle_deg < 45 :
                phases[samplecoord, iens] = 5 
            elif magnitude > 1 and angle_deg >= 45 and angle_deg < 90 :
                phases[samplecoord, iens] = 6 
            elif magnitude > 1 and angle_deg >= 90 and angle_deg < 135 :
                phases[samplecoord, iens] = 7 
            elif magnitude > 1 and angle_deg >= 135 and angle_deg < 180 :
                phases[samplecoord, iens] = 8 
            elif magnitude > 1 and angle_deg >= 180 and angle_deg < 225 :
                phases[samplecoord, iens] = 1 
            elif magnitude > 1 and angle_deg >= 225 and angle_deg < 270 :
                phases[samplecoord, iens] = 2     
            elif magnitude > 1 and angle_deg >= 270 and angle_deg < 315 :
                phases[samplecoord, iens] = 3 
            else:    #angle_deg >= 315 and angle_deg < 359 :
                phases[samplecoord, iens] = 4 
         
    # Use indices to identify phases of the processed data # TODO: There are 120 missing days!
    # Indices for phase 1
    # Indices for phase 2
    # Indices for phase 3
    # Indices for phase 4
    # Indices for phase 5
    # Indices for phase 6
    # Indices for phase 7
    # Indices for phase 8
    # Indices for phase 0

    # Select days corresponding with the time indices
    # Average over the input variables for each phase group

    # Plot the averaged phase groups 0 - 9 in a subplot structure over the MJO region 


    return MJOens1, MJOens2, MJOens3, phases











    # Each file contains RMM1 and RMM2 for three variables : (1) U250, (2) U850, (3) PRECT (??)
    # Or only for the EOFs probably, these are just indices for the day regardless of variable?