"""
Indices Compositor: Calculate composites of data according to MJO/ENSO Indices

Functions: -----------------------
compositeindices

"""
import configs
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt

def compositeindices(config, daprocessed): 
    """
    Inputs: 
    - Realtime Mulitvariate MJO Indices (time series of RMM1, RMM2, ... RMMn)
    - Time Series Data (PRECT, TS)

    Outputs: 
    - Composite graphs of each phase (1-9) for each variable (PRECT, TS..)

    """
    # Open MJO RMM1 + RMM2 files: 
    # TODO: make flexible for different chunks
    for iens, ens in enumerate(config['databuilder']['ensembles']):
        with open(config['data_dir'] + ens + '/MJO_historical_' + config['databuilder']['ensemble_codes'][iens] + '_1850-2014.pkl', 'rb') as MJO_file:
            frontnans = np.nan * np.ones([120, 7])
            # First 120 days of the dataset are nans - they were eliminated as part of the RMM Calculation Process (Wheeler & Henden 2004)
            # Therefore I have added 120 nans as placeholders at the beginning of the array to represent the lost values

            if ens == "ens1":
                MJOens1 = np.load(MJO_file, allow_pickle=True)
                MJOens1 = np.asarray(MJOens1)
                MJOens1 = np.append(frontnans, MJOens1, axis = 0)
            if ens == "ens2":
                MJOens2 = np.load(MJO_file, allow_pickle=True)
                MJOens2 = np.asarray(MJOens2)
                MJOens2 = np.append(frontnans, MJOens2, axis = 0)
            if ens == "ens3":
                MJOens3 = np.load(MJO_file, allow_pickle=True)
                MJOens3 = np.asarray(MJOens3)
                MJOens3 = np.append(frontnans, MJOens3, axis = 0)
           
    # Combine indices for easier looping: 
    MJOindices = np.array([MJOens1, MJOens2, MJOens3])

    # Desired phase number output array: 
    phases = np.zeros([len(MJOindices[1]), np.size(config["databuilder"]["ensembles"])])
    print(f"shape phases array: {phases.shape}")

    phaseindex = np.nan * np.ones([phases.shape[0], 9, np.size(config["databuilder"]["ensembles"])])
    print(f"shape phaseindex: {phaseindex.shape}")

    # FIRST: Identify which phase of MJO each datapoint is in: 
    for iens, ens in enumerate(config["databuilder"]["ensembles"]):
        for samplecoord in range(120, len(MJOindices[0,:,0])-120):
            
            # calculate coordinate angle: 
            RMM1 = MJOindices[iens, samplecoord, 2]
            RMM2 = MJOindices[iens, samplecoord, 3]

            dY = RMM2
            dX = RMM1

            angle_deg = np.rad2deg(np.arctan2(dY,dX))
            if angle_deg < 0: 
                angle_deg = 360 - np.abs(angle_deg)

            amplitude = np.sqrt(RMM1**2 + RMM2**2)
            assert amplitude >= 0

            # (1) If the amplitude of the line of coord (RMM1, RMM2) < 0 - phase 0 (Non-phase)
            if amplitude <= 1: 
                phases[samplecoord, iens] = 0
                
            # If the coordinate point of (RMM1, RMM2) angle with (0,0) is [0,45] = phase 5... etc. 
            elif angle_deg >= 0 and angle_deg < 45 :
                phases[samplecoord, iens] = 5 
            elif angle_deg >= 45 and angle_deg < 90 :
                phases[samplecoord, iens] = 6 
            elif angle_deg >= 90 and angle_deg < 135 :
                phases[samplecoord, iens] = 7 
            elif angle_deg >= 135 and angle_deg < 180 :
                phases[samplecoord, iens] = 8 
            elif angle_deg >= 180 and angle_deg < 225 :
                phases[samplecoord, iens] = 1 
            elif angle_deg >= 225 and angle_deg < 270 :
                phases[samplecoord, iens] = 2     
            elif angle_deg >= 270 and angle_deg < 315 :
                phases[samplecoord, iens] = 3 
            elif angle_deg >= 315 and angle_deg <= 360 :
                phases[samplecoord, iens] = 4 
            else: 
                print(f"angle: {angle_deg}, amplitude: {amplitude}")
                raise ValueError("Sample does not fit into a phase (?)")
            
        # Use indices to identify phases of the processed data
        for phase in range(0, 9):
            collectedphaseindices = np.where(phases[:,iens]==phase)[0]
            print(len(collectedphaseindices))

            phaseindex[0:len(collectedphaseindices), phase, iens] = collectedphaseindices

            # Select non-nan values for each phase: 
            _phasecontainer = phaseindex[:,phase, iens]
            non_nans = _phasecontainer[~np.isnan(_phasecontainer)]

            # Select days corresponding with the time indices for ALL VARS and take ~mean~
            print(daprocessed[non_nans].mean())
            averagedphase = daprocessed[non_nans].mean()

            for ichannel in range(daprocessed.shape[-1]):
                fig, ax = plt.subplot(9)
                ax[phase].plot(averagedphase)

        # Plot the averaged phase groups 0 - 9 in a subplot structure over the MJO region 


    return MJOens1, MJOens2, MJOens3, phaseindex











    # Each file contains RMM1 and RMM2 for three variables : (1) U250, (2) U850, (3) PRECT (??)
    # Or only for the EOFs probably, these are just indices for the day regardless of variable?