"""
Indices Compositor: Calculate composites of data according to MJO/ENSO Indices

Functions: -----------------------
compositeindices

"""
import configs
import json
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

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

        MJOfilename = ens + '/MJO_historical_' + config['databuilder']['ensemble_codes'][iens] + '_1850-2014.pkl'

        with open(config['data_dir'] + MJOfilename, 'rb') as MJO_file:
            frontnans = np.nan * np.ones([120, 7])
            # First 120 days of the dataset are nans - they were eliminated as part of the RMM Calculation Process (Wheeler & Henden 2004)
            # Therefore I have added 120 nans as placeholders at the beginning of the array to represent the lost values

            print(MJOfilename)
            shortdatarange = config["databuilder"]["data_range"]
            #print(shortdatarange)
            beginningindex = (shortdatarange[0] - 1850) * 365
            endingindex = (shortdatarange[1] - 1850) * 365
            #print(beginningindex, endingindex)

            if ens == "ens1":
                MJOens1 = np.load(MJO_file, allow_pickle=True)
                #print(f"ens1: {MJOens1}")
                MJOens1 = np.asarray(MJOens1)
                MJOens1 = np.append(frontnans, MJOens1, axis = 0)
                #SHORTEN
                
                MJOens1 = MJOens1[beginningindex : endingindex, :]
    
            elif ens == "ens2":
                MJOens2 = np.load(MJO_file, allow_pickle=True)
                #print(f"mjoEns2 shape: {MJOens2.shape}")
                MJOens2 = np.asarray(MJOens2)
                MJOens2 = np.append(frontnans, MJOens2, axis = 0)
                #SHORTEN
                #print(MJOens2[endingindex, 4:7])
                MJOens2 = MJOens2[beginningindex : endingindex, :]
            elif ens == "ens3":
                MJOens3 = np.load(MJO_file, allow_pickle=True)
                #print(f"ens3: {MJOens3}")
                MJOens3 = np.asarray(MJOens3)
                MJOens3 = np.append(frontnans, MJOens3, axis = 0)
                #SHORTEN
                MJOens3 = MJOens3[beginningindex : endingindex, :]
           
    # Combine indices for easier looping: 
    MJOindices = np.array([MJOens1, MJOens2, MJOens3])

    # Desired phase number output array: 
    phases = np.zeros([len(MJOindices[1]), np.size(config["databuilder"]["ensembles"])])
    phaseqty = 9
    
    # FIRST: Identify which phase of MJO each datapoint is in: 
    for iens, ens in enumerate(config["databuilder"]["ensembles"]):
        for ichannel in range(daprocessed.shape[-1]):

            fig, ax = plt.subplots(9, 1, figsize=(10, 12), subplot_kw={'projection': ccrs.PlateCarree(central_longitude= 180)})
            extent = [ 40, -80, -14.5, 14.5]
            fonty = 18

            for samplecoord in range(0, len(MJOindices[iens,:,2])):
                
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

            
            correctorder = [0, 8, 1, 2, 3, 4, 5, 6, 7]
            # Use indices to identify phases of the processed data
            for phase in range(0, phaseqty):
                print(ens)
                collectedphaseindices = np.where(phases[:,iens]==phase)[0]
                averagedphase = daprocessed[collectedphaseindices].mean(axis = 0)
                # ! does averagedphase need to be collected into three buckets then plotted? Is it ok that is (hypothetically) being written over with each iteration of ensemble member? 
                
                # PLOTS ----------------------------------------------
                if ichannel == 0:
                    img = averagedphase[..., ichannel].plot(ax=ax[correctorder[phase]], cmap='BrBG', transform=ccrs.PlateCarree(), add_colorbar = False)
                    #ax[correctorder[phase]].set_extent(extent, crs=ccrs.PlateCarree(central_longitude= 180))
                    ax[correctorder[phase]].coastlines()
                elif ichannel == 1:
                    img = averagedphase[..., ichannel].plot(ax=ax[correctorder[phase]], cmap='coolwarm', transform=ccrs.PlateCarree(), add_colorbar = False)
                    #ax[correctorder[phase]].set_extent(extent, crs=ccrs.PlateCarree(central_longitude= 180))
                    ax[correctorder[phase]].coastlines()
                if phase == 0:
                    ax[correctorder[phase]].set_title(f'Neutral', x = -0.08,  y = 0.3, pad = 14, size = fonty)
                else:
                    ax[correctorder[phase]].set_title(f'Phase {phase}', x = -0.08, y = 0.3, pad = 14,  size = fonty)

            
            plt.suptitle(f"Ensemble " + str(iens+1)+ "\nInput Variable: " + str(config["databuilder"]["input_vars"][ichannel]+"\n"), fontsize = fonty)
            plt.tight_layout()
           
            cbar_ax = fig.add_axes([1.01, 0.28, 0.02, 0.4])
            cbar_ax.tick_params(labelsize=fonty)
            fig.colorbar(img, cax=cbar_ax)

            plt.savefig('/Users/C830793391/Documents/Research/E3SM/visuals/' + str(ens) + '/' + str(ens) + str(config["databuilder"]["input_vars"][ichannel])+ '1900-1950.png', format='png', bbox_inches ='tight', dpi = config["fig_dpi"], transparent =True)
            plt.show() 

    return MJOindices, MJOens1, MJOens2, MJOens3











# line 138: 
                # phaseindex[0:len(collectedphaseindices), phase, iens] = collectedphaseindices
                # # Select non-nan values for each phase: 
                # _phasecontainer = phaseindex[:,phase, iens]
                # non_nans = _phasecontainer[~np.isnan(_phasecontainer)]
                # non_nans_int = non_nans.astype(int)
                #print(non_nans_int)