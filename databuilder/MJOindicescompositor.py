"""
Indices Compositor: Calculate composites of data according to MJO/ENSO Indices

Functions: -----------------------
compositeindices

"""
import math
import configs
import json
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker

def compositeindices(config, daprocessed, iens=None): 
    """
    Inputs: 
    - Realtime Mulitvariate MJO Indices (time series of RMM1, RMM2, ... RMMn)
    - Time Series Data (PRECT, TS)

    Outputs: 
    - Composite graphs of each phase (1-9) for each variable (PRECT, TS..)

    """
    # Open MJO RMM1 + RMM2 files: 
    # TODO: make flexible for different chunks
    expconfig = config["databuilder"]

    MJOfilename = '/MJO_historical_' + expconfig['ensemble_codes'][iens] + '_1850-2014.pkl'

    with open(config['data_dir'] + MJOfilename, 'rb') as MJO_file:
        frontnans = np.nan * np.ones([120, 7])
        # First 120 days of the dataset are nans - they were eliminated as part of the RMM Calculation Process (Wheeler & Henden 2004)
        # Therefore I have added 120 nans as placeholders at the beginning of the array to represent the lost values

        # shortdatarange = expconfig["data_range"]
        # beginningindex = (shortdatarange[0] - 1850) * 365
        # endingindex = (shortdatarange[1] - 1850) * 365

        # Load MJO Data Array (MJOda), append frontnans, shorten to desired timerange
        MJOda = np.load(MJO_file, allow_pickle=True)
        MJOda = np.asarray(MJOda)
        MJOda = np.append(frontnans, MJOda, axis = 0)
        #SHORTEN
        # MJOda = MJOda[ beginningindex : endingindex, :]
        print(MJOda.shape)
      
    # Create phase number output array: 
    phases = np.zeros(len(MJOda))
    phaseqty = 9
    
    # FIRST: Identify which phase of MJO each datapoint is in: 
    for ichannel in range(daprocessed.shape[-1]):

        fig, ax = plt.subplots(5, 2, figsize=(15, 18), subplot_kw={'projection': ccrs.PlateCarree(central_longitude= 180)})
        extent = [ 40, -80, -14.5, 14.5]
        fonty = 18

        for samplecoord in range(0, len(MJOda[:,2])):
            
            # calculate coordinate angle: 
            RMM1 = MJOda[samplecoord, 2]
            RMM2 = MJOda[samplecoord, 3]

            if math.isnan(RMM1) == False:
            
                dY = RMM2
                dX = RMM1

                angle_deg = np.rad2deg(np.arctan2(dY,dX))
                if angle_deg < 0: 
                    angle_deg = 360 - np.abs(angle_deg)

                amplitude = np.sqrt(RMM1**2 + RMM2**2)
                
                assert amplitude >= 0

                # (1) If the amplitude of the line of coord (RMM1, RMM2) < 0 - phase 0 (Non-phase)
                if amplitude <= 1: 
                    phases[samplecoord] = 0
                    
                # If the coordinate point of (RMM1, RMM2) angle with (0,0) is [0,45] = phase 5... etc. 
                elif angle_deg >= 0 and angle_deg < 45 :
                    phases[samplecoord] = 5 
                elif angle_deg >= 45 and angle_deg < 90 :
                    phases[samplecoord] = 6 
                elif angle_deg >= 90 and angle_deg < 135 :
                    phases[samplecoord] = 7 
                elif angle_deg >= 135 and angle_deg < 180 :
                    phases[samplecoord] = 8 
                elif angle_deg >= 180 and angle_deg < 225 :
                    phases[samplecoord] = 1 
                elif angle_deg >= 225 and angle_deg < 270 :
                    phases[samplecoord] = 2     
                elif angle_deg >= 270 and angle_deg < 315 :
                    phases[samplecoord] = 3 
                elif angle_deg >= 315 and angle_deg <= 360 :
                    phases[samplecoord] = 4 
                else: 
                    print(f"angle: {angle_deg}, amplitude: {amplitude}")
                    raise ValueError("Sample does not fit into a phase (?)")
        
        column1 = np.tile(np.arange(5), 2)
        column2 = np.repeat(np.arange(2), 5)

        # Combine column 1 and column 2 into a 2D array
        plotorder = np.column_stack((column1, column2))
        
        # Use indices to identify phases of the processed data
        for phase in range(0, phaseqty):
            collectedphaseindices = np.where(phases == phase)[0]
            averagedphase = daprocessed[collectedphaseindices].mean(axis = 0)
            
            plot_ax = plotorder[phase, 0],plotorder[phase, 1]

            # PLOTS ----------------------------------------------
            if ichannel == 0:
                img = averagedphase[..., ichannel].plot(ax=ax[plot_ax], cmap='BrBG', transform=ccrs.PlateCarree(), add_colorbar = False)
                ax[plot_ax].coastlines()
                #ax.set_extent([-100, 30, 0, 80], crs=ccrs.PlateCarree())

            elif ichannel == 1:
                img = averagedphase[..., ichannel].plot(ax=ax[plot_ax], cmap='coolwarm', transform=ccrs.PlateCarree(), add_colorbar = False)
                ax[plot_ax].coastlines()
                #ax.set_extent([-100, 30, 0, 80], crs=ccrs.PlateCarree())

            if phase == 0:
                ax[plot_ax].set_title(f'Neutral', size = fonty)
                
                # Define the xticks for longitude
                gl = ax[plot_ax].gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlines = False
                gl.ylines = False

            else:
                ax[plot_ax].set_title(f'Phase {phase}', size = fonty)

                # Define the xticks for longitude
                gl = ax[plot_ax].gridlines(draw_labels=True)
                gl.top_labels = False
                gl.right_labels = False
                gl.xlines = False
                gl.ylines = False
                
        plt.tight_layout()
        ax[4,1].set_axis_off()
        plt.suptitle(f"Ensemble " + str(iens+1)+ "\nInput Variable: " + str(expconfig["input_vars"][ichannel]+"\n"), fontsize = fonty)
        plt.tight_layout()
        
        cbar_ax = fig.add_axes([0.92, 0.28, 0.01, 0.4])
        cbar_ax.tick_params(labelsize=fonty)
        fig.colorbar(img, cax=cbar_ax)
        plt.tight_layout()

        # TODO: Fix Colorbar!!!!! 
        # cbar = plt.colorbar(img)

        plt.savefig('/pscratch/sd/p/plutzner/E3SM/visuals/' + str(expconfig["ensembles"][iens]) + '/' + 'Global_' + str(expconfig["ensembles"][iens]) + str(expconfig["input_vars"][ichannel]) + str(expconfig["data_range"][0]) + '-' + str(expconfig["data_range"][1]) + '.png', format='png', bbox_inches ='tight', dpi = config["fig_dpi"], transparent =True)
        plt.show() 

    return MJOda




def phasecompare():
    # compare last 30 years to first 30 years - difference plot of each phase globally for precip

    pass