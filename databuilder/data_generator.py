""" 
Data Building Modules

Functions: ---------------- 
    Extract Region
    Rolling Average
    Create Data
    Fetch Data
    Process Data
    Subtract Trend
    Trend Remove Seasonal Cycle
    Mask LandOcean

    multi_input_data_organizer

Classes: ------------------
    Climate Data()

"""

import gc
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy
import numpy as np
import xarray as xr
import pickle
import gzip
import utils
import math
import time
import utils.filemethods as filemethods
from databuilder.sampleclass import SampleDict
import cartopy.crs as ccrs  
import cartopy.feature as cfeature
from cartopy.crs import PlateCarree
from analysis.analysis_metrics import save_pickle
from utils.filemethods import open_data_file as open_data_file
from itertools import islice
import pandas as pd
import gc


# -----------------------------------------------------

class ClimateData:
    " Custom dataset for climate data and processing "

    def __init__(self, config, expname, seed, data_dir, figure_dir, target_only=False, fetch=True, verbose=False):
   
        self.config = config
        self.expname = expname
        self.seed = seed
        self.data_dir = data_dir
        self.figure_dir = figure_dir
        self.verbose = verbose
        self.target_only = target_only
    
        if fetch:
            self.fetch_data()

    def fetch_data(self, verbose=None):
        if verbose is not None: 
            self.verbose = verbose

        self.d_train = SampleDict()
        self.d_val = SampleDict()
        self.d_test = SampleDict()

        self._create_data() 

        # if self.verbose:
        #     self.d_train.summary()
        #     self.d_val.summary()
        #     self.d_test.summary()

        return self.d_train, self.d_val, self.d_test 

    def _create_data(self):  
        if "ERA5" in self.config["data_source"]: # ERA5 REANALYSIS DATA PROCESSING ---------------------------------------------------
            # input_ds = filemethods.get_netcdf_da(self.data_dir + "ERA5/ERA5_1x1_input_vars_1940-2023_regrid.nc")
            input_ds = filemethods.get_netcdf_da(self.data_dir + "ERA5/ERA5_1x1_input_vars_P_TS_Z500_1940-2023_daily_m.nc")

            print("Process whole ERA5 dataset in one shot")
            f_dict_whole = self._process_data(input_ds)
            # save f_dict_whole for use later: 
            # save_pickle(f_dict_whole, self.data_dir + "ERA5/ERA5_complete_processed_1940-2023.pkl")
            print("Splitting data into train, val, and test sets")

            f_dict_train = SampleDict()
            f_dict_val = SampleDict()
            f_dict_test = SampleDict()

            f_dict_train['x'] = f_dict_whole['x'].sel(time = slice(str(self.config["train_years"][0]), str(self.config["train_years"][1])))
            f_dict_train['y'] = f_dict_whole['y'].sel(time = slice(str(self.config["train_years"][0]), str(self.config["train_years"][1])))
            f_dict_val['x'] = f_dict_whole['x'].sel(time = slice(str(self.config["val_years"][0]), str(self.config["val_years"][1])))
            f_dict_val['y'] = f_dict_whole['y'].sel(time = slice(str(self.config["val_years"][0]), str(self.config["val_years"][1])))
            f_dict_test['x'] = f_dict_whole['x'].sel(time = slice(str(self.config["test_years"][0]), str(self.config["test_years"][1])))
            f_dict_test['y'] = f_dict_whole['y'].sel(time = slice(str(self.config["test_years"][0]), str(self.config["test_years"][1])))

            self.d_train.concat(f_dict_train) 
            self.d_val.concat(f_dict_val) 
            self.d_test.concat(f_dict_test) 

        elif self.config["data_source"] == "E3SM": # E3SM ENSEMBLES DATA PROCESSING ---------------------------------------------------
            for iens, ens in enumerate(self.config["ensembles"]):
                print("Opening .nc files")
                train_ds = filemethods.get_netcdf_da(self.data_dir +    "/input_vars.P_T_Z5.v2.LR.historical_0101.eam.h1.1850-2014_precip_mmday.nc")
                validate_ds = filemethods.get_netcdf_da(self.data_dir + "/input_vars.P_T_Z5.v2.LR.historical_0151.eam.h1.1850-2014_precip_mmday.nc")
                test_ds = filemethods.get_netcdf_da(self.data_dir +     "/input_vars.P_T_Z5.v2.LR.historical_0201.eam.h1.1850-2014_precip_mmday.nc")
    
            if self.config["data_source"] != self.config["inference_data"]: # For Experiments with OUT OF DISTRIBUTION Inference - Process Inference Data
                f_dict_train = SampleDict()
                f_dict_val = SampleDict()
                f_dict_test = SampleDict()

                train_ds = train_ds.sel(time = slice(str(self.config["train_years"][0]), str(self.config["train_years"][1])))
                validate_ds = validate_ds.sel(time = slice(str(self.config["val_years"][0]), str(self.config["val_years"][1])))
                test_ds = test_ds.sel(time = slice(str(self.config["test_years"][0]), str(self.config["test_years"][1])))

                print("Processing training")
                f_dict_train = self._process_data(train_ds)
                print("Processing Validation")
                f_dict_val = self._process_data(validate_ds)
                print("Processing Testing")
                f_dict_test = self._process_data(test_ds)

            else:
                print("Processing Training")
                f_dict_train = self._process_data(train_ds)
                f_dict_train['x'] = f_dict_train['x'].sel(time = slice(str(self.config["train_years"][0]), str(self.config["train_years"][1])))
                f_dict_train['y'] = f_dict_train['y'].sel(time = slice(str(self.config["train_years"][0]), str(self.config["train_years"][1])))
                print("Processing validation")
                f_dict_val = self._process_data(validate_ds)
                f_dict_val['x'] = f_dict_val['x'].sel(time = slice(str(self.config["val_years"][0]), str(self.config["val_years"][1])))
                f_dict_val['y'] = f_dict_val['y'].sel(time = slice(str(self.config["val_years"][0]), str(self.config["val_years"][1])))
                print("Processing testing")
                f_dict_test = self._process_data(test_ds)
                f_dict_test['x'] = f_dict_test['x'].sel(time = slice(str(self.config["test_years"][0]), str(self.config["test_years"][1])))
                f_dict_test['y'] = f_dict_test['y'].sel(time = slice(str(self.config["test_years"][0]), str(self.config["test_years"][1])))   
            

            # print(f"magnitude of processed precip: {f_dict_train['x'][300:310, 10, 30, 0]}")
            # print(f"magnitude of processed temp: {f_dict_train['x'][300:310, 10, 30, 1]}")
            # print(f"magnitude of processed z500: {f_dict_train['x'][300:310, 10, 30, 2]}")

            self.d_train.concat(f_dict_train) 
            self.d_val.concat(f_dict_val) 
            self.d_test.concat(f_dict_test) 
            # print(f"shape of f_dict_train input: {f_dict_train['x'].shape}")
            # print(f"shape of f_dict_train target: {f_dict_train['y'].shape}")

    def _process_data(self, ds):
        '''
        Motivation: create file data dictionary to contain samples for use in ML model

        Input: 
        - Xarray DataSet
            Input dataset contains all input variables in one file

        Output: 
        - Dictionary containing Xarray DataArrays
            Output f_dict contains 'da'. 
            'da' contains multiple dimensions of masked, de-trended, de-seasonalized anomalies for all input variables. 
            
            f_dict contains 'da' using preprocessing keys as pointers

        '''

        f_dict = SampleDict() 

        ## TRYING MEMORY EFFICIENT METHOD ******
        input_vars_container = [None] * len(self.config["input_vars"])

        # (1) Isolate the individual dataset values of ds : PRECT, TS, etc. for INPUTS:
        if self.config["input_vars"] == "None": 
            pass
        else:
            for ivar, var in enumerate(self.config["input_vars"]):
                if ivar == 0: # PRECIPITATION VARIABLE MUST ALWAYS BE FIRST VARIABLE IN DS TO BE LOADED!! FOR ALL DATASETS!!
                    # da = ds[var]
                    input_vars_container[ivar] = ds[var]
                    print(f"shape of da: {input_vars_container[ivar].shape}")
                    print("Isolating variables from Dataset")
                else:
                    input_vars_container[ivar] = ds[var]
                #     if len(self.config["input_vars"]) > 1: # If there is more than one input variable to process here
                #         da = da.expand_dims(dim={"channel": 1}, axis = -1)   # (2) Create a channel dimension in da
                # else: 
                #     da = xr.concat([da, ds[var]], dim = "channel")  # (3) Fill channel dim with var array
      
            # da = da.rename('SAMPLES')
            # da.attrs['long_name'] = 'long_name'
            # da.attrs['units'] = 'units'
            # da.attrs['cell_methods'] = 'cell_methods'

        # For each input variable or data entity you would like to process: 
        for ikey, key in enumerate(f_dict):
            if key == "y" and self.config["target_var"] != "None":
                print("Processing target output") # PROCESSING TARGET ------------------------------------------
                
                f_dict[key] = ds[self.config["target_var"]]
                

                # EXTRACT TARGET LOCATION
                if len(self.config["target_region"]) == 2: # Specific city / lat lon location
                    print("Target region is a single grid point")
                    targetlat = self.config["target_region"][0]
                    targetlon = self.config["target_region"][1]
                    f_dict[key] = f_dict[key].sel(lat = targetlat, lon = targetlon, method = 'nearest')
                
                elif len(self.config["target_region"]) == 4 : # Generalized region of interest (lat-lon box)
                    print("Target region is a box region. Calculating regional average")
                    min_lat, max_lat = self.config["target_region"][:2]
                    min_lon, max_lon = self.config["target_region"][2:]

                    # Convert longitudes from -180 to 180 range to 0 to 360 range
                    if min_lon < 0:
                        min_lon += 360
                    if max_lon < 0:
                        max_lon += 360
                    
                    # print(f"min_lon: {min_lon}, max_lon: {max_lon}")
                    # print(f"min_lat: {min_lat}, max_lat: {max_lat}")
        
                    if isinstance(f_dict[key], xr.DataArray):
                        mask_lon = (f_dict[key].lon >= min_lon) & (f_dict[key].lon <= max_lon)
                        mask_lat = (f_dict[key].lat >= min_lat) & (f_dict[key].lat <= max_lat)

                        data_masked = f_dict[key].where(mask_lon & mask_lat, drop=True)
                
                        if self.config["target_mask"] == "land":
                            mask = xr.open_dataset(self.data_dir + "/landfrac.bilin.nc")["LANDFRAC"][0, :, :]
                                          
                            if str(data_masked.lat.values[0]).split(".")[1] == "0":
                                # Interpolate LANDFRAC to data grid
                                mask_interp = mask.interp(
                                    lat=data_masked.lat,
                                    lon=data_masked.lon,
                                    method="nearest"
                                )
                                data_masked = data_masked.where(mask_interp > 0.3)
                            else:
                                data_masked = data_masked.where(mask > 0.5)

                            # print(f"shape of data_masked: {data_masked.shape}")
                            print("Masking land, Plotting for confirmation: \n")
                        else: 
                            pass

                        masked_for_plot = data_masked.isel(time = 10)

                        fig, ax = plt.subplots(1, 1, figsize=(8, 7), subplot_kw={'projection': ccrs.PlateCarree()})
                        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='black')
                        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')
                        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='black')


                        if self.config["target_var"] == "PRECT" or self.config["target_var"] == "tp":
                            cmap_color = 'viridis_r'
                            extent_region = [-135, -110, 36.5, 58.5]
                            units = "mm/day"
                        elif self.config["target_var"] == "Z500" or self.config["target_var"] == "z":
                            cmap_color = "PuOr_r"
                            extent_region = [-180, -120, 30.5, 60.5]
                            units = "m"

                        lon_2d, lat_2d = np.meshgrid(data_masked.lon, data_masked.lat)
                        im = ax.pcolormesh(lon_2d, lat_2d, masked_for_plot.values, 
                                transform=ccrs.PlateCarree(), 
                                cmap=cmap_color)
                        
                        ax.set_extent(extent_region, crs=ccrs.PlateCarree())
                        ax.set_xlim(extent_region[:2])
                        ax.set_xticks(np.arange(extent_region[0], extent_region[1]+1, 4), crs=ccrs.PlateCarree())
                        ax.tick_params(axis='x', rotation=45)
                        ax.set_yticks(np.arange(32, 61, 4), crs=ccrs.PlateCarree())
                        gl = ax.gridlines(draw_labels=False, linewidth=0.5, alpha=0.3, linestyle='--')

                        from mpl_toolkits.axes_grid1 import make_axes_locatable
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="3%", pad=0.1, axes_class=plt.Axes)
                        cbar = plt.colorbar(im, cax=cax)
                        cbar.set_label(f'{self.config["target_var"]} ({units})', fontsize=12)

                        plt.tight_layout()
                        plt.savefig(self.figure_dir + str(self.expname) + "/" + str(self.expname) + "_target_masked.png", dpi=300)
                        plt.show()

                        f_dict[key] = data_masked.mean(['lat', 'lon'])

                else:
                    raise NotImplementedError("data must be xarray")

                # REMOVE SEASONAL CYCLE 
                print("removing seasonal cycle")
                f_dict[key] = self.trend_remove_seasonal_cycle(f_dict[key])

                # ROLLING AVERAGE
                print("rolling average")
                f_dict[key] = self.rolling_ave(f_dict[key]) # first 13 values are now nans due to 14-day rolling mean    
                
                print("completed processing target")
                print(f"shape of target is: {f_dict[key].shape}")


            else: # PROCESSING INPUTS ---------------------------------------------------------------------------
                if self.target_only is True:
                    print("Target only is true, skipping input processing")
                    pass
                else: # PROCESSING NETWORK INPUT VARIABLES
                    print("Processing inputs")
                    if len(self.config["input_vars"]) == 1:
                        f_dict[key] = da
                    
                        ## EXTRACT REGION
                        f_dict[key] = self._extractinputregion(f_dict[key])

                        ## MASK LAND/OCEAN 
                        f_dict[key] = self._masklandocean(f_dict[key])
                    
                        ## REMOVE SEASONAL CYCLE
                        f_dict[key] = self.trend_remove_seasonal_cycle(f_dict[key])

                        ## ROLLING AVERAGE 
                        f_dict[key] = self.rolling_ave(f_dict[key])

                    else:

                        ## TRYING MEMORY EFFICIENT METHOD ******
                        processed_channels = [None] * len(self.config["input_vars"])

                        for ivar, var_da in enumerate(input_vars_container):

                            processed_var = var_da

                            ## EXTRACT REGION
                            processed_var = self._extractinputregion(processed_var)

                            ## MASK LAND/OCEAN
                            processed_var = self._masklandocean(processed_var)

                            ## REMOVE SEASONAL CYCLE
                            processed_var = self.trend_remove_seasonal_cycle(processed_var)

                            ## ROLLING AVERAGE
                            processed_var = self.rolling_ave(processed_var)
 

                            # store in preallocated list: 
                            processed_channels[ivar] = processed_var

                            # clear memory
                            del var_da, processed_var
                            gc.collect()
                            print(f"completed processing variable {ivar+1}, memory cleared")

                        
                        print("concatenating channels")

                        print("aligning with referenc coords")
                        reference_coords = processed_channels[0].coords
                        for i in range(1, len(processed_channels)):
                            processed_channels[i] = processed_channels[i].reindex_like(processed_channels[0])


                        for i, processed_var in enumerate(processed_channels):
                            if i == 0:
                                f_dict[key] = processed_var.expand_dims(dim={"channel": 1}, axis=-1)
                            else:
                                f_dict[key] = xr.concat([f_dict[key], processed_var], dim="channel")

                        del processed_channels
                        gc.collect()

                        # # LOAD f_dict dictionary with unprocessed channels of 'da'
                        # f_dict[key] = var_da

                        # ## EXTRACT REGION
                        # f_dict[key] = self._extractinputregion(f_dict[key])

                        # ## MASK LAND/OCEAN 
                        # f_dict[key] = self._masklandocean(f_dict[key])

                        # # REMOVE SEASONAL CYCLE
                        # for ichannel in range(f_dict[key].shape[-1]):
                        #     f_dict[key][..., ichannel] = self.trend_remove_seasonal_cycle(f_dict[key][...,ichannel])
                    
                        # # checkplot = f_dict[key].sel(time = '1905-01-01')
                        # # checkplot[...,1].plot()

                        # ## ROLLING AVERAGE 
                        # f_dict[key] = self.rolling_ave(f_dict[key])
                    
                    print(f"shape of input is : {f_dict[key].shape}")
                    # Confirmed smoothed, detrended, deseasonalized, anomalies of PRECT and TS
            
        return f_dict
    
    def _extractinputregion(self, da): 
        if self.config["input_region"] == "None": 
            
            # "input_region": [[-15.0, 15.0, 40.0, 300.0],
            #              [-15.0, 15.0, 40.0, 300.0]],
            
            min_lon, max_lon = [0, 360]
            min_lat, max_lat = [-90, 90]
            print("input region is none")
        else:
            min_lat, max_lat = self.config["input_region"][:2]
            min_lon, max_lon = self.config["input_region"][2:]

        if isinstance(da, xr.DataArray):
            mask_lon = (da.lon >= min_lon) & (da.lon <= max_lon)
            mask_lat = (da.lat >= min_lat) & (da.lat <= max_lat)
            data_masked = da.where(mask_lon & mask_lat, drop=True)
            return (
                data_masked #,
                #data_masked["lat"].to_numpy().astype(np.float32),
                #data_masked["lon"].to_numpy().astype(np.float32),
            )
        else:
            raise NotImplementedError("data must be xarray")
        
    
    def _masklandocean(self, da):
        if self.config["input_mask"][0] == "None":
            return da
        
        mask = xr.open_dataset(self.data_dir + "/landfrac.bilin.nc")["LANDFRAC"][0, :, :]

        if self.config["input_mask"][0] == "land":
            da_masked = da * xr.where(mask > 0.5, 1.0, 0.0)
        elif self.config["input_mask"][0] == "ocean":
            da_masked = da * xr.where(mask > 0.5, 0.0, 1.0)
        else: 
            raise NotImplementedError('oops NONE error - line 147 of _masklandocean')
        
        return da_masked

    def subtract_trend(self, x): 
        
        detrendOrder = 3

        curve = np.polynomial.polynomial.polyfit(np.arange(0, x.shape[0]), x, detrendOrder)
        trend = np.polynomial.polynomial.polyval(np.arange(0, x.shape[0]), curve) 
    
        try: 
            detrend = x - np.swapaxes(trend, 0, 1)
        except:
            detrend = x - trend
        return detrend 
    
    
    def trend_remove_seasonal_cycle(self, da):

        if len(np.array(da.shape)) == 1: 
            return da.groupby("time.dayofyear").map(self.subtract_trend).dropna("time")
        
        else: 
            da_copy = da.copy()

            inc = 45 # 45 degree partitions in longitude to split up the data
        
            for iloop in np.arange(0, da_copy.shape[2] // inc + 1):
                start = inc * iloop
                end = np.min([inc * (iloop + 1), da_copy.shape[2]])
                if start == end:
                    break

                stacked = da_copy[:, :, start:end].stack(z=("lat", "lon"))

                da_copy[:, :, start:end] = stacked.groupby("time.dayofyear").map(self.subtract_trend).unstack()

        return da_copy.dropna("time")

    def rolling_ave(self, da):
        if self.config["averaging_length"] == 0:
            return da
        else: 
            if len(da.shape) == 1: 
                return da.rolling(time = self.config["averaging_length"]).mean()
            else: 
                da_copy = da.copy()
                inc = 45
                for iloop in np.arange(0, da.shape[2] // inc + 1): 
                    start = inc * iloop
                    end = np.min([inc *(iloop + 1), da_copy.shape[2]])
                    if start == end: 
                        break

                    da_copy[:, :, start:end] = da_copy[:, :, start:end].rolling(time = self.config["averaging_length"]).mean()

                return da_copy
            





















def multi_input_data_organizer(config, fn1, fn2, fn3, MJO=False, ENSO = False, other = False):
    """
        train {x: RMM1, RMM2, Nino34}, 
              {y: target}

        val   {x: RMM1, RMM2, Nino34},
              {y: target}

        test  {x: RMM1, RMM2, Nino34}, 
              {y: target}
    """
    start_year = config["databuilder"]["input_years"][0]
    end_year = config["databuilder"]["input_years"][1]

    
    # OPEN PREPROCESSED TARGET INPUT  ------------------------------- 

    print("Opening PNW PRECIP target data for TRAINING")
    with gzip.open(fn1, "rb") as obj:
        d_train_target = pickle.load(obj)

    print("Opening PNW PRECIP target data for VALIDATION")
    with gzip.open(fn2, "rb") as obj:
        d_val_target = pickle.load(obj)

    print("Opening PNW PRECIP target data for TESTING")
    with gzip.open(fn3, "rb") as obj:
        d_test_target = pickle.load(obj)

    da_length = len(d_train_target['y']) + 1
    
    # print(f"time training target data from processed pkl : {d_train_target['y'].time}")

    # MJO Principle Components --------------------------------------------

    if MJO == True: 
        print("Opening MJO PCs")
        if config["data_source"] == "E3SM":
            MJOsavename = '/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/MJOarray.leadnans.1850-2014.pkl'
            with gzip.open(MJOsavename, "rb") as obj:
                MJOarray = pickle.load(obj)
            obj.close()

            if start_year == 1850:
                # Due to EOF processing (by Po-Lun) the first four months of MJO dataset are NANS
                nan_rows = MJOarray[:120]

                # Filter rows based on input years
                filtered_rows = MJOarray[120:][(MJOarray[120:, 4, 0] >= start_year) & (MJOarray[120:, 4, 0] <= end_year)]

                # Combine NaN rows and filtered rows
                filtered_MJOarray = np.vstack((nan_rows, filtered_rows))
            else:
                # Directly filter all rows by input years
                filtered_MJOarray = MJOarray[(MJOarray[:, 4, 0] >= start_year) & (MJOarray[:, 4, 0] <= end_year)]

            # Replace the original MJOarray with the filtered version
            MJOarray = filtered_MJOarray

            # Optional: Print the filtered array or its shape for verification
            print(f"Filtered MJOarray shape: {MJOarray.shape}")
        
        elif config["data_source"] == "ERA5":

            MJOsavename = '/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/rmm.74toRealtime.txt'
            ## RMM values up to "real time". 19740601-20131231: Both SST1 variability (ENSO) and 120-day mean have been removed in these RMM values; 20140101-: Only the 120-day has been removed.
            MJOdf = open_data_file(MJOsavename)
            # print(f"MJO array: {MJOarray}")
            MJOdf.columns = MJOdf.columns.str.strip().str.rstrip(',')
            MJOdf['date'] = pd.to_datetime(dict(year=MJOdf.year, month=MJOdf.month, day=MJOdf.day))
            MJOdf.set_index('date', inplace=True)

            RMM1 = MJOdf["RMM1"]
            RMM2 = MJOdf["RMM2"]

            train_start_year = config["databuilder"]["train_years"][0]
            train_end_year = config["databuilder"]["train_years"][1]
            val_start_year = config["databuilder"]["val_years"][0]
            val_end_year = config["databuilder"]["val_years"][1]
            test_start_year = config["databuilder"]["test_years"][0]
            test_end_year = config["databuilder"]["test_years"][1]

            # Add 5 months worth of nans to the beginning of the training set
            # Original data begins 1974-06-01, but we want to start at 1974-01-01
            nan_dates = pd.date_range(start='1974-01-01', end='1974-05-31', freq='D')
            nan_series = pd.Series(data=np.nan, index=nan_dates)

            RMM1_orig = RMM1.loc[f'{train_start_year}-06-01':f'{train_end_year}-12-31']
            RMM1_train = pd.concat([nan_series, RMM1_orig])
            RMM1_val = RMM1.loc[f'{val_start_year}-01-01':f'{val_end_year}-12-31']
            RMM1_test = RMM1.loc[f'{test_start_year}-01-01':f'{test_end_year}-04-29']

            RMM2_orig = RMM2.loc[f'{train_start_year}-06-01':f'{train_end_year}-12-31']
            RMM2_train = pd.concat([nan_series, RMM2_orig])
            RMM2_val = RMM2.loc[f'{val_start_year}-01-01':f'{val_end_year}-12-31']
            RMM2_test = RMM2.loc[f'{test_start_year}-01-01':f'{test_end_year}-04-29']

            RMM1_data_dict = {0: RMM1_train, 1: RMM1_val, 2: RMM1_test}
            RMM2_data_dict = {0: RMM2_train, 1: RMM2_val, 2: RMM2_test}
        else:
            pass
            

    # ENSO Indices / Temperature Time Series of Nino3.4 -------------------
    if ENSO == True: 
        if config["data_source"] == "E3SM":
            print("Opening high-res DAILY Linearly Interpolated Nino34 Data")
            ninox_array = np.zeros([da_length, 3])
            for iens, ens in enumerate(config["databuilder"]["ensemble_codes"]):
                fpath = config["perlmutter_data_dir"] + "ENSO_Data/E3SM/ENSO_ne30pg2_HighRes/nino.member" + str(ens) + "_daily_linterp_shifted.nc"
                ninox = filemethods.get_netcdf_da(fpath)
                ninox = ninox.sel(time = slice(str(start_year), str(end_year)))
                nino34 = ninox.nino34.values
                print(f"shape of nino34: {nino34.shape}")
                print(f"nino time coordinate: {ninox.time}")

                if start_year == 1850: 
                    # add 15 new days of nans to the beginning of the array such that the total array length is now 15 values longer:
                    nan_array = np.zeros(15)
                    ninox_array[:, iens] = np.concatenate((nan_array, nino34, nan_array), axis = 0)
                    print(f"shape of ninox_array after adding 15 frontnans: {ninox_array.shape}")
                else: 
                    ninox_array[:, iens] = nino34
            print(f"filtered ninox_array shape: {ninox_array.shape}")
                # 15 values missing (1850-01-01 to 1850-01-15) from 60225 total samples due to backward rolling average and monthly time step configuration
                # By starting at index 15, the ninox array should begin on 0 days since 1850-01-01
        
        elif config["data_source"] == "ERA5":
            print("Opening Observational Nino34 Data")
            fpath = '/pscratch/sd/p/plutzner/E3SM/bigdata/ENSO_Data/OBS/nino34.long.anom_daily_linterp_shifted.nc'
            ninox = open_data_file(fpath)
            nino34 = ninox.value
            ninox_array = np.zeros([len(RMM1_train) + 1, 3])

            nino34_train = nino34.sel(time = slice(str(train_start_year), str(train_end_year)))
            nino34_val = nino34.sel(time = slice(str(val_start_year), str(val_end_year)))
            nino34_test = nino34.sel(time = slice(str(test_start_year), str(test_end_year)))

            ninox_array[:len(nino34_train), 0] = nino34_train
            ninox_array[:len(nino34_val), 1] = nino34_val
            ninox_array[:len(nino34_test), 2] = nino34_test
        else:
            pass
  
    # Create Input and Target Arrays ------------------------------------------------------------
    
    # NO LAGGING OCCURS IN THIS CODE
    print("Combining Input and target data")

    inputda = np.nan * np.ones([da_length, 3, 3])

    target_dict = {0: d_train_target, 1: d_val_target, 2: d_test_target}

    # establish correct time coordinates for input data: 
    training_time = xr.date_range(
        start = f'{config["databuilder"]["train_years"][0]}-01-01', 
        end = f'{config["databuilder"]["train_years"][1]}-12-31',
        freq = "1D", 
        calendar = "standard"
    )
    validation_time = xr.date_range(
        start = f'{config["databuilder"]["val_years"][0]}-01-01', 
        end = f'{config["databuilder"]["val_years"][1]}-12-31',
        freq = "1D", 
        calendar = "standard"
    )
    testing_time = xr.date_range(
        start = f'{config["databuilder"]["test_years"][0]}-01-01', 
        end = f'{config["databuilder"]["test_years"][1]}-12-31',
        freq = "1D", 
        calendar = "standard"
    )

    time_dict = {0: training_time, 1: validation_time, 2: testing_time}
    
    for key, value in target_dict.items():
        inputda[:,  0, key] = ninox_array[:, key] #ENSO
        inputda[:len(RMM1_data_dict[key]) , 1, key] = RMM1_data_dict[key] #RMM1
        inputda[:len(RMM2_data_dict[key]) , 2, key] = RMM2_data_dict[key] #RMM2

    # INPUT DICT
    s_dict_train = SampleDict()
    s_dict_val  = SampleDict()
    s_dict_test = SampleDict()

    # Collect input and target data
    input_dicts = [s_dict_train, s_dict_val, s_dict_test]

    # Assign target time coordinate to input data in new xarray dataarray
    for idict, s_dict in enumerate(input_dicts):
        if config["data_source"] == "E3SM": 
            s_dict["x"] = xr.DataArray(
                inputda[:, :, idict], 
                dims=["time", "channel"],  # Specify the dimensions
                coords={
                    "time": d_train_target['y'].coords["time"],  # Use the 'time' from 'y'
                    "channel": ["ENSO", "RMM1", "RMM2"] 
                },
                attrs = {"description" : "Input dataset with time metadata from target precip netcdf"}
            )
            # Assign target data from preprocessed target data above
            s_dict["y"] = target_dict[idict]["y"]

        elif config["data_source"] == "ERA5":
            s_dict["x"] = xr.DataArray(
                inputda[:time_dict[idict].shape[0], :, idict], 
                dims=["time", "channel"],  # Specify the dimensions
                coords={
                    "time": time_dict[idict],  # Use the 'time' created from split obs dataset
                    "channel": ["ENSO", "RMM1", "RMM2"] 
                },
                attrs = {"description" : "Input dataset with time metadata from target precip netcdf"}
            )
            # Assign target data from preprocessed target data above
            s_dict["y"] = target_dict[idict]["y"]

    # Confirm correct metadata for input and time coordinates
    # print(f"s_dict_train input time coordinate: {s_dict_train['x'].time}")
    # print(f"s_dict_train target time coordinate: {s_dict_train['y'].time}")

    # # print indices corresponding to nan values in input data
    # print(f"nan indices in input data: {np.where(np.isnan(s_dict_train['x'].values))}")
    # print(f"nan indices in input data: {np.where(np.isnan(s_dict_val['x'].values))}")
    # print(f"nan indices in input data: {np.where(np.isnan(s_dict_test['x'].values))}")
    return s_dict_train, s_dict_val, s_dict_test




def uniform_dist(lowerbound, upperbound, n, expname, config):
    dist = np.random.uniform(lowerbound, upperbound, n)
    
    # Save distribution to file
    save_pickle(dist, config["perlmutter_output_dir"] + str(expname) + "uniform_dist.pkl")

    return dist

def adjust_data_split(input_paths, config):
    """
    Stack processed ens1 + ens2 + ens3
    Split data according to desired training, validation, and testing ratio

    """
    # Open ens1+ens2
    ens1 = open_data_file(input_paths[0])
    ens2 = open_data_file(input_paths[1])

    


# -----------------------------------------------------------------------------
 # MJOsavename = '/pscratch/sd/p/plutzner/E3SM/bigdata/MJO_Data/rmm.74toRealtime.txt'
            ## RMM values up to "real time". 19740601-20131231: Both SST1 variability (ENSO) and 120-day mean have been removed in these RMM values; 20140101-: Only the 120-day has been removed.
            # MJOdf.columns = MJOdf.columns.str.strip().str.rstrip(',')
            # MJOdf['date'] = pd.to_datetime(dict(year=MJOdf.year, month=MJOdf.month, day=MJOdf.day))
            # MJOdf.set_index('date', inplace=True)