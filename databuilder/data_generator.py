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
        for iens, ens in enumerate(self.config["ensembles"]):
            print("Opening .nc files")
            if self.verbose:
                print(ens)
            if ens == "ens1":   
                # train_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc")
                train_ds = filemethods.get_netcdf_da(self.data_dir +  "/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc")
                # train_ds = filemethods.get_netcdf_da(self.data_dir +  "/Z500.v2.LR.historical_0101.eam.h1.1850-2014.nc")

            if ens == "ens2":
                # validate_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0151.eam.h1.1850-2014.nc")
                validate_ds = filemethods.get_netcdf_da(self.data_dir + "/input_vars.v2.LR.historical_0151.eam.h1.1850-2014.nc")
                # validate_ds = filemethods.get_netcdf_da(self.data_dir + "/Z500.v2.LR.historical_0151.eam.h1.1850-2014.nc")

            elif ens == "ens3":
                # test_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc")
                test_ds = filemethods.get_netcdf_da(self.data_dir + "/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc")
                # test_ds = filemethods.get_netcdf_da(self.data_dir + "Z500.v2.LR.historical_0201.eam.h1.1850-2014.nc")
        
        print(self.config["input_years"])

        train_ds = train_ds.sel(time = slice(str(self.config["input_years"][0]), str(self.config["input_years"][1])))
        validate_ds = validate_ds.sel(time = slice(str(self.config["input_years"][0]), str(self.config["input_years"][1])))
        test_ds = test_ds.sel(time = slice(str(self.config["input_years"][0]), str(self.config["input_years"][1])))

        # Get opened X and Y data
        # Process Data (compute anomalies)
        print("Processing training")
        f_dict_train = self._process_data(train_ds)
        print("Processing validation")
        f_dict_val = self._process_data(validate_ds)
        print("Processing testing")
        f_dict_test = self._process_data(test_ds)

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

        # (1) Isolate the individual dataset values of ds : PRECT, TS, etc. for INPUTS:
        if self.config["input_vars"] == "None": 
            pass
        else:
            for ivar, var in enumerate(self.config["input_vars"]):
                if ivar == 0:
                    da = ds[var]
                    print(f"shape of da: {da.shape}")
                    print("Isolating variables from Dataset")

                    if var == "PRECT" and int(math.floor(math.log10(da[10, 30, 120].values))) < - 5 : ## CONVERTING PRECIP TO MM/DAY!
                        da_copy = da.copy()

                        inc = 45 # 45 degree partitions in longitude to split up the data
                    
                        for iloop in np.arange(0, da_copy.shape[2] // inc + 1):
                            start = inc * iloop
                            end = np.min([inc * (iloop + 1), da_copy.shape[2]])
                            if start == end:
                                break
                            
                            mm_day = da_copy[:,:,start:end] * 10**3 * 86400
                            da[:, :, start:end] = mm_day

                        assert -150 < da[10, 30, 120].values < 150
                        # print(f"da post incremental unit conversion: {da[500:505].values}")
                    else:
                        pass

                    if len(self.config["input_vars"]) > 1: # If there is more than one input variable to process here
                        da = da.expand_dims(dim={"channel": 1}, axis = -1)   # (2) Create a channel dimension in da
                else: 
                    da = xr.concat([da, ds[var]], dim = "channel")  # (3) Fill channel dim with var array
      
            da = da.rename('SAMPLES')
            da.attrs['long_name'] = 'long_name'
            da.attrs['units'] = 'units'
            da.attrs['cell_methods'] = 'cell_methods'

        # For each input variable or data entity you would like to process: 
        for ikey, key in enumerate(f_dict):
            if key == "y" and self.config["target_var"] != "None":
                print("Processing target output")
                
                f_dict[key] = ds[self.config["target_var"]]
                
                # print(f"magnitude of target pre-unit conversion: {f_dict[key][500:505].values}")
                
                if self.config["target_var"] == "PRECT" and int(math.floor(math.log10(f_dict[key][10, 30, 120].values))) < - 5: # CONVERTING PRECIP TO MM/DAY!
                    da_copy = f_dict[key].copy()
                    
                    inc = 45 # 45 degree partitions in longitude to split up the data
                
                    for iloop in np.arange(0, da_copy.shape[2] // inc + 1):
                        start = inc * iloop
                        end = np.min([inc * (iloop + 1), da_copy.shape[2]])
                        if start == end:
                            break
                        
                        mm_day = da_copy[:,:,start:end] * 10**3 * 86400
                        f_dict[key][:, :, start:end] = mm_day

                    assert -150 < f_dict[key][10, 30, 120].values < 150
                # print(f"magnitude of target post unit-conversion: {f_dict[key][500:505].values}") 
                
                # fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
                # ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
                # ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='black')
                # one_day = f_dict[key][10, ...]
                # one_day.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis_r')
                # ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
                # ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
                # plt.tight_layout()
                # plt.show()
                # plt.savefig(self.figure_dir + str(self.expname) + "/" + str(self.expname) + "_target_masked_prelandmask.png", dpi=200)


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
        
                    if isinstance(f_dict[key], xr.DataArray):
                        mask_lon = (f_dict[key].lon >= min_lon) & (f_dict[key].lon <= max_lon)
                        mask_lat = (f_dict[key].lat >= min_lat) & (f_dict[key].lat <= max_lat)

                        data_masked = f_dict[key].where(mask_lon & mask_lat, drop=True)
                
                        if self.config["target_mask"] == "land":
                            mask = xr.open_dataset(self.data_dir + "/landfrac.bilin.nc")["LANDFRAC"][0, :, :]
                            data_masked = data_masked.where(mask > 0.5)
                            print(f"shape of data_masked: {data_masked.shape}")
                            print("Masking land, Plotting for confirmation: \n")
                            # data_masked = data_masked.dropna(dim="time", how="all")
                            # print(f"shape of data_masked after dropping nans: {data_masked.shape}")
                        else: 
                            pass

                        print(data_masked[10, ...])
                        fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
                        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='black')
                        ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='black')
                        ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='black')
                        data_masked_oneday = data_masked[10, ...]
                        data_masked_oneday.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis_r')
                        ax.set_xticks(np.arange(-180, 181, 4), crs=ccrs.PlateCarree())
                        ax.set_yticks(np.arange(-90, 91, 4), crs=ccrs.PlateCarree())
                        ax.set_ylim([36.5, 58.5])
                        ax.set_xlim([-135, -110])
                        plt.tight_layout()
                        plt.show()
                        plt.savefig(self.figure_dir + str(self.expname) + "/" + str(self.expname) + "_target_masked.png", dpi=300)

                        f_dict[key] = data_masked.mean(['lat', 'lon'])

                else:
                    raise NotImplementedError("data must be xarray")

                # REMOVE SEASONAL CYCLE 
                print("removing seasonal cycle")
                f_dict[key] = self.trend_remove_seasonal_cycle(f_dict[key])
                
                # print(f"Shape of f_dict[key] after seasonal cycle removal: {f_dict[key].shape}")
                # print(f_dict[key][500:540])

                # ROLLING AVERAGE
                print("rolling average")
                f_dict[key] = self.rolling_ave(f_dict[key]) # first 13 values are now nans due to 14-day rolling mean    
                
                print("completed processing target")
                print(f"shape of target is: {f_dict[key].shape}")
            else: 
                if self.target_only == True:
                    pass
                else:
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
                        # LOAD f_dict dictionary with unprocessed channels of 'da'
                        f_dict[key] = da 
                
                        ## EXTRACT REGION
                        f_dict[key] = self._extractinputregion(f_dict[key])

                        ## MASK LAND/OCEAN 
                        f_dict[key] = self._masklandocean(f_dict[key])

                        # REMOVE SEASONAL CYCLE
                        for ichannel in range(f_dict[key].shape[-1]):
                            f_dict[key][..., ichannel] = self.trend_remove_seasonal_cycle(f_dict[key][...,ichannel])
                        
                        # checkplot = f_dict[key].sel(time = '1905-01-01')
                        # checkplot[...,1].plot()

                        ## ROLLING AVERAGE 
                        f_dict[key] = self.rolling_ave(f_dict[key])
                    
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

                stacked = da[:, :, start:end].stack(z=("lat", "lon"))

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

                    da_copy[:, :, start:end] = da[:, :, start:end].rolling(time = self.config["averaging_length"]).mean()

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

    print("Opening Seattle-area PRECIP target data for TRAINING")
    with gzip.open(fn1, "rb") as obj:
        d_train_target = pickle.load(obj)

    print("Opening Seattle-area PRECIP target data for VALIDATION")
    with gzip.open(fn2, "rb") as obj:
        d_val_target = pickle.load(obj)

    print("Opening Seattle-area PRECIP target data for TESTING")
    with gzip.open(fn3, "rb") as obj:
        d_test_target = pickle.load(obj)

    da_length = len(d_train_target['y'])
    
    # print(f"time training target data from processed pkl : {d_train_target['y'].time}")

    # MJO Principle Components --------------------------------------------

    if MJO == True: 
        print("Opening MJO PCs")
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
    else:
        pass

    # ENSO Indices / Temperature Time Series of Nino3.4 -------------------
    if ENSO == True: 
        print("Opening high-res Nino34 Data")
        ninox_array = np.zeros([da_length, 3])
        for iens, ens in enumerate(config["databuilder"]["ensemble_codes"]):
            fpath = config["perlmutter_data_dir"] + "presaved/ENSO_ne30pg2_HighRes/nino.member" + str(ens) + ".daily.nc"
            ninox = filemethods.get_netcdf_da(fpath)
            ninox = ninox.sel(time = slice(str(start_year), str(end_year)))
            nino34 = ninox.nino34.values

            if start_year ==1850: 
                # add 30 new days of nans to the beginning of the array such that the total array length is now 30 values longer:
                nan_array = np.zeros(31)
                ninox_array[:, iens] = np.concatenate((nan_array, nino34), axis = 0)
            else: 
                ninox_array[:, iens] = nino34
        print(f"filtered ninox_array shape: {ninox_array.shape}")
            # 30 values missing (first month) from 60225 total samples due to backward rolling average and monthly time step configuration
            # By starting at index 31, the ninox array should begin on 0 days since 1850-01-01 rather than 31 days since 1850-01-01
    else:
        pass

    # OTHER INPUT:  -------------------------------
    # if other == True: 
    #     print("Opening OTHER")
   
    # Create Input and Target Arrays ------------------------------------------------------------
    
    # NO LAGGING OCCURS IN THIS CODE
    print("Combining Input and target data")

    inputda = np.zeros([da_length, 3, 3])

    target_dict = {0: d_train_target, 1: d_val_target, 2: d_test_target}
    
    for key, value in target_dict.items():
        inputda[:,  0, key] = ninox_array[:, key] #ENSO
        inputda[: , 1, key] = MJOarray[:, 2, key]  #RMM1
        inputda[: , 2, key] = MJOarray[:, 3, key]  #RMM2

    # INPUT DICT
    s_dict_train = SampleDict()
    s_dict_val  = SampleDict()
    s_dict_test = SampleDict()

    # Collect input and target data
    input_dicts = [s_dict_train, s_dict_val, s_dict_test]

    # Assign target time coordinate to input data in new xarray dataarray
    for idict, s_dict in enumerate(input_dicts):
        s_dict["x"] = xr.DataArray(
            inputda[:, :, idict], 
            dims=["time", "variables"],  # Specify the dimensions
            coords={
                "time": d_train_target['y'].coords["time"],  # Use the 'time' from 'y'
                "variables": ["ENSO", "RMM1", "RMM2"] 
            },
            attrs = {"description" : "Input dataset with time metadata from target precip netcdf"}
        )
        # Assign target data from preprocessed target data above
        s_dict["y"] = target_dict[idict]["y"]

    # Confirm correct metadata for input and time coordinates
    # print(f"s_dict_train input time coordinate: {s_dict_train['x'].time}")
    # print(f"s_dict_train target time coordinate: {s_dict_train['y'].time}")
    return s_dict_train, s_dict_val, s_dict_test




def uniform_dist(lowerbound, upperbound, n, expname, config):
    dist = np.random.uniform(lowerbound, upperbound, n)
    
    # Save distribution to file
    save_pickle(dist, config["perlmutter_output_dir"] + str(expname) + "uniform_dist.pkl")

    return dist