""" 
Data Building Modules

Functions: ---------------- 


Classes: ------------------
Climate Data()

"""

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy
import numpy as np
import xarray as xr

# import utils
import databuilder.filemethods as filemethods
#import visuals.plots as plots
import databuilder.filemethods as filemethods
from databuilder.sampleclass import SampleDict

# -----------------------------------------------------

class ClimateData:
    " Custom dataset for climate data and processing "

    def __init__(self, config, expname, process_keys, seed, data_dir, figure_dir, fetch =True, verbose=False, ):
        print("initzializing climatedata class")
        self.config = config
        self.expname = expname
        self.seed = seed
        self.data_dir = data_dir
        self.figure_dir = figure_dir
        self.verbose = verbose
        self.process_keys = process_keys
        print("finished inizializing climatedata class")
        if fetch:
            self.fetch_data()

    def fetch_data(self, verbose=None):
        if verbose is not None: 
            self.verbose = verbose

        print("begin fetch func")

        self.d_train = SampleDict()
        self.d_val = SampleDict()
        self.d_test = SampleDict()

        self._create_data() 

        # if self.verbose:
        #     self.d_train.summary()
        #     self.d_val.summary()
        #     self.d_test.summary()

        #TODO: Why is there no return method here for d_train, d_val and d_test?

    def _create_data(self):  
        print("starting create data")
        for iens, ens in enumerate(self.config["ensembles"]):
            if self.verbose:
                print(ens)
            if ens == "ens1":   
                print("ens1 for training")
                train_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0101.eam.h1.1850-1852.nc")
                #train_da = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc")
            if ens == "ens2":
                print("ens2 for val")
                validate_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0151.eam.h1.1850-1852.nc")
                #validate_da = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0151.eam.h1.1850-2014.nc")
            elif ens == "ens3":
                print("ens3 for testing")
                test_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0201.eam.h1.1850-1852.nc")
                #test_da = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc")

                
        # Get opened X and Y data
        # Process Data (compute anomalies)
        f_dict_train = self._process_data(train_ds)
        f_dict_val = self._process_data(validate_ds)
        f_dict_test = self._process_data(test_ds)

        self.d_train.concat(f_dict_train) 
        self.d_val.concat(f_dict_val) 
        self.d_test.concat(f_dict_test) 
  

        # add latitude and longitude
        # self.lat = train_da.lat.values
        # self.lon = train_da.lon.values

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
        
            * more detail here about structure of 'da' * 
        '''


        f_dict = SampleDict() 

        # (1) Isolate the individual dataset values of ds : PRECT, TS, etc. 
        for ivar, var in enumerate(self.config["input_vars"]):
            if ivar == 0:
                da = ds[var]
                da = da.expand_dims(dim={"channel": 1}, axis = -1)   # (2) Create a channel dimension in da
            else: 
                da = xr.concat([da, ds[var]], dim = "channel")  # (3) Fill channel dim with var arrays

        da = da.rename('SAMPLES')
        da.attrs['long_name'] = None
        da.attrs['units'] = None
        da.attrs['cell_methods'] = None

        # For each input variable or data entity you would like to process: 
        for ikey, key in enumerate(f_dict):
            if key in self.process_keys: 
                assert key in f_dict.keys()
                # LOAD f_dict dictionary with unprocessed channels of 'da'
                print(f"ikey: {ikey}, key: {key}")
                f_dict[key] = da.sel(channel = ikey)
    
                ## EXTRACT REGION
                f_dict[key] = self._extractregion(f_dict[key])
                print(f"f_dict[key].shape out of extractregion: {f_dict[key].shape}")

                ## MASK LAND/OCEAN #TODO: FIX Mask function
                # f_dict[key] = self._masklandocean(f_dict[key])

                ## REMOVE SEASONAL CYCLE
                f_dict[key] = self.trend_remove_seasonal_cycle(f_dict[key])

                ## ROLLING AVERAGE 
                f_dict[key] = self.rolling_ave(f_dict[key])

        return f_dict
    
    def _extractregion(self, da): 
        if self.config["input_region"] is None: 
            # TODO: concerned the mask isn't working - how to index into this dict more deeply so that it sees the lat lons?
            min_lon, max_lon = [0, 360]
            min_lat, max_lat = [-90, 90]
            print("input region is none")
        else:
            min_lat, max_lat = self.config["input_region"][0][:2]
            min_lon, max_lon = self.config["input_region"][0][2:]

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
        if self.config["input_mask"][0] == None:
            return da
        
        mask = xr.open_dataset(self.data_dir + "/landfrac.bilin.nc")["LANDFRAC"][0, :, :]

        if self.config["input_mask"] == "land":
            da_masked = da * xr.where(mask > 0.5, 1.0, 0.0)
        elif self.config["input_mask"] == "ocean":
            da_masked = da * xr.where(mask > 0.5, 0.0, 1.0)
        else: 
            raise NotImplementedError('oops NONE error - line 147 of _masklandocean')
            ##TODO: What to do about "NONE" given that it's the first of the config specs? Should I specify [0]? Why do we specify both NONE and "ocean" and how does that contribute to our preferred output? - make it so it makes sense to you!
            # Also I put "None" in quotes in the config file - is this OK?
        
        return da_masked

    def subtract_trend(self, x): 
        print(f" Subtract trend input is X = {x}")

        detrendOrder = 3

        print(f" poly x is: \n{len(np.arange(0, x.shape[1]))}\n") # 1350
        print(f" poly y is \n{x.shape}\n") # (3, 1350)

        # TODO: Error is being thrown in line 201 due to mismatched X and Y lengths
        curve = np.polynomial.polynomial.polyfit(np.arange(0, x.shape[0]), x, detrendOrder)
        trend = np.polynomial.polynomial.polyval(np.arange(0, x.shape[0]), curve) 

        # THERE ARE MORE THAN 500 OUTPUTS OF PLOTS: 
        # plt.figure()
        # plt.plot(np.arange(0,  x.shape[1]), x[1])
        # plt.plot(np.arange(0,  x.shape[1]), trend)

        # raise ValueError
    
        try: 
            detrend = x - np.swapaxes(trend, 0, 1)
        except:
            detrend = x - trend
        return detrend 
    
    def trend_remove_seasonal_cycle(self, da):
        print("we are here at trendremoveseasonalcycle")
        if len(np.array(da.shape)) == 1: 
            print("shape of data = 1")
            return da.groupby("time.dayofyear").map(self.subtract_trend).dropna("time")
        
        else: 
            print("shape of da is not equal to 1")
            da_copy = da.copy()

            inc = 45 # 45 degree partitions in longitude to split up the data
        
            for iloop in np.arange(0, da_copy.shape[2] // inc + 1):
                start = inc * iloop
                end = np.min([inc * (iloop + 1), da_copy.shape[2]])
                if start == end:
                    break

          
                # Shape of stacked is ( len(time), len(lat)*len(longitude slice) ) = (1095, 1350)
                # 1350 is the number of (lat,lon) locations that exist within each longitude partition
                # stacked is a dataArray. 
                # Stacked.z contains all 1350 lat,long coordinates
                # Stacked.time contains time 1095
                stacked = da[:, :, start:end].stack(z=("lat", "lon"))

                #print(f"stacked stuff: {stacked.groupby('time.dayofyear')}") 
                #print(f"type(stacked) : {type(stacked)}")
                
                print(f"stacked Z: \n{stacked.coords}\n")
               
                print(f"first step: \n{stacked.groupby('time.dayofyear')}\n")

                da_copy[:, :, start:end] = stacked.groupby("time.dayofyear").map(self.subtract_trend).unstack()

                print(f"first step: \n{da_copy[:, :, start:end]}\n")

                # da_copy[:, :, start:end] = da_copy[:, :, start:end].map(self.mean())
                # print(f"subtract trend: \n{da_copy[:, :, start:end]}\n")

                # da_copy[:, :, start:end] = da_copy[:, :, start:end].unstack()
                # print(f"unstack: \n{da_copy[:, :, start:end]}\n")

                #da_copy[:, :, start:end] = stacked.groupby("time.dayofyear").map(self.subtract_trend).unstack()

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
            