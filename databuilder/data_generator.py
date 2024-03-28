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

    def __init__(self, config, expname, seed, data_dir, figure_dir, fetch =True, verbose=False):
        
        self.config = config
        self.expname = expname
        self.seed = seed
        self.data_dir = data_dir
        self.figure_dir = figure_dir
        self.verbose = verbose
        self.process_keys = ("x", "y")

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

        #TODO: Why is there no return method here for d_train, d_val and d_test?

    def _create_data(self):  

        for iens, ens in enumerate(self.config["ensembles"]):
            if self.verbose:
                print(ens)
            if ens == "ens1":                   
                train_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0101.eam.h1.1850-1852.nc")
                #train_da = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc")
            if ens == "ens2":
                validate_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0151.eam.h1.1850-1852.nc")
                #validate_da = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0151.eam.h1.1850-2014.nc")
            elif ens == "ens3":
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
        # CREATE FILE DATA DICTIONARY --------
        f_dict = SampleDict() 
        
        # We're filling f_dict using information from da by passing da (train_da, validate_da, test_da) directly to f_dict which becomes f_dict_train, f_dict_val, f_dict_test

        # (1) isolate the individual dataset values of ds : PRECT, TS, etc. 
        for ivar, var in enumerate(self.config["input_vars"]):
            if ivar == 0:
                da = ds[var]
                da = da.expand_dims(dim={"channel": 1}, axis = -1)   # (2) create a channel dimension in da
            else: 
                da = xr.concat([da, ds[var]], dim = "channel")  # (3) fill channel dim with var arrays

        #Preliminary f_dict data input (just to have data)
        f_dict["x"] = da
        # in reality, f_dict["x"] should contain two dimensions of masked, de-trended, de-seasonalized anomalies for PRECT and TS (two streams of input samples)

        ## EXTRACT REGION
        f_dict = self._extractregion(f_dict, da)
        print(f"f_dict is: \n {f_dict}")

        ## MASK LAND/OCEAN
        #f_dict = self._masklandocean(f_dict, da) 

        ## REMOVE SEASONAL CYCLE
        for key in f_dict:
            if key in self.process_keys:
                f_dict[key] = self.trend_remove_seasonal_cycle(f_dict[key])

        ## ROLLING AVERAGE 
        f_dict = self.rolling_ave(f_dict)

        return f_dict
    
    def _extractregion(self, f_dict, da): 
        # add for loop for each variable here too? so that unique input regions can be assigned?
        if self.config["input_region"] is None: # TODO: concerned the mask isn't working - how to index into this dict more deeply so that it sees the lat lons?
            min_lon, max_lon = [0, 360]
            min_lat, max_lat = [-90, 90]
            print("input region is none")
        else:
            min_lat, max_lat = self.config["input_region"][0][:2]
            min_lon, max_lon = self.config["input_region"][0][2:]

        if isinstance(da, xr.DataArray):
            mask_lon = (da.lon >= min_lon) & (da.lon <= max_lon)
            mask_lat = (da.lat >= min_lat) & (da.lat <= max_lat)
            f_dict["x"] = da.where(mask_lon & mask_lat, drop=True)

        return f_dict
    
    def _masklandocean(self, f_dict, da):
        if self.config["input_mask"][0] == None:
            return da
        
        mask = xr.open_dataset(self.data_dir + "/landfrac.bilin.nc")["LANDFRAC"][0, :, :]

        if self.config["input_mask"] == "land":
            da_mask = da * xr.where(mask > 0.5, 1.0, 0.0)
        elif self.config["input_mask"] == "ocean":
            da_mask = da * xr.where(mask > 0.5, 0.0, 1.0)
        else: 
            raise NotImplementedError('oops NONE error - line 147 of _masklandocean')
            ##TODO: What to do about "NONE" given that it's the first of the config specs? Should I specify [0]? Why do we specify both NONE and "ocean" and how does that contribute to our preferred output? 
            # Also I put "None" in quotes in the config file - is this OK?
        
        f_dict["x"] = da_mask 
        return f_dict

    def subtract_trend(self, f_dict): #TODO: Polynomial only wants a 1 or 2D array.. we have time, lat, lon, and channel in f_dict["x"].. should I loop through lat and lon as well?
        detrendOrder = 3
        for i_inputvar in np.arange(0, np.size(self.config["input_vars"])):
            curve = np.polynomial.polynomial.polyfit(np.arange(0, f_dict["x"].shape[0]), f_dict["x"][:, :, :, i_inputvar], detrendOrder)
            trend = np.polynomial.polynomial.polyval(np.arange(0, f_dict["x"][:, :, :, i_inputvar], curve, tensor = True))

            #curve = np.polynomial.polynomial.polyfit(np.arange(0, f_dict["x"].shape[0]), f_dict["x"], detrendOrder)
            #trend = np.polynomial.polynomial.polyval(np.arange(0, f_dict["x"], curve, tensor = True))

            try: 
                detrend = f_dict["x"][:, :, :, i_inputvar] - np.swapaxes(trend, 0, 1)
            except:
                detrend = f_dict["x"][:, :, :, i_inputvar] - trend

        return detrend 
    
    def trend_remove_seasonal_cycle(self, da):
        if len(f_dict["x"].shape) == 1:
            return f_dict["x"].groupby("time.dayofyear").map(self.subtract_trend).dropna("time")
        
        else: 
            f_dict_copy = f_dict.copy()
            inc = 45 # What does this increment refer to? 
            for iloop in np.arange(0, f_dict_copy["x"].shape[2] // inc + 1):
                start = inc * iloop
                end = np.min([inc * (iloop + 1), f_dict_copy["x"].shape[2]])
                if start == end:
                    break
        
                stacked = f_dict["x"][:, :, start:end, :].stack(z=("lat", "lon", "channel")) 

                f_dict_copy[:, :, start:end, :] = stacked.groupby("time.dayofyear").map(self.subtract_trend(f_dict)).unstack()

        return f_dict_copy.dropna("time")


    # this func really has the same structure as trend_remove_seasonal_cycle.. ?
    def rolling_ave(self, f_dict):
        if self.config["averaging_length"] == 0:
            return f_dict
        else: 
            if len(f_dict["x"].shape) == 1: 
                return f_dict["x"].rolling(time = self.config["averaging_length"]).mean()
            else: 
                f_dict_copy = f_dict["x"].copy()
                inc = 45
                for iloop, in np.arange(0, f_dict["x"].shape[2] // inc + 1): 
                    start = inc * iloop
                    end = np.min([inc *(iloop + 1), f_dict_copy.shape[2]])
                    if start == end: 
                        break
                    f_dict_copy[:, :, start:end, :] = f_dict["x"][:, :, start:end, :].rolling(time = self.config["averaging_length"]).mean()

                return f_dict_copy
            