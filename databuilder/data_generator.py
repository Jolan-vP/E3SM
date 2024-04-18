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

    def __init__(self, config, expname, seed, data_dir, figure_dir, fetch =True, verbose=False, ):
   
        self.config = config
        self.expname = expname
        self.seed = seed
        self.data_dir = data_dir
        self.figure_dir = figure_dir
        self.verbose = verbose
    
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
            if self.verbose:
                print(ens)
            if ens == "ens1":   
                train_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0101.eam.h1.1850-1860.nc")
                #train_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc")
            if ens == "ens2":
                validate_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0151.eam.h1.1850-1860.nc")
                #validate_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0151.eam.h1.1850-2014.nc")
            elif ens == "ens3":
                test_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0201.eam.h1.1850-1860.nc")
                #test_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc")

                
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
            if len(self.config["input_vars"]) == 1:
                f_dict[key] = da
               
                ## EXTRACT REGION
                f_dict[key] = self._extractregion(f_dict[key])

                ## MASK LAND/OCEAN 
                f_dict[key] = self._masklandocean(f_dict[key])
            
                ## REMOVE SEASONAL CYCLE
                f_dict[key] = self.trend_remove_seasonal_cycle(f_dict[key])

                ## ROLLING AVERAGE 
                f_dict[key] = self.rolling_ave(f_dict[key])

                plt.figure()
                plt.plot(f_dict[key].sel(lat = 10, lon = 10, method = 'nearest'))

            else:
                # LOAD f_dict dictionary with unprocessed channels of 'da'
                f_dict[key] = da #.sel(channel = ikey)
               #TODO: Something is still wrong here
                # plt.figure()
                # plt.plot(f_dict[key].sel(lat = 30, lon = 10, method = 'nearest'), color = 'green')
                # plt.ylabel(f'Var: '+str(self.config["input_vars"][ikey]) + '\n raw input data (lat:30, lon:10)')
                # plt.xlabel("Time")

                ## EXTRACT REGION
                f_dict[key] = self._extractregion(f_dict[key])

                ## MASK LAND/OCEAN 
                f_dict[key] = self._masklandocean(f_dict[key])
            
                print(f"channel 1: \n{f_dict[key][...,0]}")
                print(f"channel 2: \n{f_dict[key][...,1]}")

                # REMOVE SEASONAL CYCLE
                for ichannel in range(f_dict[key].shape[-1]):
                    f_dict[key][..., ichannel] = self.trend_remove_seasonal_cycle(f_dict[key][...,ichannel])
                #f_dict[key] = self.trend_remove_seasonal_cycle(f_dict[key])


                ## ROLLING AVERAGE 
                f_dict[key] = self.rolling_ave(f_dict[key])
            
                # Confirmed smoothed, detrended, deseasonalized anomalies of PRECT and TS
                 

        for ichannel in range(f_dict[key].shape[-1]):
                plt.figure()
                plt.plot(f_dict["x"][...,ichannel].sel(lat = 30, lon = 10, method = 'nearest'))
                plt.ylabel(f'Var: '+str(self.config["input_vars"][ichannel]) + '\ndetrended deseasonalized anomalies (lat:30, lon:10)')
                plt.xlabel("Time")
        print(f"channel 1: \n{f_dict[key][...,0]}")
        print(f"channel 2: \n{f_dict[key][...,1]}")
        # print(f_dict["x"])
        return f_dict
    
    def _extractregion(self, da): 
        if self.config["input_region"] is None: 

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
        # print(x.shape)
        detrendOrder = 3

        curve = np.polynomial.polynomial.polyfit(np.arange(0, x.shape[0]), x[:,0], detrendOrder)
        trend = np.polynomial.polynomial.polyval(np.arange(0, x.shape[0]), curve) 
    
        try: 
            detrend = x[:,0] - np.swapaxes(trend, 0, 1)
        except:
            detrend = x[:,0] - trend
        return detrend 
    
    def trend_remove_seasonal_cycle(self, da):

        if len(np.array(da.shape)) == 1: 
            print("shape of data = 1")
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
            