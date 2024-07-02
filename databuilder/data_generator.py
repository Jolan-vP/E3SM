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
import time
import utils.filemethods as filemethods
from databuilder.sampleclass import SampleDict

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
                #train_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc")
                train_ds = filemethods.get_netcdf_da(self.data_dir +  "/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc")

            if ens == "ens2":
                #validate_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0151.eam.h1.1850-2014.nc")
                validate_ds = filemethods.get_netcdf_da(self.data_dir + "/input_vars.v2.LR.historical_0151.eam.h1.1850-2014.nc")

            elif ens == "ens3":
                #test_ds = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc")
                test_ds = filemethods.get_netcdf_da(self.data_dir + "/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc")
        
        train_ds = train_ds.sel(time = slice("1850", "2014"))
        validate_ds = validate_ds.sel(time = slice("1850", "2014"))
        test_ds = test_ds.sel(time = slice("1850", "2014"))

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
                    print("Isolating variables from Dataset")

                    if var == "PRECT": ## CONVERTING PRECIP TO MM/DAY!
                        da_copy = da.copy()

                        inc = 45 # 45 degree partitions in longitude to split up the data
                    
                        for iloop in np.arange(0, da_copy.shape[2] // inc + 1):
                            start = inc * iloop
                            end = np.min([inc * (iloop + 1), da_copy.shape[2]])
                            if start == end:
                                break
                            
                            mm_day = da_copy[:,:,start:end] * 10e3 * 86400
                            da[:, :, start:end] = mm_day
                            # stacked = da[:, :, start:end].stack(z=("lat", "lon"))
                            # da_copy[:, :, start:end] = stacked.groupby("time.dayofyear").map(self.subtract_trend).unstack()

                        print(f"da post incremental unit conversion: {da.values[500:540]}")
                    else:
                        pass

                    if len(self.config["input_vars"]) > 1: # If there is more than one input variable to process here
                        da = da.expand_dims(dim={"channel": 1}, axis = -1)   # (2) Create a channel dimension in da
                else: 
                    da = xr.concat([da, ds[var]], dim = "channel")  # (3) Fill channel dim with var array
      
            da = da.rename('SAMPLES')
            da.attrs['long_name'] = None
            da.attrs['units'] = None
            da.attrs['cell_methods'] = None


        # For each input variable or data entity you would like to process: 
        for ikey, key in enumerate(f_dict):
            if key == "y":
                print("Processing target output")
                
                f_dict[key] = ds[self.config["target_var"]]
                
                if self.config["target_var"] == "PRECT": # CONVERTING PRECIP TO MM/DAY! Must do this twice, one for input PRECT, one for Target PRECT
                    da_copy = f_dict[key].copy()

                    inc = 45 # 45 degree partitions in longitude to split up the data
                
                    for iloop in np.arange(0, da_copy.shape[2] // inc + 1):
                        start = inc * iloop
                        end = np.min([inc * (iloop + 1), da_copy.shape[2]])
                        if start == end:
                            break
                        
                        mm_day = da_copy[:,:,start:end] * 10e3 * 86400
                        f_dict[key][:, :, start:end] = mm_day
                
                # EXTRACT TARGET LOCATION
                if len(self.config["target_region"]) == 2: # Specific city / lat lon location
                    print("Target region is a single grid point")
                    targetlat = self.config["target_region"][0]
                    targetlon = self.config["target_region"][1]
                    f_dict[key] = f_dict[key].sel(lat = targetlat, lon = targetlon, method = 'nearest')
                
                elif len(self.config["target_region"]) == 4: # Generalized region of interest (lat-lon box)
                    print("Target region is a box region. Calculating regional average")
                    min_lat, max_lat = self.config["target_region"][:2]
                    min_lon, max_lon = self.config["target_region"][2:]
    
                    if isinstance(f_dict[key], xr.DataArray):
                        mask_lon = (f_dict[key].lon >= min_lon) & (f_dict[key].lon <= max_lon)
                        mask_lat = (f_dict[key].lat >= min_lat) & (f_dict[key].lat <= max_lat)
                        # print(f"mask_lat = {mask_lat}")
                        # print(f"mask_lon = {mask_lon}")
                        data_masked = f_dict[key].where(mask_lon & mask_lat, drop=True)
                        f_dict[key] = data_masked.mean(['lat', 'lon'])

                    else:
                        raise NotImplementedError("data must be xarray")
                
                # print(f"Shape of f_dict[key] before seasonal cycle removal: {f_dict[key].shape}")
                # print(f_dict[key][500:540])

                # REMOVE SEASONAL CYCLE 
                print("removing seasonal cycle")
                f_dict[key] = self.trend_remove_seasonal_cycle(f_dict[key])
                
                # print(f"Shape of f_dict[key] after seasonal cycle removal: {f_dict[key].shape}")
                # print(f_dict[key][500:540])

                # ROLLING AVERAGE
                print("rolling average")
                f_dict[key] = self.rolling_ave(f_dict[key]) # first six values are now nans due to 7-day rolling mean

                # LAG ADJUSTMENT OF TARGET DATASET : Lagging by self.config["lagtime"] number of days allows the input and target samples to align
                #  such that each input is paired with a target that is X days in the future
                print("lag")
                if self.config["lagtime"] != 0: 
                    f_dict[key] = f_dict[key][ self.config["lagtime"]: ]
                 #TODO: Confirm addition of nans?? "Lead/Lag code for y - shift forward 10 days = input 10x nans at the beginning of the dataset"
                
                print("completed processing target")
            else: 
                if self.target_only == True:
                    pass
                else:
                    print("Processing inputs")
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

                        ## LAG ADJUSTMENT OF INPUT: 
                        f_dict[key] = f_dict[key][0 : -self.config["lagtime"], ...]

                    else:
                        # LOAD f_dict dictionary with unprocessed channels of 'da'
                        f_dict[key] = da 
                
                        ## EXTRACT REGION
                        f_dict[key] = self._extractregion(f_dict[key])

                        ## MASK LAND/OCEAN 
                        f_dict[key] = self._masklandocean(f_dict[key])

                        # REMOVE SEASONAL CYCLE
                        for ichannel in range(f_dict[key].shape[-1]):
                            f_dict[key][..., ichannel] = self.trend_remove_seasonal_cycle(f_dict[key][...,ichannel])
                        
                        # checkplot = f_dict[key].sel(time = '1905-01-01')
                        # checkplot[...,1].plot()

                        ## ROLLING AVERAGE 
                        f_dict[key] = self.rolling_ave(f_dict[key])

                        ## LAG ADJUSTMENT OF INPUT: 
                        f_dict[key] = f_dict[key][0 : -self.config["lagtime"], ...]
                    
                    # Confirmed smoothed, detrended, deseasonalized, lag-adjusted anomalies of PRECT and TS
            
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
            





















def multi_input_data_organizer(config, MJO=False, ENSO = False, other = False):
    """
        train {x: RMM1, RMM2, Nino34}, 
              {y: target}

        val   {x: RMM1, RMM2, Nino34},
              {y: target}

        test  {x: RMM1, RMM2, Nino34}, 
              {y: target}
    """

    # MJO Principle Components --------------------------------------------
    if MJO == True: 
        print("Opening MJO PCs")
        # MJOsavename = '/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/MJOarray.leadnans.1850-2014.pkl'
        MJOsavename = '/Users/C830793391/BIG_DATA/E3SM_Data/presaved/MJOarray.leadnans.1850-2014.pkl'
        with gzip.open(MJOsavename, "rb") as obj:
            MJOarray = pickle.load(obj)
        obj.close()
    else:
        pass

    # ENSO Indices / Temperature Time Series of Nino3.4 -------------------
    if ENSO == True: 
        print("Opening high-res Nino34 Data")
        ninox_array = np.zeros([60225, 3])
        for iens, ens in enumerate(config["databuilder"]["ensemble_codes"]):
            fpath = config["data_dir"] + "presaved/ENSO_ne30pg2_HighRes/nino.member" + str(ens) + ".daily.nc"
            print(fpath)
            ninox = filemethods.get_netcdf_da(fpath)
            ninox_array[30:,iens] = ninox.nino34
            # 104 front nans, 30 values missing (first month) from 60225 total samples due to backward rolling average and monthly time step configuration
            # By starting at index 31, the ninox array should begin on 0 days since 1850-01-01 rather than 31 days since 1850-01-01
    else:
        pass

    # OTHER INPUT:  -------------------------------
    if other == True: 
        print("Opening OTHER")

        print("Opening exp005 VC TS input data for TRAINING")
        input_savename = config['data_dir'] + 'presaved/exp005_d_train_VC_TS_1850-2014.pkl'
        with gzip.open(input_savename, "rb") as obj:
            exp005_d_train_PNW_TS = pickle.load(obj)

        print("Opening exp005 VC TS input data for VALIDATION")
        input_savename = config['data_dir'] + 'presaved/exp005_d_val_VC_TS_1850-2014.pkl'
        with gzip.open(input_savename, "rb") as obj:
            exp005_d_val_PNW_TS = pickle.load(obj)

        print("Opening exp005 VC TS input data for TESTING")
        input_savename = config['data_dir'] + 'presaved/exp005_d_test_VC_TS_1850-2014.pkl'
        with gzip.open(input_savename, "rb") as obj:
            exp005_d_test_PNW_TS = pickle.load(obj)

    else:
        pass

    # Target : Lagged Precip at Target Location : --------------------------
    
    # config = utils.get_config("exp002")
    # seed = config["seed_list"][0]

    # data = ClimateData(
    #     config["databuilder"], 
    #     expname = config["expname"],
    #     seed=seed,
    #     data_dir = config["data_dir"], 
    #     figure_dir=config["figure_dir"],
    #     target_only=True,
    #     fetch=False,
    #     verbose=False
    # )
    
    # d_train, d_val, d_test = data.fetch_data()

    # data_savename1 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp002_d_train_TARGET.pkl"
    # data_savename2 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp002_d_val_TARGET.pkl"
    # data_savename3 = "/pscratch/sd/p/plutzner/E3SM/bigdata/presaved/exp002_d_test_TARGET.pkl"

    # if not os.path.exists(data_savename1) or not os.path.exists(data_savename2) or not os.path.exists(data_savename3):

    #     #data_savename1 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp001_d_train.pkl"
    #     with gzip.open(data_savename1, "wb") as fp:
    #         pickle.dump(d_train, fp)

    #     #data_savename2 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp001_d_val.pkl"
    #     with gzip.open(data_savename2, "wb") as fp:
    #         pickle.dump(d_val, fp)

    #     #data_savename3 = "/Users/C830793391/BIG_DATA/E3SM_Data/presaved/exp001_d_test.pkl"
    #     with gzip.open(data_savename3, "wb") as fp:
    #         pickle.dump(d_test, fp)

    #     print(f"Exp002 Target files have been created and data has been pickled.")

   
    print("Opening exp006 Seattle-area PRECIP target data for TRAINING")
    target_savename = config['data_dir'] + 'presaved/exp006_d_train_SeattleRegional_PRECT_1850-2014.pkl'
    with gzip.open(target_savename, "rb") as obj:
        exp006_d_train_target = pickle.load(obj)

    print("Opening exp006 Seattle-area PRECIP target data for VALIDATION")
    target_savename = config['data_dir'] + 'presaved/exp006_d_val_SeattleRegional_PRECT_1850-2014.pkl'
    with gzip.open(target_savename, "rb") as obj:
        exp006_d_val_target = pickle.load(obj)

    print("Opening exp006 Seattle-area PRECIP target data for TESTING")
    target_savename = config['data_dir'] + 'presaved/exp006_d_test_SeattleRegional_PRECT_1850-2014.pkl'
    with gzip.open(target_savename, "rb") as obj:
        exp006_d_test_target = pickle.load(obj)
    
 
    # Create Input and Target Arrays ------------------------------------------------------------

    print("Combining Input and target data")

    # if MJO == True and ENSO == True: 
    inputda = np.zeros([60225 - config["databuilder"]["lagtime"], 3, 3])
    print(inputda.shape)

    target = np.zeros([60225 - config["databuilder"]["lagtime"], 3], dtype=float) 
    
    target_dict = {0: exp006_d_train_target, 1: exp006_d_val_target, 2:exp006_d_test_target}

    for key, value in target_dict.items():
        inputda[:,0,key] = ninox_array[ :-config["databuilder"]["lagtime"], key] #ENSO
        inputda[:30,0,key] = np.nan # Fill beginning 30 zeros with Nans
        inputda[:,1,key] = MJOarray[ :-config["databuilder"]["lagtime"], 2,key]  #RMM1
        inputda[:,2,key] = MJOarray[ :-config["databuilder"]["lagtime"], 3,key]  #RMM2
    
        target[:,key] = value["y"] # Target - REGIONAL TEMP : SEATTLE METRO AREA: TARGET HAS ALREADY BEEN LAGGED IN PRE-PROCESSING (CLIMATE DATA CLASS - config_001/analysis_001)

    # INPUT DICT - Save to Pickle
    s_dict_train = SampleDict()
    s_dict_train["x"] = inputda[:,:,0]
    s_dict_train["y"] = target[:,0]

    s_dict_val  = SampleDict()
    s_dict_val["x"] = inputda[:,:,1]
    s_dict_val["y"] = target[:,1]

    s_dict_test = SampleDict()
    s_dict_test["x"] = inputda[:,:,2]
    s_dict_test["y"] = target[:,2]

    return s_dict_train, s_dict_val, s_dict_test

