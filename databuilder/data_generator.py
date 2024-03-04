""" 
Data Building Modules

Functions: ---------------- 


Classes: ------------------
Climate Data()

"""

#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy
import numpy as np
import xarray as xr

import utils
import databuilder.filemethods as filemethods
#import visuals.plots as plots
#import databuilder.filemethods as filemethods
from databuilder.sampleclass import SampleDict

# -----------------------------------------------------

class ClimateData:
    " Custom dataset for climate data and processing "

    def __init__(self, config, expname, seed, data_dir, figure_dir, fetch =True, verbose=False):
        
        self.config = config
        self.expname = expname
        #self.ens = ens   ## ARE THESE NECESSARY??
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

        self._create_data() #TODO: self.ens??

        if self.verbose:
            self.d_train.summary()
            self.d_val.summary()
            self.d_test.summary()

    def _create_data(self): # TODO: do I need to include "ens" as input variable here? 
         # select correct config in configs file
        
        #filenames = filemethods.get_netcdf_da(self.data_dir)
    
        for iens, ens in enumerate(self.config["ensembles"]):
            if self.verbose:
                print(ens)

            if ens == "ens1":          
                print(ens)          
                train_da = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0101.eam.h1.1850-2014.nc")
            if ens == "ens2":
                print(ens)
                validate_da = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0151.eam.h1.1850-2014.nc")
            elif ens == "ens3":
                print(ens)
                test_da = filemethods.get_netcdf_da(self.data_dir + ens + "/input_vars.v2.LR.historical_0201.eam.h1.1850-2014.nc")
            # else:
            #     raise ValueError('choose from one of the available ensembles')
                
            # Get opened X and Y data
            # Process Data (compute anomalies)
        f_dict_train = self._process_data(train_da)
        f_dict_val = self._process_data(validate_da)
        f_dict_test = self._process_data(test_da)

        return f_dict_train, f_dict_val, f_dict_test
    

    def _process_data(self, da):
        # CREATE FILE DATA DICTIONARY
        f_dict = SampleDict()

        return f_dict
    

    # def _get_members(self):

    #     self.train_members = []
    #     self.val_members = []
    #     self.test_members = []

    #     for splitvec in self.config["n_train_val_test"]:
    #         n_train = splitvec[0]
    #         n_val = splitvec[1]
    #         n_test = splitvec[2]
    #         all_members = np.arange(0, n_train + n_val + n_test) ..........