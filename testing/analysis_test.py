import xarray as xr
import os
import sys
import matplotlib.pyplot as plt
import importlib as imp
import pandas as pd
import numpy as np
#import nc_time_axis
import cftime
from scipy import stats
from scipy import integrate
import scipy as scipy
import sklearn
import tensorflow as tf
from sklearn import datasets, model_selection
from glob import glob

# import expsettings from ..setup

import setup.fileops as fileops
import setup.processdata as processdata
import setup.expsettings as expsettings
import  setup


# import expsettings from setup
settings = expsettings.get_settings("exp101")
x_train, x_t = processdata.load_inputs(directory, settings)


def test_load_file():

    settings = expsettings.get_settings("exp101")
    directory = ''
    x_train, x_t = processdata.load_inputs(directory, settings)
    print(x_train)