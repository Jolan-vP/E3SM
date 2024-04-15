"""
Indices Compositor: Calculate composites of data according to MJO/ENSO Indices

Functions: -----------------------
compositeindices

"""
import configs
import json
import pickle


def compositeindices(exp, timeseriesdata, phases): 
    """
    Inputs: 
    - Realtime Mulitvariate MJO Indices (time series of RMM1, RMM2, ... RMMn)
    - Time Series Data (PRECT, TS)
    - Phases: How many phases (8 + 1 for MJO; 8 phases, 1 non-phase)


    Outputs: 
    - Composite graphs of each phase (1-9) for each variable (PRECT, TS..)

    """
    config = open('config_'+str(exp)+'.json')

    for iens, ens in enumerate(config['databuilder']['ensembles']):
        with open(str(config['data_dir']) + str(ens) + '/MJO_EOF1_EOF2_historical_' + config['data_dir']['ensemle_codes'][iens] + '_1850-2014.pkl', 'rb') as MJO_file:
            if ens == "ens1":
                MJOens1 = pickle.load(MJO_file)
            if ens == "ens2":
                MJOens2 = pickle.load(MJO_file)
            if ens == "ens3":
                MJOens3 = pickle.load(MJO_file)
    
    return MJOens1, MJOens2, MJOens3