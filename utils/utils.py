"""
Utilities: Classes and Functions

Functions: --------------------
get_config(exp_name)



"""
import json
import pandas as pd
import numpy as np

def get_config(exp_name):

    basename = "exp"

    with open("configs/config_" + exp_name[len(basename) :] + ".json") as f:
        config = json.load(f)

    assert config["expname"] == basename + exp_name[len(basename) :], "Exp_Name must be equal to config[exp_name]"

    # add additional attributes for easier use later
    config["databuilder"]["fig_dpi"] = config["fig_dpi"]
    config["databuilder"]["data_dir"] = config["data_dir"]

    return config