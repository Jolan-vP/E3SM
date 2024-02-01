# do I have to specify SST and precip only here, or just select from all available? 
"""
Functions containing settings for each experiment run: 
---------------------------------------------------------------
exp01: basic data loading, processing, and computing anomalies

exp101: first run of machine learning model

exp102: second run of machine learning model with 'X' changes... 

"""

__date__ = "26 January 2024"


def get_settings(experiment_name):
    experiments = {

        # Initial compute anomalies
        "exp01": {
            "target_variable": "PRECT", 

            "input_variables": ("PRECT", "TS",),

            "training_ens" : "0101", 
            "validation_ens" : "0151"
        }
    }

    settings = experiments[experiment_name]
    settings["exp_name"] = experiment_name

    return settings