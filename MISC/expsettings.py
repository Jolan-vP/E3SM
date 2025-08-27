"""
Functions containing settings for each experiment run: 
---------------------------------------------------------------
exp01: basic data loading, processing, and computing anomalies

exp101: first run of machine learning model

exp102: second run of machine learning model with 'X' changes... 

"""

__date__ = "26 January 2024"

# do I have to specify SST and precip only here, or just select from all available? 

def get_settings(experiment_name):
    experiments = {

        #  CRPS ENSEMBLE ------------------------------------------
        "exp101": {
            "presaved_exp": "exp001",
            "target_var": "PRECT",
            "target_region": [47.6, 360.0 - 122.33],
            "target_months": [4, 5, 6, 7, 8, 9],

            "input_variables": ("PRECT", "TS",),
            "input_region": [[-15., 15., 40., 300.],
                             [-15., 15., 40., 300.]],
            "input_mask": [None, "ocean"],
            "leadtime": 14,
            "averaging_length": 7,
            "training_ens": "0101",
            "validation_ens": "0151",
            "test_val_ratio": 3,
            "reduce_data": False,

            "learning_rate": 1.e-5,
            "lr_schedule": (10, -.3),
            "batch_size": 32,
            "patience": 5,
            "min_delta": 0.00001,
            "n_epochs": 10_000,

            "network_type": "cnn",
            'uncertainty_type': 'crps',
            'n_members': 500,
            "random_seed_list": [123, ],
            "kernel_size": 5,
            "kernels": [64, 64, 64],
            "kernel_act": ["relu", "relu", "relu"],
            "hiddens": [5, 5],
            "dropout_rate": [0.3, 0.0],
            "act_fun": ["tanh", "tanh"],
            "y_std_scale": 100.
        },
        "exp102": {
            "presaved_exp": None,
            "target_var": "PRECT",
            "target_region": [47.6, 360.0 - 122.33],
            "target_months": [4, 5, 6, 7, 8, 9],

            "input_variables": ("PRECT", "TS",),
            "input_region": [[-15., 15., 40., 300.],
                             [-15., 15., 40., 300.]],
            "input_mask": [None, "ocean"],
            "leadtime": 21,
            "averaging_length": 7,
            "training_ens": "0101",
            "validation_ens": "0151",
            "test_val_ratio": 3,
            "reduce_data": False,

            "learning_rate": 1.e-5,
            "lr_schedule": (10, -.3),
            "batch_size": 32,
            "patience": 5,
            "min_delta": 0.00001,
            "n_epochs": 10_000,

            "network_type": "cnn",
            'uncertainty_type': 'crps',
            'n_members': 500,
            "random_seed_list": [123, ],
            "kernel_size": 5,
            "kernels": [64, 64, 64],
            "kernel_act": ["relu", "relu", "relu"],
            "hiddens": [5, 5],
            "dropout_rate": [0.3, 0.0],
            "act_fun": ["tanh", "tanh"],
            "y_std_scale": 100.
        },
        "exp103": {
            "presaved_exp": None,
            "target_var": "PRECT",
            "target_region": [47.6, 360.0 - 122.33],
            "target_months": [4, 5, 6, 7, 8, 9],

            "input_variables": ("PRECT", "TS",),
            "input_region": [[-15., 15., 40., 300.],
                             [-15., 15., 40., 300.]],
            "input_mask": [None, "ocean"],
            "leadtime": 28,
            "averaging_length": 7,
            "training_ens": "0101",
            "validation_ens": "0151",
            "test_val_ratio": 3,
            "reduce_data": False,

            "learning_rate": 1.e-5,
            "lr_schedule": (10, -.3),
            "batch_size": 32,
            "patience": 5,
            "min_delta": 0.00001,
            "n_epochs": 10_000,

            "network_type": "cnn",
            'uncertainty_type': 'crps',
            'n_members': 500,
            "random_seed_list": [123, ],
            "kernel_size": 5,
            "kernels": [64, 64, 64],
            "kernel_act": ["relu", "relu", "relu"],
            "hiddens": [5, 5],
            "dropout_rate": [0.3, 0.0],
            "act_fun": ["tanh", "tanh"],
            "y_std_scale": 100.
        },

            #  SHASH4 ------------------------------------------------
        "exp000": {
            "presaved_exp": "exp001",
            "target_var": "PRECT",
            "target_region": [47.6, 360.0 - 122.33],
            "target_months": [4, 5, 6, 7, 8, 9],

            "input_variables": ("PRECT", "TS",),
            "input_region": [[-15., 15., 40., 300.],
                             [-15., 15., 40., 300.]],
            "input_mask": [None, "ocean"],
            "leadtime": 7,
            "averaging_length": 7,
            "training_ens": "0101",
            "validation_ens": "0151",
            "test_val_ratio": 3,
            "reduce_data": False,

            "learning_rate": 5.e-6,
            "lr_schedule": (10_000, 0.),
            "batch_size": 32,
            "patience": 5,
            "min_delta": 0.0001,
            "n_epochs": 10_000,

            "network_type": "cnn",
            'uncertainty_type': 'shash4',
            "random_seed_list": [123, ],
            "kernel_size": 5,
            "kernels": [32, 32, 32],
            "kernel_act": ["relu", "relu", "relu"],
            "hiddens": [16, 8],
            "dropout_rate": [0.3, 0.0],
            "act_fun": ["relu", "relu"],
            "y_std_scale": 1.
        },
        "exp001": {
            "presaved_exp": "exp001",
            "target_variables": "PRECT",
            "target_region": [47.6, 360.0 - 122.33],
            "target_months": [4, 5, 6, 7, 8, 9],

            "input_vars": ("PRECT", "TS",),
            "input_region": [[-15., 15., 40., 300.],
                             [-15., 15., 40., 300.]],
            "input_mask": [None, "ocean"],
            "leadtime": 14,
            "averaging_length": 7,
            "training_ens": "0101",
            "validation_ens": "0151",
            "test_val_ratio": 3,
            "reduce_data": False,

            "learning_rate": 5.e-6,
            "lr_schedule": (10_000, 0.),
            "batch_size": 32,
            "patience": 5,
            "min_delta": 0.0001,
            "n_epochs": 10_000,

            "network_type": "cnn",
            'uncertainty_type': 'shash4',
            "random_seed_list": [123, ],
            "kernel_size": 5,
            "kernels": [32, 32, 32],
            "kernel_act": ["relu", "relu", "relu"],
            "hiddens": [16, 8],
            "dropout_rate": [0.3, 0.0],
            "act_fun": ["relu", "relu"],
            "y_std_scale": 1.
        },
        "exp002": {
            "presaved_exp": None,
            "target_variables": "PRECT",
            "target_region": [47.6, 360.0 - 122.33],
            "target_months": [4, 5, 6, 7, 8, 9],

            "input_vars": ("PRECT", "TS",),
            "input_region": [[-15., 15., 40., 300.],
                             [-15., 15., 40., 300.]],
            "input_mask": [None, "ocean"],
            "leadtime": 21,
            "averaging_length": 7,
            "training_ens": "0101",
            "validation_ens": "0151",
            "test_val_ratio": 3,
            "reduce_data": False,

            "learning_rate": 5.e-6,
            "lr_schedule": (10_000, 0.),
            "batch_size": 32,
            "patience": 5,
            "min_delta": 0.0001,
            "n_epochs": 10_000,

            "network_type": "cnn",
            'uncertainty_type': 'shash4',
            "random_seed_list": [123, ],
            "kernel_size": 5,
            "kernels": [32, 32, 32],
            "kernel_act": ["relu", "relu", "relu"],
            "hiddens": [16, 8],
            "dropout_rate": [0.3, 0.0],
            "act_fun": ["relu", "relu"],
            "y_std_scale": 1.
        },
    }

    settings = experiments[experiment_name]
    settings["exp_name"] = experiment_name

    return settings
