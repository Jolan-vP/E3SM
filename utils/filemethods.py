"""
Functions for working with generic files.

Functions: -----------------------
get_model_name(settings)
get_netcdf_da(filename)
save_pred_obs(pred_vector, filename)

"""
import xarray as xr

def get_netcdf_da(filename):
    ds = xr.open_dataset(filename)
    return ds