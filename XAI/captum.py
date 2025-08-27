"""
XAI methods carried out using the Captum library.
- compute_attributions: Computes the attributions of the input features.
- average_attributions: Averages the attributions of the input features over selected indices.
- visualize_attributions: Visualizes the attributions of the input features.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import captum.attr 
from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel, visualization
import analysis.analysis_metrics as analysis_metrics
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define a function for computing attributions
def compute_attributions(model, inputs, output_column, method="integrated_gradients"):
    if method == "integrated_gradients":
        ig = IntegratedGradients(model)
        attributions = ig.attribute(inputs, target=output_column)
    elif method == "deeplift":
        dl = DeepLift(model)
        attributions = dl.attribute(inputs, target=output_column)
    elif method == "saliency":
        saliency = Saliency(model)
        attributions = saliency.attribute(inputs, target=output_column)
    else:
        raise ValueError("Unsupported method")
    return attributions


# Function to compute and average attributions for a list of dates
def average_attributions(model, input_maps, timestamps, device, output_column, config, method= None, keyword = None):
    # Convert inputs to tensor
    input_maps_filtered = input_maps.sel(time = timestamps)
    input_maps_tensor = torch.tensor(input_maps_filtered.values, dtype=torch.float32).to(device)
    
    # Preallocate storage for attributions
    num_samples = len(timestamps)
    sample_shape = input_maps_tensor[0].shape
    attributions_array = np.zeros((num_samples, *sample_shape))
    
    print("Computing attributions for each time stamp")
    for i, date in enumerate(timestamps):
        input_maps_tensor_single = input_maps_tensor[i, ...].unsqueeze(0).to(device)
        attributions = compute_attributions(model, input_maps_tensor_single, output_column, method)
        attributions_array[i] = attributions.squeeze(0).cpu().detach().numpy()

    print("Taking mean of computed attributions")
    avg_attributions = np.mean(attributions_array, axis=0)

    analysis_metrics.save_pickle(avg_attributions, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/average_attributions_'
     + str(method) + "_" + str(keyword) + '.pkl')

    for channel in range(input_maps.shape[3]):
        print(f"channel {channel}")
        input_maps_single_var = input_maps_filtered.isel(channel= channel).mean(dim = 'time')
        avg_attributions_single_var = avg_attributions[..., channel]
        print(f"input_maps_single_var shape: {input_maps_single_var.shape}")
        print(f"avg_attributions shape: {avg_attributions.shape}")
        if channel == 0:
            new_keyword = keyword + "_PRECT"
        elif channel == 1:
            new_keyword = keyword + "_SKINTEMP"
        visualize_average_attributions(avg_attributions_single_var, input_maps_single_var, config, new_keyword)

    return avg_attributions


# Function to visualize the average attributions
def visualize_average_attributions(avg_attributions, input_map, config, keyword = None):
    
    # setup colormap for attributions
    colors = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (173/255, 74/255, 1), (102/255, 10/255, 78/255)] #(220/255, 182/255, 251/255), 
    n_bins = 100
    cmap_name = "xai_purple"
    new_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Normalize average attributions
    avg_attributions = avg_attributions / np.max(np.abs(avg_attributions))
    
    if avg_attributions is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), subplot_kw= {'projection': ccrs.PlateCarree(central_longitude=180)})

        ax = axes[0]
        ax.set_title(f"Input Map - " + str(keyword))
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        if "PRECT" in keyword:
            cmap = 'BrBG'
            cbar_label = "Precipitation Anomaly (mm/day)"
        else:
            cmap = 'RdBu_r'
            cbar_label = "Skin Temperature Anomaly (K)"

        # vmin/vmax
        max_val_input = np.nanmax(np.abs(input_map))
        im = ax.imshow(input_map, extent=[0, 360, -90, 90], cmap = cmap, transform=ccrs.PlateCarree(), origin='lower', vmin = -max_val_input, vmax = max_val_input)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction = 0.026, pad = 0.04)
        cbar.set_label(str(cbar_label))
        ax.set_xticks(np.arange(0, 361, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        # Plot the average attributions
        ax = axes[1]
        ax.set_title("Average Attributions")
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        max_val_attr = np.nanmax(np.abs(avg_attributions))
        im = ax.imshow(avg_attributions, extent=[0, 360, -90, 90], transform=ccrs.PlateCarree(), cmap=new_cmap, origin='lower', vmin = -max_val_attr, vmax = max_val_attr)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction = 0.026, pad = 0.04)
        cbar.set_label("Attribution Value")
        ax.set_xticks(np.arange(0, 361, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_average_attributions.png', format='png', bbox_inches ='tight', dpi = 250)

    else: 
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw= {'projection': ccrs.PlateCarree(central_longitude=180)})
        ax.set_title(f"Input Map - " + str(keyword))
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        if "PRECT" in keyword:
            cmap = 'BrBG'
        else:
            cmap = 'RdBu_r'
        im = ax.imshow(input_map, extent=[0, 360, -90, 90], cmap = cmap, transform=ccrs.PlateCarree(), origin='lower')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction = 0.026, pad = 0.04)
        cbar.set_label(str(keyword))
        ax.set_xticks(np.arange(0, 361, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + 'composite_attribution_map.png', format='png', bbox_inches ='tight', dpi = 250)