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


# Function to compute and average attributions for a list of indices
def average_attributions(model, inputs, indices, device, output_column, config, method="integrated_gradients"):
    # Convert inputs to tensor
    inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
    
    # Preallocate storage for attributions
    num_samples = len(indices)
    sample_shape = inputs[0].shape
    attributions_array = np.zeros((num_samples, *sample_shape))
    
    for i, idx in enumerate(indices):
        input_tensor = inputs[idx].unsqueeze(0).to(device)
        attributions = compute_attributions(model, input_tensor, output_column, method)
        attributions_array[i] = attributions.squeeze(0).cpu().detach().numpy()
    
    avg_attributions = np.mean(attributions_array, axis=0)

    analysis_metrics.save_pickle(avg_attributions, str(config["perlmutter_output_dir"]) + str(config["expname"]) + '/average_attributions_'
     + str(method) + '.pkl')

    return avg_attributions


# Function to visualize the average attributions
def visualize_average_attributions(avg_attributions, input_map, config, keyword = None):
    
    # setup colormap for attributions
    colors = [(1, 1, 1), (1, 1, 1), (1, 1, 1), (173/255, 74/255, 1), (102/255, 10/255, 78/255)]
    n_bins = 100
    cmap_name = "xai_purple"
    new_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
    
    if avg_attributions is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), subplot_kw= {'projection': ccrs.PlateCarree(central_longitude=180)})

        ax = axes[0]
        ax.set_title(f"Input Map - " + str(keyword))
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        if 'Precip' in keyword:
            cmap = 'viridis_r'
        else:
            cmap = 'RdBu_r'
        im = ax.imshow(input_map, extent=[0, 360, -90, 90], cmap = cmap, transform=ccrs.PlateCarree(), origin='lower')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction = 0.026, pad = 0.04)
        cbar.set_label(str(keyword))
        ax.set_xticks(np.arange(0, 361, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        # Plot the average attributions
        ax = axes[1]
        ax.set_title("Average Attributions")
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        im = ax.imshow(avg_attributions, extent=[0, 360, -90, 90], transform=ccrs.PlateCarree(), cmap=new_cmap, origin='lower')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction = 0.026, pad = 0.04)
        cbar.set_label("Attribution Value")
        ax.set_xticks(np.arange(0, 361, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_average_attributions.png', format='png', bbox_inches ='tight', dpi = 300)

    else: 
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw= {'projection': ccrs.PlateCarree(central_longitude=180)})
        ax.set_title(f"Input Map - " + str(keyword))
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        if 'Precip' in keyword:
            cmap = 'viridis_r'
        else:
            cmap = 'RdBu_r'
        im = ax.imshow(input_map, extent=[0, 360, -90, 90], cmap = cmap, transform=ccrs.PlateCarree(), origin='lower')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction = 0.026, pad = 0.04)
        cbar.set_label(str(keyword))
        ax.set_xticks(np.arange(0, 361, 60), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())

        plt.savefig(str(config["perlmutter_figure_dir"]) + str(config["expname"]) + '/' + str(keyword) + '_map.png', format='png', bbox_inches ='tight', dpi = 300)