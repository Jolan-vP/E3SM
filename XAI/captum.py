"""
XAI methods carried out using the Captum library.
- compute_attributions: Computes the attributions of the input features.
- average_attributions: Averages the attributions of the input features over selected indices.
- visualize_attributions: Visualizes the attributions of the input features.

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import captum.attr 
from captum.attr import IntegratedGradients, Saliency, DeepLift, NoiseTunnel, visualization

# Define a function for computing attributions
def compute_attributions(model, inputs, target, method="integrated_gradients"):
    if method == "integrated_gradients":
        ig = IntegratedGradients(model)
        attributions = ig.attribute(inputs, target=target)
    elif method == "saliency":
        saliency = Saliency(model)
        attributions = saliency.attribute(inputs, target=target)
    else:
        raise ValueError("Unsupported method")
    return attributions


# Function to compute and average attributions for a list of indices
def average_attributions(model, inputs, target, indices, device, method="integrated_gradients"):
    attributions_list = []
    for idx in indices:
        input_tensor = inputs[idx].unsqueeze(0).to(device)
        target_tensor = target[idx].unsqueeze(0).to(device)
        attributions = compute_attributions(model, input_tensor, target_tensor, method)
        attributions_list.append(attributions.squeeze(0).cpu().detach().numpy())
    
    avg_attributions = np.mean(attributions_list, axis=0)
    return avg_attributions


# Function to visualize the average attributions
def visualize_average_attributions(avg_attributions, input_map):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Input")
    plt.imshow(input_map, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Average Attributions")
    plt.imshow(avg_attributions, cmap='hot')
    plt.show()