import configs
import json
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plotmap(data, conifg, colorscheme, imagefilepath, subplotquantity = None, extent = None):
    if subplotquantity == None: 
        if ichannel == 0:
            img = averagedphase[..., ichannel].plot(ax=ax[correctorder[phase]], cmap='BrBG', transform=ccrs.PlateCarree(), add_colorbar = False)
            ax[correctorder[phase]].set_extent(extent, crs=ccrs.PlateCarree())
            ax[correctorder[phase]].coastlines()
        elif ichannel == 1:
            img = averagedphase[..., ichannel].plot(ax=ax[correctorder[phase]], cmap='coolwarm', transform=ccrs.PlateCarree(), add_colorbar = False)
            ax[correctorder[phase]].set_extent(extent, crs=ccrs.PlateCarree())
            ax[correctorder[phase]].coastlines()
        if phase == 0:
            ax[correctorder[phase]].set_title(f'Neutral', x = -0.08,  y = 0.425, pad = 14, size = fonty)
        else:
            ax[correctorder[phase]].set_title(f'Phase {phase}', x = -0.08, y = 0.425, pad = 14,  size = fonty)

        plt.suptitle(f"Ensemble " + str(iens+1)+ " - Input Variable: " + str(config["databuilder"]["input_vars"][ichannel]+"\n"), fontsize = fonty)
        plt.tight_layout()

        cbar_ax = fig.add_axes([1.01, 0.28, 0.02, 0.4])
        cbar_ax.tick_params(labelsize=fonty)
        fig.colorbar(img, cax=cbar_ax)

    elif subplotquantity != None: 
           
        fig, ax = plt.subplots(9, 1, figsize=(10, 20), subplot_kw={'projection': ccrs.PlateCarree()})
        extent = [ 40, 180, -14.5, 14.5]
        fonty = 20

        if ichannel == 0:
            img = averagedphase[..., ichannel].plot(ax=ax[correctorder[phase]], cmap='BrBG', transform=ccrs.PlateCarree(), add_colorbar = False)
            ax[correctorder[phase]].set_extent(extent, crs=ccrs.PlateCarree())
            ax[correctorder[phase]].coastlines()
        elif ichannel == 1:
            img = averagedphase[..., ichannel].plot(ax=ax[correctorder[phase]], cmap='coolwarm', transform=ccrs.PlateCarree(), add_colorbar = False)
            ax[correctorder[phase]].set_extent(extent, crs=ccrs.PlateCarree())
            ax[correctorder[phase]].coastlines()
        if phase == 0:
            ax[correctorder[phase]].set_title(f'Neutral', x = -0.08,  y = 0.425, pad = 14, size = fonty)
        else:
            ax[correctorder[phase]].set_title(f'Phase {phase}', x = -0.08, y = 0.425, pad = 14,  size = fonty)


        plt.suptitle(f"Ensemble " + str(iens+1)+ " - Input Variable: " + str(config["databuilder"]["input_vars"][ichannel]+"\n"), fontsize = fonty)
        plt.tight_layout()

        cbar_ax = fig.add_axes([1.01, 0.28, 0.02, 0.4])
        cbar_ax.tick_params(labelsize=fonty)
        fig.colorbar(img, cax=cbar_ax)

    plt.savefig('/Users/C830793391/Documents/Research/E3SM/visuals/' + str(ens) + '/' + str(ens) + str(config["databuilder"]["input_vars"][ichannel])+ '1900-1950.png', format='png', bbox_inches ='tight', dpi = config["fig_dpi"], transparent =True)
    plt.show() 