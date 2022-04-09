import xarray
import rockhound as rh
import xarray as xr
from scipy.ndimage import label 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Retrieve  the bathymetry and ice data
shelf = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
# Get matrix of bathymetry values
depth = np.asarray(shelf.bed.values)
# Get matrix of ice values (grounded floating noice)
ice = np.asarray(shelf.icemask_grounded_and_shelves.values)

#We don't care about the structure of the bathymetry off the shelf break for now
# so we'll call the off shelf water the part that's greater than 2000 meters deep
# and just set it to 2000 meters
depth[depth<-2000] = -2000

## Grab a single point in the open ocean. We just made all the open ocean one depth so it is connected 
openoceancoord = np.where(depth==-2000)
openoceancoord = (openoceancoord[0][0],openoceancoord[1][0])

## Decide the vertical resolution at which we'd like to search from -2000 to 0
slices = - np.asarray(range(0,2000,5)[::-1])
## A matrix output we will fill with nans as we go.
GLIB = np.full_like(depth,np.nan)
for iso_z in tqdm(slices):
    ### A mask that shows all points below our search depth that arent land or grounded ice.
    below = np.logical_and((depth<=iso_z),ice!=0)
    ### Use the image library label function to label connected regions.
    regions, _ = label(below)
    ### If any region is connected to open ocean at this depth and it hasn't been assigned a GLIB before (all ocean points are connected at z=0)
    ### then set it's glib to iso_z -20. The minus 20 is there because we are looking for the depth at which its not connected
    GLIB[np.logical_and(regions==regions[openoceancoord],np.isnan(GLIB))] = iso_z-20
    

plt.imshow(GLIB,cmap="magma")
plt.clim(-700,0)
plt.colorbar()
plt.show()
