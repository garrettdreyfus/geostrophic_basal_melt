import xarray
import rockhound as rh
import xarray as xr
from scipy.ndimage import label 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import pickle
from scipy.ndimage import binary_dilation as bd

def generateBedmapGLIBs(depth,ice,resolution=5):
    #We don't care about the structure of the bathymetry off the shelf break for now
    # so we'll call the off shelf water the part that's greater than 2000 meters deep
    # and just set it to 2000 meters
    depth[depth<-2000] = -2000

    
    ## Grab a single point in the open ocean. We just made all the open ocean one depth so it is connected 
    openoceancoord = np.where(depth==-2000)
    openoceancoord = (openoceancoord[0][0],openoceancoord[1][0])

    ## Decide the vertical resolution at which we'd like to search from -2000 to 0
    slices = - np.asarray(range(0,2000,resolution)[::-1])
    ## A matrix output we will fill with nans as we go.
    GLIB = np.full_like(depth,np.nan)
    for iso_z in tqdm(slices):
        ### A mask that shows all points below our search depth that arent land or grounded ice.
        below = np.logical_and((depth<=iso_z),ice!=0)
        ### Use the image library label function to label connected regions.
        regions, _ = label(below)
        ### If any region is connected to open ocean at this depth and it hasn't been assigned a GLIB before (all ocean points are connected at z=0)
        ### then set it's glib to iso_z -20. The minus 20 is there because we are looking for the depth at which its not connected
        GLIB[np.logical_and(regions==regions[openoceancoord],np.isnan(GLIB))] = iso_z-resolution
    return GLIB

def calcMSD(depth,ice,slice_depth,resolution=5,debug=False):
    iceexpand = bd(ice==0)
    glline = np.logical_and(iceexpand==1,ice==1)
    glpoints = np.asarray(np.where(glline)).T
    start_point = bd(glline,iterations=5)

    # plt.imshow(ice)
    # plt.scatter(glpoints.T[1],glpoints.T[0])
    # plt.show()

    lowest= (-np.nanmin(depth))-100
    depth[depth<-lowest] = -lowest
    depth[np.isnan(depth)] = -lowest


    ## Grab a single point in the open ocean. We just made all the open ocean one depth so it is connected 
    openoceancoord = np.where(depth==-lowest)
    openoceancoord = (openoceancoord[0][0],openoceancoord[1][0])

    ## Decide the vertical resolution at which we'd like to search from -2000 to 0
    ## A matrix output we will fill with nans as we go.
    MSD = glpoints.shape[0]*[np.nan]
    count=0 
    for l in range(len(MSD)):
        if depth[glpoints[l][0],glpoints[l][1]]<slice_depth:
            count+=1
    below = ~np.logical_and((depth<=slice_depth),ice!=0)
    #plt.imshow(below)
    #plt.show()
    iters =0 
    if debug:
        debugmap = np.full_like(depth,np.nan)
    while np.sum(~np.isnan(MSD))<count:
        ### A mask that shows all points below our search depth that arent land or grounded ice.
        ### Use the image library label function to label connected regions.
        regions, _ = label(~below)
        ### If any region is connected to open ocean at this depth and it hasn't been assigned a GLIB before (all ocean points are connected at z=0)
        ### then set it's glib to iso_z -20. The minus 20 is there because we are looking for the depth at which its not connected
        for l in range(len(MSD)):
            if depth[glpoints[l][0],glpoints[l][1]]<slice_depth and \
              np.isnan(MSD[l]) and \
              regions[glpoints[l][0],glpoints[l][1]]!=regions[openoceancoord]:
                MSD[l]=iters
        #plt.imshow(regions)
        #plt.scatter(glpoints.T[1],glpoints.T[0],c=MSD,cmap="jet")
        #plt.show()
        if debug:
            debugmap[np.logical_and(below==1,np.isnan(debugmap))] = iters
        iters+=resolution
        below = bd(below,iterations=resolution,mask=~start_point)
    # plt.imshow(below)
    # plt.scatter(glpoints.T[1],glpoints.T[0],c=MSD,cmap="jet")
    # plt.colorbar()
    # plt.show()
    # plt.hist(MSD)
    # plt.show()
    if debug:
        return MSD,debugmap
    else:
        return MSD
# Retrieve  the bathymetry and ice data
shelf = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
# Get matrix of bathymetry values
depth = np.asarray(shelf.bed.values)
# Get matrix of ice values (grounded floating noice)
ice = np.asarray(shelf.icemask_grounded_and_shelves.values)

depth = depth[2400:2900,4600:6200]
ice = ice[2400:2900,4600:6200]

iceexpand = bd(ice==0)

glline = np.logical_and(iceexpand==1,ice==1)
glpoints = np.asarray(np.where(glline)).T
#GLIB = generateBedmapGLIBs(depth,ice,resolution=5)
MSDS = []
for k in range(-700,-100,20):
    MSD, debug = calcMSD(depth,ice,k,resolution=5,debug=True)
    MSDS.append(MSD)
    plt.imshow(debug)
    plt.scatter(glpoints.T[1],glpoints.T[0],c=MSDS[-1])
    plt.show()
MSDS = np.asarray(MSDS)
# with open("data/MSDS.pickle","wb") as f:
#    pickle.dump(MSDS,f)
# with open("data/GLIBnew.pickle","rb") as f:
#     GLIB = pickle.load(f)
# with open("data/MSDS.pickle","rb") as f:
#     MSDS = pickle.load(f)
# depths = np.asarray(range(-700,-100,20))
# for l in range(10000):#MSDS.shape[1]):
    # plt.plot(MSDS[:,l],depths-GLIB[glpoints[l][0],glpoints[l][1]])
slopes = []
for l in range(MSDS.shape[1]):
    deltas = depths-GLIB[glpoints[l][0],glpoints[l][1]]
    d1 = 0
    count=0
    for step in range(20):
        if np.argmin(np.abs(deltas))+step<MSDS.shape[0]:
            count+=1
            d1 += MSDS[:,l][np.argmin(np.abs(deltas))+step]
    slopes.append(d1/count)
plt.hist(slopes)
plt.show()
plt.imshow(depth)
plt.scatter(glpoints.T[1],glpoints.T[0],c=slopes,cmap="jet")
plt.colorbar()
plt.show()
