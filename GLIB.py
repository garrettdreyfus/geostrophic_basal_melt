import xarray
import rockhound as rh
import xarray as xr
from scipy.ndimage import label 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import pickle
import copy
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

def calcMSD(depth,ice,slice_depth,points_of_interest,resolution=5,debug=False):

    # First let set the open ocean all to one depth
    lowest= (-np.nanmin(depth))-100
    depth[depth<-lowest] = -lowest
    depth[np.isnan(depth)] = -lowest

    ## Grab a single point in the open ocean. We just made all the open ocean one depth so it is connected 
    openoceancoord = np.where(depth==-lowest)
    openoceancoord = (openoceancoord[0][0],openoceancoord[1][0])

    ## We're going to need to move the points of interest
    ## this is the drawstring bag approach of course
    moving_poi=copy.deepcopy(points_of_interest)

    ## Here's where we'll store actual results
    MSD = glpoints.shape[0]*[np.nan]

    ## And let's make sure to count how many points are even eligible for finding the MSD
    ## They are eligible if the points are actually above bed rock at this depth.
    count = 0
    valid_poi = []
    for l in range(len(MSD)):
        if depth[glpoints[l][0],glpoints[l][1]]<slice_depth:
            count+=1
            valid_poi.append(True)
        else:
            valid_poi.append(False)

    valid_poi = np.asarray(valid_poi)

    ## A binary mask of all points below the slice
    below = ~np.logical_and((depth<=slice_depth),ice!=0)

    dilation = 0


    while np.sum(valid_poi)>0:

        ## we now calculate the connectedness
        regions, _ = label(~below)

        ## We need to move the points of interest out of
        ## bedrock if they have been dilated within

        for idx in range(len(moving_poi)):
            if regions[moving_poi[idx][0],moving_poi[idx][1]]==0 and valid_poi[idx]:
                i_idx = moving_poi[idx][0]
                j_idx = moving_poi[idx][1]
                r = resolution
                left = max(i_idx-r,0)
                right = min(i_idx+r+1,depth.shape[0])
                down = max(j_idx-r,0)
                up = min(j_idx+r+1,depth.shape[1])
                neighborhood = regions[left:right,down:up]
                if np.max(neighborhood)>0:
                    print("-"*10)
                    offset = np.where(neighborhood!=0)
                    print(neighborhood)
                    print(offset)
                    offset_i = offset[0][0]-resolution
                    offset_j = offset[1][0]-resolution
                    print(offset_i)
                    print(offset_j)
                    print("before: ", regions[moving_poi[idx][0],moving_poi[idx][1]])
                    moving_poi[idx][0] = moving_poi[idx][0]+offset_i
                    moving_poi[idx][1] = moving_poi[idx][1]+offset_j
                    print("after: ", regions[moving_poi[idx][0],moving_poi[idx][1]])
                else:
                    MSD[idx]=dilation
                    valid_poi[idx]=False
                        
        ## now we check for connectedness

        for idx in range(len(moving_poi)):
            if regions[moving_poi[idx][0],moving_poi[idx][1]] != regions[openoceancoord] and \
               valid_poi[idx]:
                MSD[idx]=dilation
                valid_poi[idx]=False

        # plt.imshow(regions)
        # plt.scatter(moving_poi.T[1],moving_poi.T[0],c="red")
        # plt.colorbar()
        # plt.show()

        below = bd(below,iterations=resolution)
        dilation+=resolution

    plt.imshow(np.logical_and((depth<=slice_depth),ice!=0))
    plt.scatter(points_of_interest.T[1],points_of_interest.T[0],c=MSD,cmap="jet",vmin=0,vmax=50)
    plt.colorbar()
    plt.show()
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

glline = np.logical_and(iceexpand==1,ice!=0)
plt.imshow(ice)
plt.show()
glpoints = np.asarray(np.where(glline)).T
#GLIB = generateBedmapGLIBs(depth,ice,resolution=5)
plt.imshow(ice)
plt.scatter(glpoints.T[1],glpoints.T[0])
plt.show()

MSDS = []

MSD, debug = calcMSD(depth,ice,-200,glpoints,resolution=1,debug=True)

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
