import xarray
import rockhound as rh
import xarray as xr
from scipy.ndimage import label 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import pickle
from bathtub import get_line_points, shelf_sort
import copy
from matplotlib import colors
from scipy.ndimage import binary_dilation as bd
from cdw import extract_rignot_massloss

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
    points_of_interest = np.asarray(points_of_interest)
    moving_poi=copy.deepcopy(points_of_interest)

    ## Here's where we'll store actual results
    MSD = points_of_interest.shape[0]*[np.nan]

    ## And let's make sure to count how many points are even eligible for finding the MSD
    ## They are eligible if the points are actually above bed rock at this depth.
    count = 0
    valid_poi = []
    for l in range(len(MSD)):
        if depth[points_of_interest[l][0],points_of_interest[l][1]]<slice_depth:
            count+=1
            valid_poi.append(True)
        else:
            valid_poi.append(False)

    valid_poi = np.asarray(valid_poi)

    ## A binary mask of all points below the slice
    below = ~np.logical_and((depth<=slice_depth),ice!=0)

    dilation = 0


    while np.sum(valid_poi)>0:
        # print(np.sum(valid_poi))

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
                    #print("-"*10)
                    offset = np.where(neighborhood!=0)
                    #print(neighborhood)
                    #print(offset)
                    offset_i = offset[0][0]-resolution
                    offset_j = offset[1][0]-resolution
                    #print(offset_i)
                    #print(offset_j)
                    #print("before: ", regions[moving_poi[idx][0],moving_poi[idx][1]])
                    moving_poi[idx][0] = moving_poi[idx][0]+offset_i
                    moving_poi[idx][1] = moving_poi[idx][1]+offset_j
                    #print("after: ", regions[moving_poi[idx][0],moving_poi[idx][1]])
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

    # plt.imshow(np.logical_and((depth<=slice_depth),ice!=0))
    # plt.scatter(points_of_interest.T[1],points_of_interest.T[0],c=MSD,cmap="jet")
    # plt.colorbar()
    # plt.show()
    return MSD
