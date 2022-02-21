from numba import njit
import pickle
import numpy as np
import rockhound as rh
from tqdm import tqdm
from scipy.ndimage import label 


with open("data/GLBsearchresults.pickle","rb") as f:
    physical,grid,baths,bathtubs,bathtub_depths = pickle.load(f)


def get_descendants(region_d,region_p,ls_in_slice_u):
    descendants = ls_in_slice_u
    for eyed in ls_in_slice_u:
        if eyed != 0:
            descendents = np.concatenate((descendants,region_d[eyed]))
    return np.unique(descendants)

def build_contour_tree(bedvalues,step=20,start=-2000,stop=0):
    ## previous slice
    previous_slice = np.full_like(bedvalues,0)
    unique_id = 0
    region_points = {}
    region_depths = {}
    region_descendents = {}
    region_parents = {}
    region_map = np.full_like(bedvalues,np.nan)
    next_slice =  np.full_like(bedvalues,0)
    for depth in tqdm(range(-2001,0,step)):
        next_slice[:] =0
        labels, c = label(bedvalues<depth)
        for label_number in tqdm(range(1,c+1)):
            label_mask = np.asarray(labels==label_number)
            ls_in_slice = previous_slice[label_mask]
            ls_in_slice_u = np.unique(ls_in_slice)
            if len(ls_in_slice_u)>2 or np.nanmin(ls_in_slice_u)==0 :
                ## this means that we are going to merge to regions
                new_region_id = unique_id-1
                unique_id -=1
                region_map[np.logical_and(label_mask,previous_slice==0)]=new_region_id
                region_depths[new_region_id] = [depth]
                region_descendents[new_region_id] = get_descendants(region_descendents,region_parents,ls_in_slice_u)
                for eyed in ls_in_slice_u:
                    region_parents[eyed] = new_region_id
                next_slice[label_mask] = new_region_id
            else:
                ## This regions just growing in volume or staying the same
                ## the id will be the previous slices id at that location that isn't 0,
                ## our ids are negative so we can grab that with the minimum
                region_id = np.min(ls_in_slice_u)

                #coords = np.where(np.logical_and(label_mask,previous_slice==0))
                region_map[np.logical_and(label_mask,previous_slice==0)]=region_id
                #region_points[new_region_id] = np.concatenate((region_points[new_region_id],np.ravel_multi_index(coords,bedvalues.shape)))
                #region_points[region_id].append(np.where(label_mask))
                next_slice[label_mask] = region_id
        # fig,(ax1,ax2) = plt.subplots(1,2)
        # print(labels)
        # ax1.imshow(labels)
        # ax2.imshow(next_slice)
        # plt.show()
        previous_slice=next_slice
    return region_points, region_depths, region_descendents,region_parents,region_map

def find_GLIB_of_region(region_id,region_descendents,region_parents,region_depths,ocean_regions):
    start_id = region_id
    previous_id = region_id
    count = 0
    while region_id in region_parents.keys():
        for ocean_id in ocean_regions:
            if ocean_id in region_descendents[region_id] or ocean_id == region_id:
                return previous_id, region_depths[previous_id][0]
        previous_id=region_id
        region_id = region_parents[region_id]
        count+=1
    print("All the way out: ", count)
    return start_id, np.nan

def ocean_regions(region_depths,region_map,icemask):
    ocean = []
    for k in tqdm(region_depths.keys()):
        if region_depths[k][0]<-1880 and ~(((icemask[region_map==k]) ==1).all()):
            ocean.append(k)
    return ocean

bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])

bedvalues = bedmap.bed.values

r_p,r_z, r_d, r_parents, r_m = build_contour_tree(bedvalues,step=5,start=-2000,stop=0)                

with open("data/contourtree.pickle","wb") as f:
   pickle.dump([r_p,r_z,r_d,r_parents,r_m],f)


