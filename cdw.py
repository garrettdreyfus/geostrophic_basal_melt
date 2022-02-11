import shapefile
import rockhound as rh
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import itertools, pickle
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import time,gsw,xarray, pyproj
from bathtub import closest_shelf,convert_bedmachine
from scipy import interpolate
import pandas as pd
from copy import copy
import xarray as xr



def create_depth_mask(da,d,limit):
    depthmask = np.full_like(da[0,0,:,:].values,np.nan,dtype=bool)
    for i in range(depthmask.shape[0]):
        for j in range(depthmask.shape[1]):
            if ~(np.isnan(da.values[0,:,i,j])).all():
                depthmask[i,j] = (d[~np.isnan(da.values[0,:,i,j])][-1]>limit)
    return depthmask

def shelf_average_profile(shelf,sal,temp,d):
    centroid = list(shelf.centroid.coords)[0]
    mask = np.full_like(sal.s_an[0,0,:,:].values,np.nan,dtype=bool)
    dist = np.sqrt((sal.coords["x"]- centroid[0])**2 + (sal.coords["y"] - centroid[1])**2)
    radius=1000*10**3
    mask[dist<radius] = True
    mask[dist>radius] = False
    average_s = []
    average_t = []
    [LAT,LON] = np.meshgrid(sal.lon,sal.lat)
    for i in range(len(d)):
        if np.sum(~np.isnan(sal.s_an.values[0,i][mask])) < 10:
            average_s.append(np.nan)
            average_t.append(np.nan)
        else:
            s = gsw.SA_from_SP(sal.s_an.values[0,i][mask],d[i],LON[mask],LAT[mask])
            t = gsw.CT_from_t(s,temp.t_an.values[0,i][mask],d[i])
            average_s.append(np.nanmean(sal.s_an.values[0,i][mask]))
            average_t.append(np.nanmean(temp.t_an.values[0,i][mask]))
    return mask, average_t, average_s,d

def generate_shelf_profiles(salfname,tempfname,polygons,shelf):
    sal = xarray.open_dataset(salfname,decode_times=False)
    temp = xarray.open_dataset(tempfname,decode_times=False)
    sal = sal.where(sal.lat<-60,drop=True)
    temp= temp.where(sal.lat<-60,drop=True)
    t_an = temp.t_an.values
    s_an = sal.s_an.values
    d = sal.depth.values
    lons=sal.lon.values
    lats=sal.lat.values
    projection = pyproj.Proj("epsg:3031")
    lons,lats = np.meshgrid(sal.lon,sal.lat)
    x,y = projection.transform(lons,lats)
    sal.coords["x"]= (("lat","lon"),x)
    sal.coords["y"]= (("lat","lon"),y)
    depthmask = create_depth_mask(sal.s_an,sal.depth.values,1000)
    shelf_profiles = {}
    for shelfname in polygons.keys():
        shelf = polygons[shelfname]
        _,average_t, average_s,d = shelf_average_profile(shelf,sal,temp,d)
        shelf_profiles[shelfname] = (average_t,average_s,d)

    shelf_profile_heat_functions = {}
    for k in shelf_profiles.keys():
        t,s,d = shelf_profiles[k] 
        shelf_profile_heat_functions[k] = interpolate.interp1d(d,np.asarray(t))

    return shelf_profiles, shelf_profile_heat_functions

def ice_boundary_in_bathtub(bathtubs,icemask):
    ice_boundary_points = []
    mapmask = np.full_like(icemask,np.nan,dtype=float)
    for bathtub in tqdm(bathtubs):
        count = 0
        for k in range(len(bathtub[0])):
            i,j = bathtub[0][k],bathtub[1][k]
            a = icemask[i+1][j]
            b = icemask[i-1][j]
            c = icemask[i][j+1]
            d = icemask[i][j-1]
            if np.isnan(icemask[i,j]) and (np.asarray([a,b,c,d])==1).any():
                count+=1
        ice_boundary_points.append(count)
        mapmask[bathtub]=count
    return ice_boundary_points


def heat_content(heat_function,depth,plusminus):
    #heat = gsw.cp_t_exact(s,t,d)
    xnew= np.arange(max(0,depth-plusminus),min(depth+plusminus,5000))
    #print(xnew,depth,max(d))
    ynew = heat_function(xnew)
    return np.trapz(ynew,xnew)-np.ptp(xnew)*gsw.CT_freezing(34.5,depth,0)

def heat_by_shelf(polygons,heat_functions,baths,bedvalues,grid,physical,withGLIB=True):
    shelf_heat_content = []
    shelf_heat_content_byshelf={}
    shelf_ice_boundary_byshelf={}
    for k in polygons.keys():
        shelf_heat_content_byshelf[k] = []
        shelf_ice_boundary_byshelf[k] = []

    if withGLIB:
        for l in range(len(baths)):
            if baths[l]>=0:
                baths[l]=bedvalues[grid[l][1],grid[l][0]]
    else:
        for l in range(len(baths)):
            baths[l]=bedvalues[grid[l][1],grid[l][0]]

    for l in tqdm(range(len(baths))):
        if baths[l]<0:
            coord = physical[l]
            shelfname, _,_ = closest_shelf(coord,polygons)
            shelf_heat_content.append(heat_content(heat_functions[shelfname],-baths[l],50))
            shelf_heat_content_byshelf[shelfname].append(shelf_heat_content[-1])
    return shelf_heat_content, shelf_heat_content_byshelf

def extract_rignot_massloss(fname):
    dfs = pd.read_excel(fname,sheet_name=None)
    dfs = dfs['Dataset_S1_PNAS_2018']
    print(dfs.keys())
    rignot_shelf_massloss={}
    for l in range(len(dfs["Glacier name"])):
        if  dfs["Glacier name"][l]:
            rignot_shelf_massloss[dfs["Glacier name"][l]] = dfs["Cumul Balance"][l]
    return rignot_shelf_massloss

