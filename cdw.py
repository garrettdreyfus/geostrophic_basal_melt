#import shapefile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import rockhound as rh
import matplotlib.pyplot as plt
import cmocean
from xgrads import open_CtlDataset
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

def shelf_average_profile(shelfpolygon,shelfpoints,sal,temp):
    d  = sal.depth.values
    sp = np.asarray(shelfpoints).T
    centroid = np.nanmean(sp,axis=1)
    print(centroid)
    angle = np.degrees(np.arctan2(centroid[1],centroid[0]))
    rdist = np.sqrt((sal.coords["x"]- centroid[0])**2 + (sal.coords["y"] - centroid[1])**2)
    angles = np.degrees(np.arctan2(sal.coords["y"],sal.coords["x"]))
    aslice = np.abs((angles-angle))<20
    lons,lats = np.meshgrid(sal.lon,sal.lat)
    #mask = np.logical_and(np.logical_and(sal.bed.values>-1000, np.isnan(sal.icemask)),rdist<500*10**3)
    #not as nice with less than -1000
    mask = np.logical_and(np.isnan(sal.icemask.values),sal.bed.values<-2000)
    #mask = rdist<1000*10**3
    # greater than -1000 good
    # no ice is better than ice
    #mask = np.logical_and(np.logical_and(sal.bed.values>-1000,np.isnan(sal.icemask.values)),rdist<1000*10**3)
    #mask = np.logical_and.reduce(np.asarray((sal.bed.values<-1000,sal.bed.values>-1100,np.isnan(sal.icemask.values)),rdist<1000*10**3))
    #mask = np.logical_and(np.isnan(sal.icemask.values),rdist<1000*10**3)
    #mask = np.logical_and(np.isnan(sal.icemask.values),rdist<1000*10**3)
    #mask = rdist<1000*10**3
    # plt.imshow(mask)
    # plt.show()
    #mask = np.logical_and(aslice,rdist<1000*10**3)
    #mask = np.logical_and(mask,sal.bed.values>-1000)
    #mask = np.logical_and(aslice,sal.bed.values<0)
    #mask = np.logical_and(rdist<1000*10**3,sal.bed.values>-1000)
    #mask = rdist<1000*10**3
    #mask = rdist>0
    xs,ys = sal.x.values[mask],sal.y.values[mask]
    # plt.scatter(sal.x.values.flatten(),sal.y.values.flatten(),c=sal.bed.values.flatten())
    # plt.scatter(xs,ys)
    # plt.colorbar()
    # plt.show()
    salvals,tempvals = sal.s_an.values[0,:,:,:],temp.t_an.values[0,:,:,:]
    salvals,tempvals = np.moveaxis(salvals,0,-1),np.moveaxis(tempvals,0,-1)
    average_s_profiles = []
    average_t_profiles = []
    coords = []
    for i in shelfpoints:
        coord = np.argmin((xs-i[0])**2 + (ys-i[1])**2)
        coords.append(coord)
        s = gsw.SA_from_SP(salvals[mask][coord][:],d,lons[mask][coord],lats[mask][coord])
        #s = salvals[mask][coord][:]
        #t = tempvals[mask][coord][:]
        t = gsw.CT_from_t(s,tempvals[mask][coord][:],d)
        #print("part?")
        average_s_profiles.append(s)
        average_t_profiles.append(t)
    # average_s_profiles = salvals[mask]
    # average_t_profiles = tempvals[mask]
    shelfpoints = np.asarray(shelfpoints).T
    coords = np.asarray(coords).T
    # plt.scatter(xs,ys)
    # plt.scatter(shelfpoints[0],shelfpoints[1],)
    # plt.scatter(xs[coords[0]],ys[coords[1]])
    # plt.show()
    average_s_profiles = np.asarray(average_s_profiles)
    average_s = np.nanmean(average_s_profiles,axis=0)
    average_t_profiles = np.asarray(average_t_profiles)
    average_t = np.nanmean(average_t_profiles,axis=0)

    return mask, average_t, average_s,d

def generate_shelf_profiles(woafname,is_points,polygons):
    with open("data/woawithbed.pickle","rb") as f:
        sal,temp = pickle.load(f)
    shelf_profiles = {}
    for shelfname in tqdm(polygons.keys()):
        if len(is_points[shelfname]):
            print(shelfname)
            _,average_t, average_s,d = shelf_average_profile(polygons[shelfname],is_points[shelfname],sal,temp)
            shelf_profiles[shelfname] = (average_t,average_s,d)

    shelf_profile_heat_functions = {}
    for k in shelf_profiles.keys():
        t,s,d = shelf_profiles[k] 
        shelf_profile_heat_functions[k] = (interpolate.interp1d(d,np.asarray(t)),interpolate.interp1d(d,np.asarray(s)))

    return shelf_profiles, shelf_profile_heat_functions

def cdw_interface_depth(shelf_profiles):
    interface_depths = {}
    for k in shelf_profiles.keys():
        depths = shelf_profiles[k][2]
        newdepths = range(int(np.nanmin(depths)),int(np.nanmax(depths)))
        ct = np.interp(newdepths,shelf_profiles[k][2],shelf_profiles[k][0])
        sa = np.interp(newdepths,shelf_profiles[k][2],shelf_profiles[k][1])
        pot_dens = gsw.sigma0(sa,ct)+1000
        pt = gsw.pt_from_CT(sa,ct)
        interface_i = int(np.nanargmax(pt)/2)
        nsquared = gsw.Nsquared(sa,ct,newdepths,-70)
        pmid = nsquared[1]
        p_i = np.where(pmid>20)[0][0]
        drho = (np.nanmean(pot_dens[0:interface_i])-np.nanmean(pot_dens[interface_i:interface_i*2]))/((newdepths[interface_i*2]-newdepths[0])/2)#(np.nanmean(pot_dens[0:interface_i*2]))
        gprime = -(9.8*drho)/1025.0
        #plt.plot(nsquared[0],nsquared[1])
        #plt.show()
        nsquared = nsquared[0]
        nsquared = np.nanmedian(nsquared[p_i:interface_i*2])
        nsquared = gprime
        interface_depths[k] = (newdepths[np.nanargmax(pt)]/2,gprime,nsquared)
    return interface_depths

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
    depth = np.abs(depth)
    xnew= np.arange(0,min(depth+0,5000))
    #xnew= np.arange(max(0,depth-25),min(depth+25,5000))
    #print(xnew,depth,max(d))
    ynew = heat_function[0](xnew)
    #xnew,ynew = xnew[ynew>0],ynew[ynew>0]
    ynew = ynew - gsw.CT_freezing(heat_function[1](xnew),xnew,0)
    return np.nansum(ynew>1)/np.sum(~np.isnan(ynew))
    #return np.trapz(ynew,xnew)-np.dot(np.diff(xnew),gsw.CT_freezing(heat_function[1]((xnew[:-1]+xnew[1:])/2),(xnew[:-1]+xnew[1:])/2,0))

def heat_by_shelf(polygons,heat_functions,baths,bedvalues,grid,physical,withGLIB=True,intsize=50):
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
            if shelfname in heat_functions.keys():
                shelf_heat_content.append(heat_content(heat_functions[shelfname],-baths[l],intsize))
                shelf_heat_content_byshelf[shelfname].append(shelf_heat_content[-1])
            else:
                shelf_heat_content.append(np.nan)
                shelf_heat_content_byshelf[shelfname].append(np.nan)
        else:
            shelf_heat_content.append(np.nan)
    return shelf_heat_content, shelf_heat_content_byshelf

def extract_rignot_massloss(fname):
    dfs = pd.read_excel(fname,sheet_name=None)
    dfs = dfs['Dataset_S1_PNAS_2018']
    rignot_shelf_massloss={}
    rignot_shelf_areas = {}
    sigma = {}
    for l in range(len(dfs["Glacier name"])):
        if  dfs["Glacier name"][l] and type(dfs["σ SMB"][l])==float:
            try:
                sigma[dfs["Glacier name"][l]]= float(dfs["σ SMB"][l])+float(dfs["σ D"][l])
                rignot_shelf_massloss[dfs["Glacier name"][l]] = dfs["Cumul Balance"][l]
                rignot_shelf_areas[dfs["Glacier name"][l]] = dfs["Basin.1"][l]
            except:
                1+1
            
    return rignot_shelf_massloss, rignot_shelf_areas,sigma


# llset = open_CtlDataset('data/polynall.ctl')
# projection = pyproj.Proj("epsg:3031")
# lons,lats = np.meshgrid(llset.lon,llset.lat)
# x,y = projection.transform(lons,lats)
# llset.pr.values[0,:,:][llset.pr.values[0,:,:]==0] = np.nan
# llset.coords["x"]= (("lat","lon"),x)
# llset.coords["y"]= (("lat","lon"),y)
# bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
# #
#xvals,yvals = np.meshgrid(bedmap.x,bedmap.y)
# projection = pyproj.Proj("epsg:4326")
# print("transform")
# xvals,yvals = projection.transform(xvals,yvals)
# polyna = np.full_like(bedmap.bed.values,np.nan)
# print("interp")
# from scipy.interpolate import griddata
# xvals,yvals = np.meshgrid(bedmap.x,bedmap.y)
# polyna = griddata(np.asarray([llset.x.values.flatten(),llset.y.values.flatten()]).T,llset.pr.values.flatten(),(xvals,yvals))
# with open("data/polynainterp.pickle","wb") as f:
#     pickle.dump(polyna,f)


