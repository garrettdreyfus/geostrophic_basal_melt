import shapefile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import shapely
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
from scipy.ndimage import binary_dilation
import skfmm
from matplotlib import collections  as mc
from functools import partial
from tqdm.contrib.concurrent import process_map
from scipy.ndimage import binary_dilation as bd
from scipy.ndimage import label 
import rioxarray as riox
import rasterio

def quad_local_param(thermal_forcing):
    
    """
    Apply the quadratic local parametrization.
    
    This function computes the basal melt based on a quadratic local parametrization (based on Favier et al., 2019 or DeConto and Pollard, 2016), revisited in Burgard et al. 2021.
    
    Parameters
    ----------
    gamma : scalar
        Heat exchange velocity in m per second
    melt_factor : scalar 
        Melt factor representing (rho_sw*c_pw) / (rho_i*L_i) in :math:`\mathtt{K}^{-1}`.
    thermal_forcing: scalar or array
        Difference between T and the freezing temperature Tf (T-Tf) in K or degrees C
    U_factor : scalar or array
        Factor introduced to emulate the speed of the current, see function calculate_melt_rate_2D_simple_1isf.
    Returns
    -------
    melt : scalar or array
        Melt rate in m ice per second
    """
    rho_sw = 1028. # kg m-3
    c_po = 3974. # J kg-1 K-1
    rho_i = 917. # kg m-3
    L_i = 3.34 * 10**5# J kg-1
    rho_fw = 1000.

    gamma = 5.9*10**-4

    melt_factor = (rho_sw * c_po) / (rho_i * L_i) # K-1

    U_factor=melt_factor
    
    melt = (gamma * melt_factor * U_factor * thermal_forcing * abs(thermal_forcing))*(31536000)
    return melt 

def heat_content(heat_function,depth,plusminus):
    #heat = gsw.cp_t_exact(s,t,d)
    depth = np.abs(depth)
    #xnew= np.arange(50,min(depth+0,5000))
    #xnew= np.arange(max(0,depth-plusminus),min(depth+plusminus,5000))
    xnew= np.arange(max(0,depth-plusminus),depth)
    #print(xnew,depth,max(d))
    ynew = heat_function[0](xnew)
    #xnew,ynew = xnew[ynew>0],ynew[ynew>0]
    ynew = ynew - gsw.CT_freezing(heat_function[1](xnew),xnew,depth)
    #return np.nansum(ynew>1.2)/np.sum(~np.isnan(ynew))
    if len(ynew)>0:
        return np.trapz(ynew,xnew)/len(xnew)
    else:
        return np.nan
    #return np.trapz(ynew,xnew)-np.dot(np.diff(xnew),gsw.CT_freezing(heat_function[1]((xnew[:-1]+xnew[1:])/2),(xnew[:-1]+xnew[1:])/2,0))

def heat_content_layer(heat_function,depth,plusminus,zgl):
    #heat = gsw.cp_t_exact(s,t,d)
    depth = np.abs(depth)
    #xnew= np.arange(50,min(depth+0,5000))
    #xnew= np.arange(max(0,depth-plusminus),min(depth+plusminus,5000))
    xnew= np.arange(max(0,depth-500),depth)
    #print(xnew,depth,max(d))
    ynew = heat_function[0](xnew)
    cdwdepth = xnew[np.nanargmax(ynew)]
    #xnew,ynew = xnew[ynew>0],ynew[ynew>0]
    ynew = ynew - gsw.CT_freezing(heat_function[1](xnew),zgl,0)
    #return np.nansum(ynew>1.2)/np.sum(~np.isnan(ynew))
    if len(ynew)>0:
        return cdwdepth-depth#np.trapz(ynew,xnew)/len(xnew)
    else:
        return np.nan
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

def extract_rignot_massloss2019(fname):
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

def extract_rignot_massloss2013(fname):
    dfs = pd.read_excel(fname,sheet_name=None)
    dfs = dfs['Sheet1']
    print(dfs)
    print(dfs.keys())
    rignot_shelf_massloss={}
    rignot_shelf_areas = {}
    sigma = {}
    for l in range(len(dfs["Ice Shelf Name"])):
        if  dfs["Ice Shelf Name"][l]:
            try:
                i = dfs["B my"][l].index("±")
                rignot_shelf_massloss[dfs["Ice Shelf Name"][l]] = dfs["B my"][l][:i]
                rignot_shelf_areas[dfs["Ice Shelf Name"][l]] = dfs["Actual area"][l]
            except:
                1+1
    return rignot_shelf_massloss, rignot_shelf_areas,sigma



def extract_rignot_massloss2013(fname):
    dfs = pd.read_excel(fname,sheet_name=None)
    dfs = dfs['Sheet1']
    rignot_shelf_my = {}
    sigma = {}
    for l in range(len(dfs["Ice Shelf Name"])):
        if  dfs["Ice Shelf Name"][l] and dfs["B my"][l]:
            try:
                mystr = dfs["B my"][l]
                my = float(mystr.split("±")[0])
                sig = float(mystr.split("±")[1])
                rignot_shelf_my[dfs["Ice Shelf Name"][l]] = my
                sigma[dfs["Ice Shelf Name"][l]] = sig
            except:
                1+1
            
    return rignot_shelf_my,sigma

def polyna_dataset(polygons):
    llset = open_CtlDataset('data/polynall.ctl')
    llset = llset.rename({"lat":"y","lon":"x"})
    llset = llset.rio.write_crs("epsg:4326")
    llset["pr"] = llset.pr.mean(axis=0)
    llset.rio.nodata=np.nan
    llset = llset.rio.reproject("epsg:3031")
    llset.pr.values[llset.pr.values>1000]=np.nan
    llset.rio.to_raster("data/polyna.tif")
    plt.show()
    polyna_rates = {}
    for k in tqdm(polygons.keys()):
        raster = riox.open_rasterio('data/polyna.tif')
        raster = raster.rio.write_crs("epsg:3031")
        gons = []
        parts = polygons[k][1]
        polygon = polygons[k][0]
        if len(parts)>1:
            parts.append(-1)
            for l in range(0,len(parts)-1):
                poly_path=shapely.geometry.Polygon(np.asarray(polygon.exterior.coords.xy)[:,parts[l]:parts[l+1]].T).buffer(10**4)
                gons.append(poly_path)
        else:
            gons = [polygon]
        print(k)
        print("made it hearE")
        clipped = raster.rio.clip(gons)
        #plt.imshow(clipped[0])
        #plt.show()
        polyna_rates[k] = np.nanmean(np.asarray(clipped[0])[clipped<1000])
        del clipped

    return polyna_rates


    #llset.rio.to_raster("data/amundsilli.tif")

        #melts = xr.open_dataset(dsfname)
        #melts["phony_dim_1"] = melts.x.T[0]
        #melts["phony_dim_0"] = melts.y.T[0]
        #melts = melts.drop_vars(["x","y"])
        #melts = melts.rename({"phony_dim_1":"x","phony_dim_0":"y"})
        #melts.rio.to_raster("data/amundsilli.tif")
        #print(melts)
#
    #projection = pyproj.Proj("epsg:3031")
    #lons,lats = np.meshgrid(llset.lon,llset.lat)
    #x,y = projection.transform(lons,lats)
    #llset.pr.values[0,:,:][llset.pr.values[0,:,:]==0] = np.nan
    #llset.coords["x"]= (("lat","lon"),x)
    #llset.coords["y"]= (("lat","lon"),y)
    #bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
    #xvals,yvals = np.meshgrid(bedmap.x,bedmap.y)
    #projection = pyproj.Proj("epsg:4326")
    #print("transform")
    #xvals,yvals = projection.transform(xvals,yvals)
    #polyna = np.full_like(bedmap.bed.values,np.nan)
    #print("interp")
    #from scipy.interpolate import griddata
    #xvals,yvals = np.meshgrid(bedmap.x,bedmap.y)
    ##polyna = griddata(np.asarray([llset.x.values.flatten(),llset.y.values.flatten()]).T,llset.pr.values.flatten(),(xvals,yvals))
    #with open("data/polynainterp.pickle","wb") as f:
        #pickle.dump(polyna,f)


def closest_point_pfun(grid,bedvalues,icemask,bordermask,baths,l):
    bath = baths[l]
    if np.isnan(bath):
        bath = bedvalues[grid[l][0],grid[l][1]]
    search = np.full_like(bedvalues,0)
    search[:]=0
    search[grid[l][0],grid[l][1]]=1
    route = np.logical_and(bedvalues<(bath+20),icemask!=0)
    searchsum=0
    if np.sum(np.logical_and(route,bordermask))>0:
        intersection = [[],[]]
        iters = 0
        while len(intersection[0])==0:
            iters+=20
            search = bd(search,mask=route,iterations=50)
            searchsumnew = np.sum(search)
            if searchsum !=searchsumnew:
                searchsum = searchsumnew
            else:
                return (np.nan,np.nan)
            intersection = np.where(np.logical_and(search,bordermask))
        return (intersection[0][0],intersection[1][0])
    else:
        return (np.nan,np.nan)


def closest_WOA_points_bfs(grid,baths,bedmach,debug=False):
    bedvalues = bedmach.bed.values
    icemask = bedmach.icemask_grounded_and_shelves.values
    closest_points=[np.nan]*len(baths)
    depthmask = bedvalues>-2300
    insidedepthmask = bedvalues<-1900
    bordermask = np.logical_and(insidedepthmask,depthmask)
    bordermask = np.logical_and(bordermask,np.isnan(icemask))
    itertrack = []

    f = partial(closest_point_pfun,grid,bedvalues,icemask,bordermask,baths)
    #print(pool.map(f, range(len(grid))))
    #closest_points = pool.map(f, range(10)))
    closest_points = process_map(f, range(int(len(grid))),max_workers=3,chunksize=100)

    try:
        pool.close()
    except Exception as e:
        print(e)

    if debug:
        lines = []
        plt.hist(itertrack)
        for l in tqdm(range(int(len(closest_points)))):
            if ~np.isnan(closest_points[l]).any():
                if closest_points[l][0]==0 and closest_points[l][1]==0:
                    print(l)
                lines.append(([grid[l][1],grid[l][0]],[closest_points[l][1],closest_points[l][0]]))
        lc = mc.LineCollection(lines, colors="red", linewidths=2)
        fig, ax = plt.subplots()
        ax.imshow(bedvalues)
        ax.add_collection(lc)
        plt.show()
    return closest_points


def nearest_nonzero_idx_v2(a,x,y):
    tmp = a[x,y]
    a[x,y] = 0
    r,c = np.nonzero(a)
    a[x,y] = tmp
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    return r[min_idx], c[min_idx]

def closest_WOA_points_simple(grid,baths,bedmach,debug=False):
    bedvalues = bedmach.bed.values
    icemask = bedmach.icemask_grounded_and_shelves.values
    closest_points=[np.nan]*len(baths)
    depthmask = bedvalues>-2300
    insidedepthmask = bedvalues<-1900
    bordermask = np.logical_and(insidedepthmask,depthmask)
    bordermask = np.logical_and(bordermask,np.isnan(icemask))
    itertrack = []
    closest = []
    for l in tqdm(range(len(grid))):
        closest.append(nearest_nonzero_idx_v2(bordermask,grid[l][0],grid[l][1]))
    return closest

def closest_WOA_points(grid,baths,bedmach,debug=False,method="simple"):
    if method=="bfs":
        return closest_WOA_points_bfs(grid,baths,bedmach,debug=False)
    elif method=="simple":
        return closest_WOA_points_simple(grid,baths,bedmach,debug=False)

def running_mean(x, N):
    cumsum = np.nancumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def tempFromClosestPoint(bedmap,grid,physical,baths,closest_points,sal,temp,shelves,debug=False,method="default"):
    print("temp from closest point")
    heats=[np.nan]*len(baths)
    stx = sal.coords["x"].values
    sty = sal.coords["y"].values
    projection = pyproj.Proj("epsg:3031")
    salvals,tempvals = sal.s_an.values[0,:,:,:],temp.t_an.values[0,:,:,:]
    d  = sal.depth.values
    count = 0
    lines = []
    bedvalues = bedmap.bed.values
    for l in tqdm(range(len(closest_points))[::-1]):
        if ~np.isnan(closest_points[l]).any():
            count+=1
            centroid = [bedmap.coords["x"].values[closest_points[l][1]],bedmap.coords["y"].values[closest_points[l][0]]]
            rdist = np.sqrt((sal.coords["x"]- centroid[0])**2 + (sal.coords["y"] - centroid[1])**2)
            closest=np.unravel_index(rdist.argmin(), rdist.shape)
            x = stx[closest[0],closest[1]]
            y = sty[closest[0],closest[1]]
            t = tempvals[:,closest[0],closest[1]]
            s = salvals[:,closest[0],closest[1]]
            lon,lat = projection(x,y,inverse=True)
            s = gsw.SA_from_SP(s,d,lon,lat)
            t = gsw.CT_from_t(s,t,d)
            line = ([physical[l][0],physical[l][1]],[centroid[0],centroid[1]])
            dist = np.sqrt((physical[l][1]-x)**2 + (physical[l][0]-y)**2)
            if ~(np.isnan(line).any()):
                lines.append(line)
            tinterp,sinterp = interpolate.interp1d(d,np.asarray(t)),interpolate.interp1d(d,np.asarray(s))
            # plt.scatter(t,d)
            # plt.plot(tinterp(d),d)
            # print(tinterp(d))
            # plt.show()
            if method=="default":
                if np.isnan(t[11:]).all():
                    heats[l]=np.nan#
                elif np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                    heats[l]=heat_content((tinterp,sinterp),baths[l],200)
            elif method=="a":
                zglib = baths[l]
                zgl = bedvalues[grid[l][0],grid[l][1]]
                closest_t = t[11:][np.nanargmin(np.abs(d[11:]+zglib))]
                if ~np.isnan(t[11:]).all():
                    diff_t = running_mean(np.diff(t[11:]),3)
                    cline = d[11:][np.nanargmax(diff_t)]
                    #tmax = np.nanmax(t[11:])
                    tmax=0
                    zpyc = d[11:][np.nanargmax(t[11:])]
                    n2,_ = gsw.Nsquared(s[11:],t[11:],d[11:])
                    n2 = np.asarray(n2)
                    #print(tmax/2)
                    if np.sum(t[11:]>0)>0:
                        #print(t[11:]>0)
                        z0 = np.nanmin(d[11:][t[11:]>0])
                        #z0 = np.median(t)
                    else:
                        z0=np.nanmin(d[11:])
                    heats[l]=abs(zglib)-abs(z0)
                    if zpyc>zglib and zglib > zgl:
                        heats[l]=abs(zpyc)-abs(zglib)
                    else:
                        heats[l]=np.nan
                else:
                    heats[l]=np.nan

    if debug:
        fig, ax = plt.subplots()
        lc = mc.LineCollection(lines, colors="red", linewidths=2)
        #ax.imshow(bedvalues)
        plt.xlim(-10**7,10**7)
        plt.ylim(-10**7,10**7)
        ax.add_collection(lc)
        plt.show()
    return heats

def tempFromClosestPointSimple(bedmap,grid,physical,baths,closest_points,sal,temp,debug=False,method="default"):
    print("temp from closest point")
    heats=[np.nan]*len(baths)
    stx = sal.coords["x"].values
    sty = sal.coords["y"].values
    salvals,tempvals = sal.s_an.values[0,:,:,:],temp.t_an.values[0,:,:,:]
    d  = sal.depth.values
    mask = np.zeros(salvals.shape[1:])
    mask[:]=np.inf
    for l in range(salvals.shape[1]):
        for k in range(salvals.shape[2]):
            if np.sum(~np.isnan(salvals[:,l,k]))>0 and np.max(d[~np.isnan(salvals[:,l,k])])>2000:
                mask[l,k] = 1
    count = 0
    lines = []
    bedvalues = bedmap.bed.values
    for l in tqdm(range(int(len(closest_points)))):
        if ~np.isnan(closest_points[l]).any():
            count+=1
            centroid = [bedmap.coords["x"].values[grid[l][1]],bedmap.coords["y"].values[grid[l][0]]]
            rdist = np.sqrt((sal.coords["x"]-centroid[0])**2 + (sal.coords["y"]- centroid[1])**2)*mask
            closest=np.unravel_index(rdist.argmin(), rdist.shape)
            x = stx[closest[0],closest[1]]
            y = sty[closest[0],closest[1]]
            t = tempvals[:,closest[0],closest[1]]
            s = salvals[:,closest[0],closest[1]]
            line = ([physical[l][0],physical[l][1]],[x,y])
            dist = np.sqrt((physical[l][0]-x)**2 + (physical[l][1]-y)**2)
            if ~(np.isnan(line).any()):
                lines.append(line)
            tinterp,sinterp = interpolate.interp1d(d,np.asarray(t)),interpolate.interp1d(d,np.asarray(s))
            # plt.scatter(t,d)
            # plt.plot(tinterp(d),d)
            # print(tinterp(d))
            # plt.show()
            if ~(np.isnan(line).any()):
                lines.append(line)
            if np.isnan(t[11:]).all():
                heats[l]=np.nan
            elif np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                heats[l]=heat_content((tinterp,sinterp),baths[l],6000,0)#*((baths[l]-bedvalues[grid[l][1],grid[1][0]])/(dist))
    if debug:
        fig, ax = plt.subplots()
        lc = mc.LineCollection(lines, colors="red", linewidths=2)
        #ax.imshow(bedvalues)
        plt.xlim(-10**7,10**7)
        plt.ylim(-10**7,10**7)
        ax.add_collection(lc)
        plt.show()
    return heats

def tempProfileByShelf(bedmap,grid,physical,depths,closest_points,sal,temp,shelf_keys,debug=False,method="default"):
    print("temp from closest point")
    stx = sal.coords["x"].values
    sty = sal.coords["y"].values
    salvals,tempvals = sal.s_an.values[0,:,:,:],temp.t_an.values[0,:,:,:]
    d  = sal.depth.values
    mask = np.zeros(salvals.shape[1:])
    mask[:]=np.inf
    for l in range(salvals.shape[1]):
        for k in range(salvals.shape[2]):
            if np.sum(~np.isnan(salvals[:,l,k]))>2 and np.max(d[~np.isnan(salvals[:,l,k])])>1500:
                mask[l,k] = 1
    count = 0
    lines = []
    bedvalues = bedmap.bed.values
    tempprofs = []
    salprofs = []
    for l in tqdm(range(int(len(closest_points)))):
        if ~np.isnan(closest_points[l]).any():
            count+=1
            centroid = [bedmap.coords["x"].values[grid[l][1]],bedmap.coords["y"].values[grid[l][0]]]
            rdist = np.sqrt((sal.coords["x"]-centroid[0])**2 + (sal.coords["y"]- centroid[1])**2)*mask
            rdist[np.where(rdist<250000)]=np.inf
            closest=np.unravel_index(rdist.argmin(), rdist.shape)
            x = stx[closest[0],closest[1]]
            y = sty[closest[0],closest[1]]
            t = tempvals[:,closest[0],closest[1]]
            s = salvals[:,closest[0],closest[1]]
            line = ([physical[l][0],physical[l][1]],[x,y])
            dist = np.sqrt((physical[l][0]-x)**2 + (physical[l][1]-y)**2)
            if ~(np.isnan(line).any()):
                lines.append(line)
            #tinterp,sinterp = interpolate.interp1d(d,np.asarray(t)),interpolate.interp1d(d,np.asarray(s))
            tempprofs.append(t)
            salprofs.append(s)
            line = ([x,y],[centroid[0],centroid[1]])
    tempprofs = np.asarray(tempprofs)
    salprofs = np.asarray(salprofs)
    prof_dict = {}
    shelf_keys = np.asarray(shelf_keys)
    shelf_keys[shelf_keys==None] = "None"

    fig, ax = plt.subplots()
    lc = mc.LineCollection(lines, colors="red", linewidths=2)
    #ax.imshow(bedvalues)
    plt.xlim(-10**7,10**7)
    plt.ylim(-10**7,10**7)
    ax.add_collection(lc)
    plt.show()

    for k in np.unique(shelf_keys):
        if k !="None":
            t = np.mean(tempprofs[shelf_keys==k],axis=0)
            s = np.mean(salprofs[shelf_keys==k],axis=0)
            idd = np.nanmin(np.asarray(depths)[shelf_keys==k])
            prof_dict[k] = (t,s,d,idd)
    return prof_dict

def GLIB_by_shelf(GLIB,bedmach,polygons):
    GLIBmach = bedmach.bed.copy(deep=True)
    GLIBmach.values[:] = GLIB[:]
    GLIBmach = GLIBmach.rio.write_crs("epsg:3031")
    print(GLIBmach)
    del GLIBmach.attrs['grid_mapping']
    GLIBmach.rio.to_raster("data/glibmach.tif")

    glib_by_shelf = {}
    for k in tqdm(polygons.keys()):
        raster = riox.open_rasterio('data/glibmach.tif')
        gons = []
        parts = polygons[k][1]
        polygon = polygons[k][0]
        if len(parts)>1:
            parts.append(-1)
            for l in range(0,len(parts)-1):
                poly_path=shapely.geometry.Polygon(np.asarray(polygon.exterior.coords.xy)[:,parts[l]:parts[l+1]].T).buffer(10**4)
                gons.append(poly_path)
        else:
            gons = [polygon]
        print(k)
        print("made it hearE")
        clipped = raster.rio.clip(gons)
        if k == "Ronne":
            plt.imshow(clipped[0])
            plt.show()
        glib_by_shelf[k] = np.nanmean(np.asarray(clipped[0])[clipped>-9000])
        print(glib_by_shelf[k])
    return glib_by_shelf
 
