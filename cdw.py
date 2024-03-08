import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import shapely
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import cmocean
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
from matplotlib.patches import Rectangle
import gsw,xarray, pyproj
from bathtub import closest_shelf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import interpolate
import pandas as pd
from copy import copy
from scipy.ndimage import binary_dilation
from matplotlib import collections  as mc
from functools import partial
from tqdm.contrib.concurrent import process_map
from scipy.ndimage import binary_dilation as bd, label, gaussian_filter 
from scipy.io import savemat
import rioxarray as riox
import rasterio

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w
  
def newtemp(heat_function,hub,shelf_key=None):
    hub = np.abs(hub)
    zi = np.arange(0,1500,1)
    ti = heat_function[0](zi)
    si = heat_function[1](zi)
    di = gsw.rho(si,ti,750)-1000
    if len(np.where(ti-ti[0]>0.2)[0])>0:
        mldi = np.where(ti-ti[0]>0.2)[0][0]
        mld = zi[mldi]
    else:
        mld=0

    di = moving_average(di,20)
    zi = moving_average(zi,20)
    ti = moving_average(ti,20)
    si = moving_average(si,20)
    dizi = np.diff(di)/np.diff(zi)
    thresh = np.quantile(dizi,0.98)

    zpyc = np.mean(zi[1:][dizi>thresh])
    zpyci = np.argmin(np.abs(zi-zpyc))
    hubi = np.argmin(np.abs(zi-hub))

    if hubi==0:hubi=2

    if zpyci<=hubi and hubi-zpyci<5:
        zpyci = hubi-5

    if zpyci>hubi: 
        zpyci=max(hubi-5,0)


    if ~(np.isnan(di).all()):
        return np.mean(ti[zpyci:hubi]-gsw.CT_freezing(si[zpyci:hubi],zi[zpyci:hubi],0))
    else:
        return np.nan


def gprime(heat_function,hub,shelf_key=None,lat=None,lon=None):
    hub = np.abs(hub)
    zi = np.arange(5,1500,1)
    ti = heat_function[0](zi)
    si = heat_function[1](zi)
    di = gsw.rho(si,ti,300)-1000
    if len(np.where(ti-ti[0]>0.2)[0])>0:
        mldi = np.where(ti-ti[0]>0.2)[0][0]
        mld = zi[mldi]
    else:
        mld=0

    di = moving_average(di,50)
    zi = moving_average(zi,50)
    ti = moving_average(ti,50)
    si = moving_average(si,50)

    hubi = np.argmin(np.abs(zi-abs(hub)))
    dizi = np.diff(di)/np.diff(zi)
    thresh = np.quantile(dizi,0.90)
    zpyc = np.mean(zi[1:][dizi>thresh])
    zpyci = np.argmin(np.abs(zi-zpyc))
    di = gsw.rho(si,ti,0)

    rho_1i = np.logical_and(zi<zi[zpyci],zi>zi[zpyci]-zpyci)
    rho_2i = np.logical_and(zi<zi[zpyci]+zpyci,zi>zi[zpyci])
    gprime_ext = 9.8*(np.mean(di[rho_1i])-np.mean(di[rho_2i]))/np.mean(di[np.logical_or(rho_1i,rho_2i)])
       
    if ~(np.isnan(di).all()):
        return gprime_ext
    else:
        return np.nan

def pycnocline(heat_function,hub,shelf_key=None,lat=None,lon=None):
    hub = np.abs(hub)
    zi = np.arange(5,1500,1)
    ti = heat_function[0](zi)
    si = heat_function[1](zi)
    di = gsw.rho(si,ti,300)-1000
    if len(np.where(ti-ti[0]>0.2)[0])>0:
        mldi = np.where(ti-ti[0]>0.2)[0][0]
        mld = zi[mldi]
    else:
        mld=0

    di = moving_average(di,50)
    zi = moving_average(zi,50)
    ti = moving_average(ti,50)
    si = moving_average(si,50)
    dizi = np.abs(np.diff(di)/np.diff(zi))
    thresh = np.quantile(dizi,0.98)
    zpyc = np.mean(zi[1:][dizi>thresh])
    zpyci = np.argmin(np.abs(zi-zpyc))
    di = gsw.rho(si,ti,zpyc)-1000
    gprime = (np.nanmean(di[zpyci:]) - np.nanmean(di[:zpyci]))/np.nanmean(di[:])

    if shelf_key == "Thwaites" and False:# and (-zpyc+hub) < 75:
        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.plot(ti,-zi)
        ax2.plot(ti,-zi)
        ax1.axhline(y=-hub,color="blue",label="HUB")
        ax1.legend()
        plt.title(str(round(lat,1))+" , "+str(round(lon,1)))
        plt.show()

        fig,(ax1,ax2) = plt.subplots(1,2)
        ax1.plot(di,-zi)
        ax1.axhline(y=-zpyc,color="red",label="Pycnocline")
        ax1.axhline(y=-hub,color="blue",label="HUB")
        ax2.plot(ti,-zi)
        ax1.legend()
        plt.title(str(round(lat,1))+" , "+str(round(lon,1)))
        plt.show()
    H2 = -(zpyc)+(hub)
        
    if ~(np.isnan(di).all()):
        return H2#*gprime
    else:
        return np.nan


def heat_content(heat_function,depth,plusminus):
    depth = np.abs(depth)
    xnew= np.arange(max(5,depth-plusminus),depth)
    #xnew= np.arange(max(0,depth-plusminus),min(depth+plusminus,1500))
    ynew = heat_function[0](xnew)
    ynew = ynew - gsw.CT_freezing(heat_function[1](xnew),xnew,0)
    if len(ynew)>0:
        return np.trapz(ynew,xnew)/len(xnew)
        #return np.max(ynew)
    else:
        return np.nan



def saltdiff(heat_function,depth,plusminus):
    depth = np.abs(depth)
    return abs(heat_function[1](0) - heat_function[1](depth))

def thermoclineg_depth(heat_function,depth,isopyc=0):
    depth = np.abs(depth)
    zi = np.arange(0,1500)
    ti = heat_function[0](zi)
    si = heat_function[1](zi)
    di = gsw.rho(si,ti,750)-1000
    low =  np.nanmin(ti)
    high = np.nanmax(ti)
    closest = np.abs(ti-(high+low)/2)
    #closest = np.abs(ti-0)
    closesti = np.nanargmin(closest)
    closest[zi<50]=np.inf
    isopycnaldepth = zi[closesti]
    #N = np.nanmean(N[:np.nanargmin(closest)])
    #return np.nansum((ti[closesti:-1]+1.8)*np.diff(di[closesti:])/np.diff(zi[closesti:]))

    if ~np.isnan(isopycnaldepth):
        return -(isopycnaldepth-depth)*(high-low)#np.trapz(ynew,xnew)/len(xnew)
        #return -(isopycnaldepth-depth)#np.trapz(ynew,xnew)/len(xnew)
    else:
        return np.nan

def isopycnal_depth(heat_function,depth,isopyc=0,shelfkey=None):
    depth = np.abs(depth)
    zi = np.arange(0,1000)
    ti = heat_function[0](zi)
    si = heat_function[1](zi)
    di = gsw.rho(si,ti,750)-1000
    dsigdz = np.diff(di,prepend=di[0])/np.diff(zi,prepend=zi[0]-1)

    pycz = np.trapz(dsigdz*zi,zi)/np.trapz(dsigdz,zi)
    closesti = np.nanargmin(np.abs(zi-pycz))
    #gprime = np.nanmedian(dsigdz[int(closesti*0.5):int(closesti*1.5)])
    gprime = np.nanmedian(di[closesti:])-np.nanmedian(di[:closesti])
    isopycnaldepth = zi[closesti]
    if shelfkey == "Filchner" or False:
        plt.plot(di,-zi)
        plt.axhline(y=-isopycnaldepth,color="red")
        plt.axhline(y=-depth,color="blue")
        plt.show()

    #low =  np.nanmin(ti)
    #high = np.nanmax(ti)
    if ~np.isnan(isopycnaldepth):
        return -(isopycnaldepth-depth)*gprime#np.trapz(ynew,xnew)/len(xnew)
        #return -(isopycnaldepth-depth)#np.trapz(ynew,xnew)/len(xnew)
    else:
        return np.nan


def therm_depth(heat_function,depth,therm=0):
    depth = np.abs(depth)
    xnew= np.arange(0,depth+500)
    ynew = heat_function[0](xnew)
    closest = np.abs(ynew-therm)
    closest[xnew<75]=np.inf
    thermdepth = xnew[np.argmin(closest)]
    #plt.plot(ynew,xnew)
    #plt.scatter(therm,xnew[np.argmin(closest)])
    #plt.show()

    if ~np.isnan(thermdepth):
        return (thermdepth-depth)#np.trapz(ynew,xnew)/len(xnew)
    else:
        return np.nan

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


def extract_adusumilli(fname):
    dfs = pd.read_csv(fname)
    rignot_shelf_my = {}
    sigma = {}
    areas = {}
    for l in range(len(dfs["Ice Shelf"])):
        if  dfs["Ice Shelf"][l] and dfs["Basal melt rate, 1994–2018 (m/yr)"][l]:
            shelves = dfs["Ice Shelf"][l].split("\n")
            melts = dfs["Basal melt rate, 1994–2018 (m/yr)"][l].split("\n")
            areasline =  dfs["Area (km 2)"][l].split("\n")
            for i in range(len(shelves)):
                mystr = melts[i]
                shelfname = shelves[i]
                if shelfname[-1] == " ":
                    shelfname = shelves[i][:-1]
                shelfname = shelfname.replace(" ","_")
                my = float(mystr.split("±")[0])
                sig = float(mystr.split("±")[1])
                rignot_shelf_my[shelfname] = my
                areas[shelfname] = float(areasline[i])
                sigma[shelfname] = sig
            
    return rignot_shelf_my,areas,sigma

def closest_point_for_graphing(grid,bedvalues,icemask,bordermask,baths,l):
    bath = baths[l]
    if np.isnan(bath):
        bath = bedvalues[grid[l][0],grid[l][1]]
    search = np.full_like(bedvalues,0)
    search[:]=0
    search[grid[l][0],grid[l][1]]=1
    route = np.logical_and(bedvalues<(min(bath+20,0)),icemask!=0)
    searchsum=0
    if np.sum(np.logical_and(route,bordermask))>0:
        intersection = [[],[]]
        while len(intersection[0])==0:
            search = bd(search,mask=route,iterations=200)
            searchsumnew = np.sum(search)
            if searchsum !=searchsumnew:
                searchsum = searchsumnew
            else:
                return (np.nan,np.nan)
            intersection = np.where(np.logical_and(search,bordermask))
        return (intersection[0][0],intersection[1][0])
    else:
        return (np.nan,np.nan)



def closest_point_pfun(grid,bedmach,baths,l):
    bedvalues = bedmach.bed.values
    icemask = bedmach.icemask_grounded_and_shelves.values
    closest_points=[np.nan]*len(baths)
    depthmask = bedvalues>-2300
    insidedepthmask = bedvalues<-1900
    bordermask = np.logical_and(insidedepthmask,depthmask)
    bordermask = np.logical_and(bordermask,np.isnan(icemask))
    #FRmask = [3000:6000,3000:7000]
    itertrack = []

    bath = baths[l]
    if np.isnan(bath):
        bath = bedvalues[grid[l][0],grid[l][1]]
    search = np.full_like(bedvalues,0)
    search[:]=0
    search[grid[l][0],grid[l][1]]=1
    route = np.logical_and(bedvalues<(min(bath+20,0)),icemask!=0)
    searchsum=0
    searchcopy = np.full_like(route,np.nan,dtype=float)
    searchcopy[route]=4
    searchcopy[bordermask]=10

    if np.sum(np.logical_and(route,bordermask))>0:
        intersection = [[],[]]
        while len(intersection[0])==0:
            searchcopy[search==1]=1
            plt.xlim(3000,7000)
            plt.ylim(7000,3000)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.gca().set_xticks([], minor=True)
            plt.gca().set_yticks([], minor=True)
            plt.gcf().set_size_inches(7, 7)

            search = bd(search,mask=route,iterations=400)

            searchsumnew = np.sum(search)
            if searchsum !=searchsumnew:
                searchsum = searchsumnew
            else:
                return (np.nan,np.nan)
            intersection = np.where(np.logical_and(search,bordermask))

        searchcopy[search==1]=1
        return (intersection[0][0],intersection[1][0])
    else:
        return (np.nan,np.nan)


def closest_WOA_points_bfs(grid,baths,bedmach,debug=False):
    print("bfs")
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

def closestHydro(bedmap,grid,physical,closest_points,sal,temp,shelves):
    print("temp from closest point")
    indexes=[np.nan]*len(physical)
    stx = sal.coords["x"].values
    sty = sal.coords["y"].values
    projection = pyproj.Proj("epsg:3031")
    salvals,tempvals = sal.s_an.values[0,:,:,:],temp.t_an.values[0,:,:,:]
    d  = sal.depth.values
    lines = []
    bedvalues = bedmap.bed.values
    mask = np.zeros(salvals.shape[1:])

    mask[:]=np.inf
    for l in range(salvals.shape[1]):
        for k in range(salvals.shape[2]):
            if np.sum(~np.isnan(salvals[:,l,k]))>0 and np.max(d[~np.isnan(salvals[:,l,k])])>1500:
                mask[l,k] = 1
    for l in tqdm(range(len(closest_points))[::-1]):
        if ~np.isnan(closest_points[l]).any():
            centroid = [bedmap.coords["x"].values[closest_points[l][1]],bedmap.coords["y"].values[closest_points[l][0]]]
            rdist = np.sqrt((sal.coords["x"]- centroid[0])**2 + (sal.coords["y"] - centroid[1])**2)
            rdist = rdist*mask
            indexes[l] = int(rdist.argmin())
    return indexes

def averageForShelf(soi,bedmap,grid,physical,baths,closest_hydro,sal,temp,shelves,debug=False,quant="glibheat",shelfkeys=None,timestep=0):
    reshapeval = sal.coords["x"].shape
    stx = sal.coords["x"].values
    sty = sal.coords["y"].values
    projection = pyproj.Proj("epsg:3031")
    salvals,tempvals = sal.s_an.values,temp.t_an.values
    d  = sal.depth.values
    lines = []
    bedvalues = bedmap.bed.values
    bedynamic = {}
    avg_t = []
    avg_s = []
    for l in tqdm(range(len(closest_hydro))):
        if soi ==shelves[l]:
            val = closest_hydro[l]
            if ~np.isnan(val):
                closest=np.unravel_index(int(val), sal.coords["x"].shape)
                x = stx[closest[0],closest[1]]
                y = sty[closest[0],closest[1]]
                lon,lat = projection(x,y,inverse=True)
                for timestep in range(salvals.shape[0]):
                    t = tempvals[timestep,:,closest[0],closest[1]]
                    s = salvals[timestep,:,closest[0],closest[1]]
                    s = gsw.SA_from_SP(s,d,lon,lat)
                    #FOR MIMOC MAKE PT
                    #t = gsw.CT_from_pt(s,t)
                    t = gsw.CT_from_t(s,t,d)
                    avg_t.append(t)
                    avg_s.append(s)
    avg_t = np.asarray(avg_t)
    avg_s = np.asarray(avg_s)
    print(np.shape(avg_t))
    print(np.shape(avg_s))

    avg_t = np.mean(avg_t,axis=0)
    avg_s = np.mean(avg_s,axis=0)
    return avg_s,avg_t,d
        
def revampedClosest(bedmap,grid,physical,baths,closest_hydro,sal,temp,shelves,debug=False,quant="glibheat",shelfkeys=None,timestep=0):
    heats=np.empty((sal.s_an.shape[0],len(physical)))
    cdws=np.empty((sal.s_an.shape[0],len(physical)))
    gprimes=np.empty((sal.s_an.shape[0],len(physical)))
    heats[:]=np.nan
    cdws[:]=np.nan
    gprimes[:]=np.nan
    reshapeval = sal.coords["x"].shape
    stx = sal.coords["x"].values
    sty = sal.coords["y"].values
    projection = pyproj.Proj("epsg:3031")
    salvals,tempvals = sal.s_an.values,temp.t_an.values
    d  = sal.depth.values
    lines = []
    bedvalues = bedmap.bed.values
    bedynamic = {}
 
    for l in tqdm(range(len(closest_hydro))):
        val = closest_hydro[l]
        if ~np.isnan(val):
            closest=np.unravel_index(int(val), sal.coords["x"].shape)
            x = stx[closest[0],closest[1]]
            y = sty[closest[0],closest[1]]
            lon,lat = projection(x,y,inverse=True)
            for timestep in range(salvals.shape[0]):
                if (val,timestep,baths[l]) in bedynamic:
                    heats[timestep,l]=heats[timestep,bedynamic[(val,timestep,baths[l])]]
                    cdws[timestep,l]=cdws[timestep,bedynamic[(val,timestep,baths[l])]]
                    gprimes[timestep,l]=gprimes[timestep,bedynamic[(val,timestep,baths[l])]]
                else:
                    bedynamic[(val,timestep,baths[l])] = l
                    t = tempvals[timestep,:,closest[0],closest[1]]
                    s = salvals[timestep,:,closest[0],closest[1]]
                    s = gsw.SA_from_SP(s,d,lon,lat)
                    #FOR MIMOC MAKE PT
                    #t = gsw.CT_from_pt(s,t)
                    t = gsw.CT_from_t(s,t,d)

                    tinterp,sinterp = interpolate.interp1d(d,np.asarray(t)),interpolate.interp1d(d,np.asarray(s))
                    if np.isnan(t[11:]).all():
                        heats[timestep,l]=np.nan#
                    elif np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                        heats[timestep,l]=heat_content((tinterp,sinterp),baths[l],100)
                        cdws[timestep,l]=pycnocline((tinterp,sinterp),baths[l],shelf_key=shelves[l],lat=lat,lon=lon)
                        gprimes[timestep,l]=gprime((tinterp,sinterp),baths[l],shelf_key=shelves[l],lat=lat,lon=lon)
    return heats,cdws,gprimes



def tempFromClosestPoint(bedmap,grid,physical,baths,closest_points,sal,temp,shelves,debug=False,quant="glibheat",shelfkeys=None):
    print("temp from closest point")
    heats=[np.nan]*len(baths)
    stx = sal.coords["x"].values
    sty = sal.coords["y"].values
    projection = pyproj.Proj("epsg:3031")
    salvals,tempvals = sal.s_an.values[0,:,:,:],temp.t_an.values[0,:,:,:]
    d  = sal.depth.values
    lines = []
    bedvalues = bedmap.bed.values
    mask = np.zeros(salvals.shape[1:])

    mask[:]=np.inf
    for l in range(salvals.shape[1]):
        for k in range(salvals.shape[2]):
            if np.sum(~np.isnan(salvals[:,l,k]))>0 and np.max(d[~np.isnan(salvals[:,l,k])])>1500:
                mask[l,k] = 1
    for l in tqdm(range(len(closest_points))[::-1]):
        if ~np.isnan(closest_points[l]).any():
            centroid = [bedmap.coords["x"].values[closest_points[l][1]],bedmap.coords["y"].values[closest_points[l][0]]]
            rdist = np.sqrt((sal.coords["x"]- centroid[0])**2 + (sal.coords["y"] - centroid[1])**2)
            rdist = rdist*mask
            closest=np.unravel_index(rdist.argmin(), rdist.shape)
            x = stx[closest[0],closest[1]]
            y = sty[closest[0],closest[1]]
            t = tempvals[:,closest[0],closest[1]]
            s = salvals[:,closest[0],closest[1]]
            lon,lat = projection(x,y,inverse=True)
            s = gsw.SA_from_SP(s,d,lon,lat)
            #FOR MIMOC MAKE PT
            #t = gsw.CT_from_pt(s,t)
            t = gsw.CT_from_t(s,t,d)
            line = ([physical[l][0],physical[l][1]],[centroid[0],centroid[1]])
            dist = np.sqrt((physical[l][1]-x)**2 + (physical[l][0]-y)**2)
            #if shelves[l]=="Getz":
                #plt.plot(t,-d)
                #plt.show()

            if ~(np.isnan(line).any()):
                lines.append(line)
            tinterp,sinterp = interpolate.interp1d(d,np.asarray(t)),interpolate.interp1d(d,np.asarray(s))
            if np.isnan(t[11:]).all():
                heats[l]=np.nan#
            elif quant=="glibheat" and np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                heats[l]=heat_content((tinterp,sinterp),baths[l],100)
            elif quant=="dsalt" and np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                heats[l]=saltdiff((tinterp,sinterp),baths[l],100)
            elif quant=="thermdepth" and np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                heats[l]=therm_depth((tinterp,sinterp),baths[l],0.5)
            elif quant=="cdwdepth" and np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                heats[l]=pycnocline((tinterp,sinterp),baths[l],shelf_key=shelves[l],lat=lat,lon=lon)
            elif quant=="gprime" and np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                heats[l]=gprime((tinterp,sinterp),baths[l],shelf_key=shelves[l],lat=lat,lon=lon)
            elif quant=="newtemp" and np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                heats[l]=newtemp((tinterp,sinterp),baths[l],shelf_key=shelves[l])
            elif quant=="isopycnaldepth" and np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                if shelfkeys:
                    heats[l]=isopycnal_depth((tinterp,sinterp),baths[l],shelfkey=shelves[l])
                else:
                    heats[l]=isopycnal_depth((tinterp,sinterp),baths[l])
            elif quant=="thermocline" and np.nanmax(d[~np.isnan(t)])>abs(baths[l]):
                heats[l]=thermoclineg_depth((tinterp,sinterp),baths[l])
            elif quant=="distance":
                heats[l]=dist
    if debug:
        fig, ax = plt.subplots()
        lc = mc.LineCollection(lines, colors="red", linewidths=2)
        ax.imshow(bedvalues)
        plt.xlim(-10**7,10**7)
        plt.ylim(-10**7,10**7)
        ax.add_collection(lc)
        plt.show()
    return heats


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
 
    print("temp from closest point")
    slopes=[np.nan]*len(depths)
    projection = pyproj.Proj("epsg:3031")

    bedvalues = bedmap.bed.values
    icefrontT = np.asarray(icefront).T
    if mode=="closest":
        for l in tqdm(range(len(closest_points))[::-1]):
            if ~np.isnan(closest_points[l]).any():
                centroid = [bedmap.coords["x"].values[closest_points[l][1]],bedmap.coords["y"].values[closest_points[l][0]]]
                rdist = np.sqrt((icefrontT[0]- centroid[0])**2 + (icefrontT[1] - centroid[1])**2)
                dist= np.nanmin(rdist)
                slopes[l] = (depths[l]/dist)
    elif mode == "euclid":
        for l in tqdm(range(len(physical))):
            if ~np.isnan(closest_points[l]).any():
                centroid = [physical[l][0],physical[l][1]]
                rdist = np.sqrt((icefrontT[0]- centroid[0])**2 + (icefrontT[1] - centroid[1])**2)
                dist= np.nanmin(rdist)
                #dist = np.nanmean(rdist[shelf_keys_edge==shelf_keys[l]])
                if dist !=0:
                    slopes[l] = abs(depths[l]/dist)
                else:
                    slopes[l] = np.nan 

    return slopes

def slope_by_shelf(bedmach,polygons):
    GLIBmach = bedmach.thickness.copy(deep=True)
    print(bedmach)
    GLIBmach.values[:] = gaussian_filter(bedmach.surface.values[:]-bedmach.thickness.values[:],10)
    GLIBmach.values[np.logical_or(bedmach.icemask_grounded_and_shelves==0,np.isnan(bedmach.icemask_grounded_and_shelves))]=np.nan
    GLIBmach = GLIBmach.rio.write_crs("epsg:3031")
    print(GLIBmach)
    del GLIBmach.attrs['grid_mapping']
    GLIBmach.rio.to_raster("data/glibmach.tif")

    glib_by_shelf = {}
    full_info = {}
    for k in tqdm(polygons.keys()):
        raster = riox.open_rasterio('data/glibmach.tif')
        gons = []
        parts = polygons[k][1]
        polygon = polygons[k][0]
        if len(parts)>1:
            parts.append(-1)
            for l in range(0,len(parts)-1):
                poly_path=shapely.geometry.Polygon(np.asarray(polygon.exterior.coords.xy)[:,parts[l]:parts[l+1]].T)#.buffer(10**4)
                gons.append(poly_path)
        else:
            gons = [polygon]
        print(k)
        print("made it hearE")
        clipped = raster.rio.clip(gons)[0]
        clipped = np.asarray(clipped)
        clipped[clipped<-9000] = np.nan
        grad = np.gradient(clipped)
        #grad[0] = grad[0]
        #grad[1] = grad[1]
        #grad = np.sqrt(np.nanmean(grad[0])**2 + np.nanmean(grad[1])**2)

        grad[0] = (grad[0]*(1/500.0))**2
        grad[1] = (grad[1]*(1/500.0))**2
        grad = np.sqrt(np.sum(grad,axis=0))
        if k == "Amery" and False:
            plt.imshow(grad,vmin=0,vmax=10)
            plt.colorbar()
            plt.show()
        #glib_by_shelf[k] = np.nanmedian(grad[grad<np.nanquantile(grad,0.75)])
        glib_by_shelf[k] = np.nanmedian(grad)
        full_info[k] = copy(clipped)

    savemat("full_info.mat",full_info)
    plt.show()

    return glib_by_shelf


