import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import shapely
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
from matplotlib.patches import Rectangle
import gsw,xarray, pyproj
from bathtub import closest_shelf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from copy import copy
from matplotlib import collections  as mc
from functools import partial
from tqdm.contrib.concurrent import process_map
from scipy import ndimage
from scipy.ndimage import binary_dilation as bd, label, gaussian_filter 
from scipy.io import savemat
import rioxarray as riox
import rasterio

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def gprime(si,ti,zi,hub,shelf_key=None,lat=None,lon=None,debug=False):
    hub = np.abs(hub)
    di = gsw.rho(si,ti,zi)

    di = moving_average(di,50)
    si = moving_average(si,50)
    ti = moving_average(ti,50)
    zi = moving_average(zi,50)

    dizi = np.abs(np.diff(di)/np.diff(zi))
    thresh = np.quantile(dizi,0.85)
    zpyc = np.mean(zi[1:][dizi>thresh])
    zpyci = np.argmin(np.abs(zi-zpyc))

    di = gsw.rho(si,ti,zpyc)

    rho_1i = np.logical_and(zi<zi[zpyci],zi>zi[zpyci]-50)
    rho_2i = np.logical_and(zi<zi[zpyci]+50,zi>zi[zpyci])

    gprime_ext = 9.8*(np.mean(di[rho_1i])-np.mean(di[rho_2i]))/np.mean(di[np.logical_or(rho_1i,rho_2i)])

    if ~(np.isnan(di).all()):
        return -(zpyc)+hub,abs(gprime_ext)
    else:
        return np.nan,np.nan

def pycnocline(heat_function,hub,shelf_key=None,lat=None,lon=None,debug=False):
    hub = np.abs(hub)
    zi = np.arange(5,1500,1)
    ti = moving_average(heat_function[0](zi),50)
    si = moving_average(heat_function[1](zi),50)
    zi = moving_average(zi,50)

    di = gsw.rho(si,ti,zi)
    dizi = np.abs(np.diff(di)/np.diff(zi))
    thresh = np.quantile(dizi,0.85)
    zpyc = np.mean(zi[1:][dizi>thresh])
    zpyci = np.argmin(np.abs(zi-zpyc))
    deltaH = -(zpyc)+(hub)

    if debug:
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

        
    if ~(np.isnan(di).all()):
        return deltaH
    else:
        return np.nan

def heat_content(si,ti,zi,depth,plusminus):
    depth = np.abs(depth)
    xnew= np.arange(max(5,depth-plusminus),depth)
    shallowbound = max(5,depth-plusminus)
    deepbound = depth
    mask = np.logical_and(zi>shallowbound,zi<depth)
    depthrange = zi[mask]
    #xnew= np.arange(max(0,depth-plusminus),min(depth+plusminus,1500))
    ynew = ti[mask]
    ynew = ynew - gsw.CT_freezing(si[mask],depthrange,0)
    if len(ynew)>0:
        return np.trapz(ynew,depthrange)/len(depthrange)
        #return np.max(ynew)
    else:
        return np.nan

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


def closest_shelfbreak_points_bfs(grid,baths,bedmach,debug=False):
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
    lons = []
    lats = []
    for l in tqdm(range(len(closest_hydro))):
        if soi == shelves[l]:
            val = closest_hydro[l]
            closest=np.unravel_index(int(val), sal.coords["x"].shape)
            if ~np.isnan(val) and not np.isnan(tempvals[:,:,closest[0],closest[1]]).all():
                x = stx[closest[0],closest[1]]
                y = sty[closest[0],closest[1]]
                lon,lat = projection(x,y,inverse=True)
                lons.append(lon)
                lats.append(lat)
                t = np.nanmean(tempvals[:,:,closest[0],closest[1]],axis=0)
                s = np.nanmean(salvals[:,:,closest[0],closest[1]],axis=0)
                s = gsw.SA_from_SP(s,d,lon,lat)
                #FOR MIMOC MAKE PT
                #t = gsw.CT_from_pt(s,t)
                t = gsw.CT_from_t(s,t,d)
                avg_t.append(t)
                avg_s.append(s)

    print(np.nanmean(lons),np.nanmean(lats))
    print(set(tuple(sorted([m, n])) for m, n in zip(lons, lats)))

    avg_t = np.asarray(avg_t)
    avg_s = np.asarray(avg_s)

    avg_t = np.mean(avg_t,axis=0)
    avg_s = np.mean(avg_s,axis=0)
    return avg_s,avg_t,d
        
def parameterization_quantities(bedmap,grid,physical,baths,closest_hydro,sal,temp,shelves,debug=False,quant="glibheat",shelfkeys=None,timestep=0):
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
 
    for val,bath in tqdm(closest_hydro.keys()):
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
                t = gsw.CT_from_pt(s,t)
                #t = gsw.CT_from_t(s,t,d)

                zi = np.arange(5,1500,1)
                ti = np.interp(zi,d,np.asarray(t))
                si = np.interp(zi,d,np.asarray(s))
                if np.nanmax(d[~np.isnan(t)])>abs(bath):
                    deltaH, gprime_ext = gprime(si,ti,zi,bath)
                    heats[timestep,closest_hydro[(val,bath)]]=heat_content(si,ti,zi,bath,100)
                    cdws[timestep,closest_hydro[(val,bath)]]= deltaH
                    gprimes[timestep,closest_hydro[(val,bath)]]=gprime_ext
    return heats,cdws,gprimes

def slope_by_shelf(bedmach,polygons):
    GLIBmach = bedmach.thickness.copy(deep=True)
    GLIBmach.values[:] = bedmach.surface.values[:]-bedmach.thickness.values[:]
    GLIBmach.values[np.logical_or(bedmach.icemask_grounded_and_shelves==0,np.isnan(bedmach.icemask_grounded_and_shelves))]=np.nan
    GLIBmach = GLIBmach.rio.write_crs("epsg:3031")
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

        clipped = raster.rio.clip(gons)[0]
        clipped = np.asarray(clipped)
        clipped[clipped<-9000] = np.nan

        label_im, nb_labels = label(~np.isnan(clipped))
        sizes = ndimage.sum(~np.isnan(clipped), label_im, range(nb_labels + 1))
        labels = np.asarray(range(nb_labels+1))
        clipped[label_im!=labels[np.argmax(sizes)]] = np.nan
        if np.sum(~np.isnan(clipped))<100:
            glib_by_shelf[k]=np.nan
        else:
            X,Y = np.meshgrid(range(np.shape(clipped)[1]),range(np.shape(clipped)[0]))
            X=X[~np.isnan(clipped)]
            Y=Y[~np.isnan(clipped)]
            xgrad,ygrad = np.gradient(clipped)
            flatclipped=clipped[~np.isnan(clipped)]
            A = np.vstack([X,Y, np.ones(len(X))]).T
            m1,m2, c = np.linalg.lstsq(A, flatclipped, rcond=None)[0]
            m1=np.abs(m1/500)
            m2=np.abs(m2/500)
            #glib_by_shelf[k] = np.sqrt(m1**2+m2**2)
            glib_by_shelf[k] = np.nanmean(np.sqrt((xgrad/500)**2+(ygrad/500)**2))

    return glib_by_shelf
