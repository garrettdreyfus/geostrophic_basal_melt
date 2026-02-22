from bathtub import closest_shelf, closest_shelves
from metpy.calc import lat_lon_grid_deltas
import struct
import pyproj
from xgrads import open_CtlDataset
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import ipdb

def Nakata(radius=30*1000):
    lons = []
    lats = []
    count = 0
    with open('data/pss06lats_v3.dat', 'rb') as fileobj:
        for chunk in tqdm(iter(lambda: fileobj.read(4), '')):
            if chunk:
                count+=1
                lats.append(struct.unpack('<i', chunk)[0]/100000)
            elif count>=1328*1264:
                break
    with open('data/pss06lons_v3.dat', 'rb') as fileobj:
        for chunk in tqdm(iter(lambda: fileobj.read(4), '')):
            if chunk:
                count+=1
                lons.append(struct.unpack('<i', chunk)[0]/100000)
            elif count>=1328*1264:
                break

    prod = np.empty((1328,1264))
    count = 0 
    for year in range(2003,2011):
        dset = open_CtlDataset('data/ice_production_ANT_{}.ctl'.format(year))
        prodarray = dset.where(dset.time<=dset.time[5],drop=True).where(dset["prod"]!=-999).sum(dim="time").sel(lev=0)
        prod = prod + prodarray["prod"].values
        count += 1

    prod[prod==0] = np.nan
    prod = prod/count
    lons = np.asarray(lons)
    lats = np.asarray(lats)
    

    with open("data/shelfpolygons.pickle","rb") as f:
        polygons = pickle.load(f)
    shelves = {}
    shelves_lat = {}
    shelves_lon = {}
    dists = {}

    lons = lons[~np.isnan(prod).flatten()]
    lats = lats[~np.isnan(prod).flatten()]
    # multiply by grid box size
    prvals = prod[~np.isnan(prod)].flatten()*((6.25*10**3)**2)

    projection = pyproj.Proj("epsg:3031")
    X,Y = projection(lons,lats)

    with open("data/bedmach.pickle","rb") as f:
        bedmach = pickle.load(f)

    outvals = []
    shelfnames = list(polygons.keys())
    for sname in shelfnames:
        shelves[sname]=[0]
        shelves_lat[sname]=[0]
        shelves_lon[sname]=[0]
    for coord in tqdm(range(len(lons))):
        frisflag = False
        rossflag = False
        val = prvals[coord]
        if val!=0 and float(bedmach.bed.sel(x=X[coord],y=Y[coord],method="nearest"))>-4000:
            names = closest_shelves((X[coord],Y[coord]),polygons,radius)
            for name in names:
                shelves_lat[name].append(X[coord])
                shelves_lon[name].append(Y[coord])
                shelves[name].append(val)
                # only closest cavity
                break
    return shelves,dists

shelves,dists = Nakata()

skeys = list(shelves.keys())
svals = list(shelves.values())

dkeys = list(dists.keys())
dvals = list(dists.values())

newsvals = []
newdvals = []
for i in  svals: newsvals.append(list(map(lambda x: x,i)))
for i in  dvals: newdvals.append(list(map(lambda x: x,i)))
newnewsvals = []
newnewsvals_weighted = []
thresh=30
for i in range(len(newsvals)):
    newnewsvals.append(np.sum((np.asarray(newsvals[i]))))

sortedp = np.argsort(newnewsvals)
sortednames = np.asarray(skeys)[sortedp]
sortedvals = np.asarray(newnewsvals)[sortedp]
rhoi = 910
final_product = dict(zip(sortednames, sortedvals))

with open("data/polyna_by_shelf_2024_nakata.pickle","wb") as f:
    pickle.dump((final_product,[]),f)

