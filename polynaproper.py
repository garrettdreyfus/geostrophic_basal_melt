from bathtub import closest_shelf, closest_shelves
from metpy.calc import lat_lon_grid_deltas
import pyproj
from xgrads import open_CtlDataset
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import ipdb
 
def generate_polynaset2016():
    llset = open_CtlDataset('data/polynall.ctl')
    lons,lats = np.meshgrid(llset.lon,llset.lat)
    dx,dy = lat_lon_grid_deltas(lons,lats)

    with open("data/shelfpolygons.pickle","rb") as f:
        polygons = pickle.load(f)

    shelves = {}
    dists = {}
    projection = pyproj.Proj("epsg:3031")
    X,Y = projection(lons,lats)
    lonishape = llset.lon.shape[0]
    latishape = llset.lat.shape[0]
    prvals = llset.pr[0].values[:-1,:-1]*dx[:-1,:]*dy[:,:-1]
    for loni in tqdm(range(lonishape-1)):
        for lati in range(latishape-1):
            val = prvals[lati,loni]
            if val!=0:
                name,_,dist = closest_shelf((X[lati,loni],Y[lati,loni]),polygons,min_dist=np.inf)
                if name not in shelves.keys():
                    shelves[name] = []
                    dists[name] = []
                shelves[name].append(val)
                dists[name].append(dist)
            

    with open("data/newpolynainfo_100.pickle","wb") as f:
        pickle.dump((shelves,dists),f)

def generate_polynaset2024(radius=10*1000):
    # df = pd.read_csv("data/Nihashi_2024_AMSRE_2003-2010_icepro 1.txt",sep = "    ",header= None)
    df = pd.read_csv("data/Nihashi_2024_AMSR2_2013-2021_icepro 3.txt",sep = "    ",header= None)
    with open("data/shelfpolygons.pickle","rb") as f:
        polygons = pickle.load(f)
    shelves = {}
    dists = {}
    lons = df[0]
    lats = df[1]
    projection = pyproj.Proj("epsg:3031")
    X,Y = projection(lons,lats)
    prvals = df[2]*36*(10**6)
    ipdb.set_trace()
    outvals = []
    shelfnames = list(polygons.keys())
    for sname in shelfnames:
        shelves[sname]=[0]
    for coord in tqdm(range(len(lons))):
            val = prvals[coord]
            if val!=0:
                names = closest_shelves((X[coord],Y[coord]),polygons,radius)
                # nameold,_,_ = closest_shelf((X[coord],Y[coord]),polygons,np.inf)
                for name in names:
                    shelves[name].append(val)
    with open("data/newpolynainfo_2024.pickle","wb") as f:
        pickle.dump((shelves,dists),f)

generate_polynaset2024(40*1000)

with open("data/newpolynainfo_2024.pickle","rb") as f:
    shelves,dists = pickle.load(f)

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
    # newnewsvals_weighted.append(np.sum((np.asarray(newsvals[i])/(np.asarray(newdvals[i])+1)**2)[np.asarray(newdvals[i])<thresh*1000]))

sortedp = np.argsort(newnewsvals)
sortednames = np.asarray(skeys)[sortedp]
sortedvals = np.asarray(newnewsvals)[sortedp]
# sortedvals_weighted = np.asarray(newnewsvals_weighted)[sortedp]
rhoi = 910
gigatonconv = 10**(-12)
scalefactor = rhoi*gigatonconv
final_product = dict(zip(sortednames, sortedvals))
# final_product_weighted = dict(zip(sortednames, sortedvals_weighted))

with open("data/polyna_by_shelf_2024.pickle","wb") as f:
    pickle.dump((final_product,[]),f)

