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

def generate_polynaset2024(radius=100*1000,dsnum = 1):
    if dsnum == 1:
        df = pd.read_csv("data/Nihashi_2024_AMSRE_2003-2010_icepro 1.txt",sep = "    ",header= None)
    else:
        df = pd.read_csv("data/Nihashi_2024_AMSR2_2013-2021_icepro 3.txt",sep = "    ",header= None)
    ipdb.set_trace()
    with open("data/shelfpolygons.pickle","rb") as f:
        polygons = pickle.load(f)
    shelves = {}
    shelves_lat = {}
    shelves_lon = {}
    dists = {}
    lons = df[0]
    lats = df[1]
    projection = pyproj.Proj("epsg:3031")
    X,Y = projection(lons,lats)

    with open("data/bedmach.pickle","rb") as f:
        bedmach = pickle.load(f)



    # As = []
    # for coord in tqdm(range(len(X))):
    #     dists = np.sqrt((X[coord]-X)**2 + (Y[coord]-Y)**2)
    #     sort_i = np.argsort(dists)
    #     A = dists[sort_i[2]]*dists[sort_i[1]]
    #     As.append(A)
    # ipdb.set_trace()



    prvals = df[2]*(6.25**2)*(10**6)
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
        if val!=0 and float(bedmach.bed.sel(x=X[coord],y=Y[coord],method="nearest"))>-2000:
            names = closest_shelves((X[coord],Y[coord]),polygons,radius)
            # nameold,_,_ = closest_shelf((X[coord],Y[coord]),polygons,np.inf)
            for name in names:
                shelves_lat[name].append(X[coord])
                shelves_lon[name].append(Y[coord])
                shelves[name].append(val)
                break
                # if ("Filchner" == name or "Ronne" == name):
                #     if not frisflag:
                #         shelves[name].append(val)
                #         frisflag = True
                # elif "Ross" in name:
                #     if not rossflag:
                #         shelves[name].append(val)
                #         frisflag = True
                # else:
                #     shelves[name].append(val)
    ipdb.set_trace()
    return shelves,dists

# shelves, dists = generate_polynaset2024(40*1000,dsnum=2)

# with open("data/newpolynainfo_2024.pickle","wb") as f:
#     pickle.dump((shelves,dists),f)

# skeys = list(shelves.keys())
# svals = list(shelves.values())

# dkeys = list(dists.keys())
# dvals = list(dists.values())

# newsvals = []
# newdvals = []
# for i in  svals: newsvals.append(list(map(lambda x: x,i)))
# for i in  dvals: newdvals.append(list(map(lambda x: x,i)))
# newnewsvals = []
# newnewsvals_weighted = []
# thresh=30
# for i in range(len(newsvals)):
#     newnewsvals.append(np.sum((np.asarray(newsvals[i]))))
#     # newnewsvals_weighted.append(np.sum((np.asarray(newsvals[i])/(np.asarray(newdvals[i])+1)**2)[np.asarray(newdvals[i])<thresh*1000]))

# sortedp = np.argsort(newnewsvals)
# sortednames = np.asarray(skeys)[sortedp]
# sortedvals = np.asarray(newnewsvals)[sortedp]
# # sortedvals_weighted = np.asarray(newnewsvals_weighted)[sortedp]
# rhoi = 910
# final_product = dict(zip(sortednames, sortedvals))
# # final_product_weighted = dict(zip(sortednames, sortedvals_weighted))

# with open("data/polyna_by_shelf_2024_ds2.pickle","wb") as f:
#     pickle.dump((final_product,[]),f)


def Komatsu(radius=30*1000):
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
    # plt.pcolormesh(dset.lon,dset.lat,prod,cmap="Reds")
    # plt.colorbar()
    # plt.show()
    

    with open("data/shelfpolygons.pickle","rb") as f:
        polygons = pickle.load(f)
    shelves = {}
    shelves_lat = {}
    shelves_lon = {}
    dists = {}

    lons = lons[~np.isnan(prod).flatten()]
    lats = lats[~np.isnan(prod).flatten()]
    prvals = prod[~np.isnan(prod)].flatten()*((6.25*10**3)**2)

    projection = pyproj.Proj("epsg:3031")
    X,Y = projection(lons,lats)

    with open("data/bedmach.pickle","rb") as f:
        bedmach = pickle.load(f)



    # As = []
    # for coord in tqdm(range(len(X))):
    #     dists = np.sqrt((X[coord]-X)**2 + (Y[coord]-Y)**2)
    #     sort_i = np.argsort(dists)
    #     A = dists[sort_i[2]]*dists[sort_i[1]]
    #     As.append(A)
    # ipdb.set_trace()



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
            # nameold,_,_ = closest_shelf((X[coord],Y[coord]),polygons,np.inf)
            for name in names:
                shelves_lat[name].append(X[coord])
                shelves_lon[name].append(Y[coord])
                shelves[name].append(val)
                break
                # if ("Filchner" == name or "Ronne" == name):
                #     if not frisflag:
                #         shelves[name].append(val)
                #         frisflag = True
                # elif "Ross" in name:
                #     if not rossflag:
                #         shelves[name].append(val)
                #         frisflag = True
                # else:
                #     shelves[name].append(val)
    return shelves,dists

shelves,dists = Komatsu()

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
final_product = dict(zip(sortednames, sortedvals))
# final_product_weighted = dict(zip(sortednames, sortedvals_weighted))

with open("data/polyna_by_shelf_2024_komatsu.pickle","wb") as f:
    pickle.dump((final_product,[]),f)

