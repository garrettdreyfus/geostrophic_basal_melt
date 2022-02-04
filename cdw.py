import shapefile
import rockhound as rh
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import itertools, pickle
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import time
import xarray, pyproj


with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

sal = xarray.open_dataset("data/woa18_decav81B0_s00_04.nc",decode_times=False)
temp = xarray.open_dataset("data/woa18_decav81B0_t00_04.nc",decode_times=False)
sal = sal.where(sal.lat<-60,drop=True)
temp= temp.where(sal.lat<-60,drop=True)

plt.figure(figsize=(16, 14))
ax = plt.subplot(111)
sal['pycnocline']  = sal.s_an.copy()
sal.pycnocline[:] = np.nan

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


def create_depth_mask(da,d,limit):
    depthmask = np.full_like(da[0,0,:,:].values,np.nan,dtype=bool)
    for i in range(depthmask.shape[0]):
        for j in range(depthmask.shape[1]):
            if ~(np.isnan(da.values[0,:,i,j])).all():
                depthmask[i,j] = (d[~np.isnan(da.values[0,:,i,j])][-1]>limit)
    return depthmask

depthmask = create_depth_mask(sal.s_an,sal.depth.values,1000)

sal.s_an[0,0,:,:].values[~depthmask]=np.nan

def shelf_average_profile(shelf,sal,temp,d):
    centroid = list(shelf.centroid.coords)[0]
    mask = np.full_like(sal.s_an[0,0,:,:].values,np.nan,dtype=bool)
    dist = np.sqrt((sal.coords["x"]- centroid[0])**2 + (sal.coords["y"] - centroid[1])**2)
    radius=1000*10**3
    mask[dist<radius] = True
    mask[dist>radius] = False
    average_s = []
    average_t = []
    for i in range(len(d)):
        if np.sum(~np.isnan(sal.s_an.values[0,i][mask])) < 10:
            average_s.append(np.nan)
            average_t.append(np.nan)
        else:
            average_s.append(np.nanmean(sal.s_an.values[0,i][mask]))
            average_t.append(np.nanmean(temp.t_an.values[0,i][mask]))
    return mask, average_s, average_t

for shelfname in polygons.keys():
    shelf = polygons[shelfname]
    _,_, average_t = shelf_average_profile(shelf,sal,temp,d)
    plt.plot(average_t,-d)
plt.show()

#np.nan
sal.s_an[0,0,:,:].values[mask]=0
sal.s_an[0,0,:,:].plot.pcolormesh(
    ax=ax, cmap="jet", cbar_kwargs=dict(pad=0.01, aspect=30),\
    x="x",y="y")
plt.show()
