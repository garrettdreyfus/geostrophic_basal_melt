import shapefile
import rockhound as rh
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import itertools, pickle
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import time,gsw,xarray, pyproj
from bathtub import closest_shelf
from scipy import interpolate
import pandas as pd
from copy import copy


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
    return mask, average_t, average_s,d

shelf_profiles = {}
for shelfname in polygons.keys():
    shelf = polygons[shelfname]
    _,average_t, average_s,d = shelf_average_profile(shelf,sal,temp,d)
    shelf_profiles[shelfname] = (average_t,average_s,d)

shelf_profile_heat_functions = {}
for k in shelf_profiles.keys():
    t,s,d = shelf_profiles[k] 
    shelf_profile_heat_functions[k] = interpolate.interp1d(d,(np.asarray(t)+273.15)*4184)

with open("data/GLBsearchresults.pickle","rb") as f:
    physical,grid,baths,bathtubs,bathtub_depths = pickle.load(f)

def heat_content(heat_function,depth,plusminus):
    #heat = gsw.cp_t_exact(s,t,d)
    xnew= np.arange(max(0,depth-plusminus),min(depth+plusminus,max(d)))
    #print(xnew,depth,max(d))
    ynew = heat_function(xnew)
    return np.trapz(ynew,xnew)

shelf_heat_content = []
shelf_heat_content_byshelf={}
for k in polygons.keys():
    shelf_heat_content_byshelf[k] = []
bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
bed = bedmap.bed.values

# newbaths = copy(baths)
# for l in range(len(baths)):
#     baths[l]=bed[grid[l][1],grid[l][0]]
#     #if baths[l]>=0:
#         #baths[l]=bed[grid[l][1],grid[l][0]]

# for l in tqdm(range(len(baths))):
#     coord = physical[l]
#     shelfname, _,_ = closest_shelf(coord,polygons)
#     shelf_heat_content.append(heat_content(shelf_profile_heat_functions[shelfname],-baths[l],5))
#     shelf_heat_content_byshelf[shelfname].append(shelf_heat_content[-1])

# with open("data/shc_noGLIB.pickle","wb") as f:
#    pickle.dump(shelf_heat_content_byshelf,f)

with open("data/shc_noGLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_noGLIB = pickle.load(f)

with open("data/shc_GLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_GLIB = pickle.load(f)

dfs = pd.read_excel("data/rignot2019.xlsx",sheet_name=None)
dfs = dfs['Dataset_S1_PNAS_2018']
print(dfs.keys())
rignot_shelf_massloss={}
for l in range(len(dfs["Glacier name"])):
    if  dfs["Glacier name"][l] in shelf_heat_content_byshelf.keys():
        rignot_shelf_massloss[dfs["Glacier name"][l]] = dfs["Cumul Balance"][l]

fig,(ax1,ax2) = plt.subplots(1,2)
shc= []
smb = []
c = []
for k in rignot_shelf_massloss.keys():
    if len(shelf_heat_content_byshelf_noGLIB[k]):
        shc.append(np.log10(np.nanmedian(shelf_heat_content_byshelf_noGLIB[k])))
        smb.append(rignot_shelf_massloss[k])
        c.append(len(shelf_heat_content_byshelf_noGLIB[k]))
        #ax1.text(shc,smb,k)
ax1.scatter(shc,smb)

for k in rignot_shelf_massloss.keys():
    if len(shelf_heat_content_byshelf_noGLIB[k]):
        shc = np.log10(np.nanmedian(shelf_heat_content_byshelf_GLIB[k]))
        smb = rignot_shelf_massloss[k]
        ax2.scatter(shc,smb)
 
plt.show()
# pc = bedmap.icemask_grounded_and_shelves.plot.pcolormesh(
#   ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30)
# )
physical = np.asarray(physical).T
plt.scatter(physical[0],physical[1],c=shelf_heat_content,vmin=4.4*10**7,vmax=4.5*10**7,cmap="jet")
plt.colorbar()

plt.show()




