import xarray
import pyproj
import numpy as np
import rockhound as rh
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
salfname,tempfname = "data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc",
sal = xarray.open_dataset(salfname,decode_times=False)
temp = xarray.open_dataset(tempfname,decode_times=False)
sal = sal.where(sal.lat<-60,drop=True)
temp= temp.where(sal.lat<-60,drop=True)
d = sal.depth.values
lons=sal.lon.values
lats=sal.lat.values
projection = pyproj.Proj("epsg:3031")
lons,lats = np.meshgrid(sal.lon,sal.lat)
x,y = projection.transform(lons,lats)
sal.coords["x"]= (("lat","lon"),x)
sal.coords["y"]= (("lat","lon"),y)
temp.coords["x"]= (("lat","lon"),x)
temp.coords["y"]= (("lat","lon"),y)
xvals,yvals = np.meshgrid(bedmap.x,bedmap.y)
sal["bed"] = bedmap.bed.interp(x=sal.x,y=sal.y)
temp["bed"] = bedmap.bed.interp(x=temp.x,y=temp.y)

with open("data/woawithbed.pickle","wb") as f:
   pickle.dump([sal,temp],f)