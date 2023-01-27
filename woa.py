import xarray
import xarray as xr
import pyproj
import numpy as np
import rockhound as rh
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

def create_WOA(bed,debug = False):
   bedmap = bed
   salfname,tempfname = "data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc"
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
   if debug:
      bedvalues = sal.bed.values
      bedvalues[np.isnan(sal.s_an.values[0,0,:,:])]=np.nan
      plt.scatter(sal.x,sal.y,c=bedvalues)
      plt.show()
   temp["bed"] = bedmap.bed.interp(x=temp.x,y=temp.y)
   sal["icemask"] = bedmap.icemask_grounded_and_shelves.interp(x=sal.x,y=sal.y)
   temp["icemask"] = bedmap.icemask_grounded_and_shelves.interp(x=sal.x,y=sal.y)
   sal.icemask.values[sal.icemask.values<1]=0
   temp.icemask.values[temp.icemask.values<1]=0
   return sal,temp


sal = xr.open_dataset("data/woa18_decav81B0_s00_04.nc",decode_times=False)
sal = sal.sel(depth=2000,drop=True)
sal = sal.isel(time=0,drop=True)
print(sal)
