import xarray
import xarray as xr
import pyproj
import numpy as np
import rockhound as rh
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import rioxarray as riox
import rasterio
from matplotlib.collections import PatchCollection
import shapely

def create_WOA(bed,debug = False):
   bedmap = bed
   salfname,tempfname = "data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc"
   #salfname,tempfname = "data/woa18_decav_s15_04.nc","data/woa18_decav_t15_04.nc"
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


def create_Pauthenet(bed,debug = True):
   bedmap = bed
   salfname,tempfname = "data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc"
   filename = "data/TS_Climato_Antarctic60S.nc"
   #salfname,tempfname = "data/woa18_decav_s15_04.nc","data/woa18_decav_t15_04.nc"
   dataset = xarray.open_dataset(filename,decode_times=False)
   dataset = dataset.rename({"Sal":"s_an","Temp":"t_an"})
   dataset["t_an"] = dataset.t_an.transpose("time","depth","lat","lon")
   dataset["s_an"] = dataset.s_an.transpose("time","depth","lat","lon")
   print(dataset)
   exit()
   sal = dataset
   temp = dataset
   sal = sal.where(sal.lat<-60,drop=True)
   temp= temp.where(temp.lat<-60,drop=True)
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
      print(sal)
      bedvalues[np.isnan(sal.s_an.values[0,0,:,:])]=np.nan
      plt.scatter(sal.x,sal.y,c=bedvalues)
      plt.show()
   temp["bed"] = bedmap.bed.interp(x=temp.x,y=temp.y)
   sal["icemask"] = bedmap.icemask_grounded_and_shelves.interp(x=sal.x,y=sal.y)
   temp["icemask"] = bedmap.icemask_grounded_and_shelves.interp(x=sal.x,y=sal.y)
   sal.icemask.values[sal.icemask.values<1]=0
   temp.icemask.values[temp.icemask.values<1]=0
   return sal,temp


def create_MIMOC(bed,debug = False):
   bedmap = bed
   salfname,tempfname = "data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc"
   filename = "data/MIMOC_Z_GRID_v2.2wm_PT_S_month01.nc"
   dataset = xarray.open_dataset(filename,decode_times=False)
   dataset["POTENTIAL_TEMPERATURE"] = dataset.POTENTIAL_TEMPERATURE.expand_dims(dim={"t": 1})
   dataset = dataset.assign(time=[1])
   dataset["SALINITY"] = dataset.SALINITY.expand_dims(dim={"t": 1})
   dataset = dataset.rename({"SALINITY":"s_an","POTENTIAL_TEMPERATURE":"t_an",\
           "PRESSURE":"depth","PRES":"depth","LATITUDE":"lat","LONGITUDE":"lon","t":"time","LAT":"lat","LONG":"lon"})
   dataset = dataset.set_coords(('lat'))
   dataset = dataset.set_coords(('lon'))
   dataset = dataset.set_coords(('time'))
   dataset = dataset.set_coords(('depth'))
   dataset["t_an"] = dataset.t_an.transpose("time","depth","lat","lon")
   dataset["s_an"] = dataset.s_an.transpose("time","depth","lat","lon")
   sal = dataset
   temp = dataset
   sal = sal.where(sal.lat<-60,drop=True)
   temp= temp.where(temp.lat<-60,drop=True)
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
      print(sal)
      bedvalues[np.isnan(sal.s_an.values[0,0,:,:])]=np.nan
      plt.scatter(sal.x,sal.y,c=bedvalues)
      plt.show()
   temp["bed"] = bedmap.bed.interp(x=temp.x,y=temp.y)
   sal["icemask"] = bedmap.icemask_grounded_and_shelves.interp(x=sal.x,y=sal.y)
   temp["icemask"] = bedmap.icemask_grounded_and_shelves.interp(x=sal.x,y=sal.y)
   sal.icemask.values[sal.icemask.values<1]=0
   temp.icemask.values[temp.icemask.values<1]=0
   return sal,temp

