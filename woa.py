import xarray
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


#salfname,tempfname = "data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc"
#sal = xarray.open_dataset(salfname,decode_times=False)
#sal = sal.sel(depth=0,drop=True)
#sal = sal.isel(time=0,drop=True)
#sal = sal.where(sal.lat<-60,drop=True)
#print(sal)
#sal = sal.rename({"lat":"y","lon":"x"})
#sal = sal.rio.write_crs("epsg:4326")
##sal.rio.nodata=np.nan
#sal = sal.drop_vars("lon_bnds")
#sal = sal.drop_vars("depth_bnds")
#sal = sal.drop_dims("nbounds")
###sal = sal.drop_vars("climatology_bounds")
#print(sal)
##print(np.nanmean(sal.s_an.values))
#sal.rio.nodata=np.nan
#
#sal = sal.rio.reproject("epsg:3031")
#sal.s_an.values[sal.s_an.values>1000] = np.nan
#
#sal.s_an.rio.write_nodata(np.nan, inplace=True)
#
#vars_list = list(sal.data_vars)  
#for var in vars_list:  
    #del sal[var].attrs['grid_mapping']
#
#sal.s_an.rio.to_raster("data/woa.tif")
#raster = riox.open_rasterio('data/woa.tif')
#raster = raster.rio.write_crs("epsg:3031")
#
#lx,ly = raster[0].shape
#print(raster.shape)
#
#with open("data/shelfpolygons.pickle","rb") as f:
    #polygons = pickle.load(f)
#

raster = riox.open_rasterio('data/woa.tif')
plt.imshow(raster.values[0])
plt.show()
#for k in tqdm(polygons.keys()):
    #if k == "Ronne":
        #raster = riox.open_rasterio('data/woa.tif')
        #raster = raster.rio.write_crs("epsg:3031")
        #gons = []
        #parts = polygons[k][1]
        #polygon = polygons[k][0]
        #if len(parts)>1:
            #parts.append(-1)
            #for l in range(0,len(parts)-1):
                #poly_path=shapely.geometry.Polygon(np.asarray(polygon.exterior.coords.xy)[:,parts[l]:parts[l+1]].T).buffer(10**4)
                #gons.append(poly_path)
        #else:
            #gons = [polygon]
        #print(k)
        #print("made it hearE")
        #clipped = raster.rio.clip(gons)
        #plt.imshow(clipped[0])
        #plt.show()
#
#
##sal.s_an.plot.pcolormesh()
##plt.show()
##print(np.nanmean(sal.s_an.values))
##
##sal.s_an[0,:,:].plot.pcolormesh()
##plt.show()


