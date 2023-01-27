import xarray
import pyproj
import numpy as np
import rockhound as rh
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
import pickle
from matplotlib.patches import Polygon
import rioxarray as riox
from matplotlib.collections import PatchCollection
import rasterio
import shapely

salfname,tempfname = "data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc"
temp = xarray.open_dataset(tempfname,decode_times=False)
temp = temp.sel(depth=1500,drop=True)
temp = temp.isel(time=0,drop=True)
temp = temp.where(temp.lat<-60,drop=True)
temp = temp.rename({"lat":"y","lon":"x"})
temp = temp.rio.write_crs("epsg:4326")
#sal.rio.nodata=np.nan
temp = temp.drop_vars("lon_bnds")
temp = temp.drop_vars("depth_bnds")
temp = temp.drop_dims("nbounds")
##sal = sal.drop_vars("climatology_bounds")
#print(np.nanmean(sal.s_an.values))
temp.rio.nodata=np.nan
import matplotlib

temp = temp.rio.reproject("epsg:3031")
temp.t_an.values[temp.t_an.values>1000] = np.nan

temp.t_an.rio.write_nodata(np.nan, inplace=True)

vars_list = list(temp.data_vars)  
for var in vars_list:  
   del temp[var].attrs['grid_mapping']

temp.t_an.rio.to_raster("data/woafig1.tif")
raster = riox.open_rasterio('data/woafig1.tif')
raster = raster.rio.write_crs("epsg:3031")

lx,ly = raster[0].shape
print(raster.shape)

with open("data/shelfpolygons.pickle","rb") as f:
   polygons = pickle.load(f)

raster = riox.open_rasterio('data/woafig1.tif')

with open("data/bedmach.pickle","rb") as f:
    bedmach = pickle.load(f)

with open("data/glib_by_shelf.pickle","rb") as f:
    glib_by_shelf = pickle.load(f)

icemask = bedmach.icemask_grounded_and_shelves.values

fig,ax = plt.subplots(1,1)

ax.pcolormesh(raster.x,raster.y,raster.values[0])

def build_bar(mapx, mapy, ax, width, xvals=['a','b','c'], yvals=[1,4,2], fcolors=[0,1]):
    ax_h = inset_axes(ax, width=width, \
                    height=width, \
                    loc=3, \
                    bbox_to_anchor=(mapx, mapy), \
                    bbox_transform=ax.transData, \
                    borderpad=0, \
                    axes_kwargs={'alpha': 0.35, 'visible': True})
    for x,y,c in zip(xvals, yvals, fcolors):
        ax_h.bar(c, y, label=str(x))
    ax_h.set_xticks(range(len(xvals)), xvals, fontsize=10, rotation=30)
    ax_h.axis('off')
    return ax_h

for k in tqdm(polygons.keys()):
    gons = []
    parts = polygons[k][1]
    polygon = polygons[k][0]
    if len(parts)>1:
        parts.append(-1)
        for l in range(0,len(parts)-1):
            poly_path=Polygon(np.asarray(polygon.exterior.coords.xy)[:,parts[l]:parts[l+1]].T)
            gons.append(poly_path)
    else:
        gons = [Polygon(np.asarray(polygon.exterior.coords.xy).T)]
    p = PatchCollection(gons)
    centroid = polygon.centroid
    print(centroid)

    ax.add_collection(p)
    if k == "Filchner" or True:
        build_bar(centroid.x,centroid.y,ax,0.2,xvals=["HUB","AISF"],yvals=[-100,-500])

#icemask[icemask==1]=np.nan
#plt.pcolormesh(bedmach.x,bedmach.y,icemask)

plt.show()
