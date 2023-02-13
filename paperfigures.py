import xarray as xr
import pyproj
import h5py
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

def overview_figure():

    salfname,tempfname = "data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc"
    temp = xr.open_dataset(tempfname,decode_times=False)


    temp = temp.sel(depth=500,drop=True)
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


    fig,ax = plt.subplots(1,1)

    icemask = bedmach.icemask_grounded_and_shelves.values
    icemask[icemask==1] = np.nan

    ax.pcolormesh(bedmach.x.values[::10],bedmach.y.values[::10],icemask[::10,::10],cmap="gray",vmin=-0.5,vmax=0.5)

    filename ='data/amundsilli.h5'
    is_wb = h5py.File(filename,'r')
    print(is_wb)
    wb = np.array(is_wb['/w_b'])

    x_wb = np.array(is_wb['/x'])
    y_wb = np.array(is_wb['/y'])
    wb = np.array(is_wb['/w_b'])

    fig, ax1 = plt.subplots()
    extent = [np.min(is_wb['x']),np.max(is_wb['x']),np.min(is_wb['y']),np.max(is_wb['y'])]
    X,Y = np.meshgrid(x_wb[::20],y_wb[::20])
    wb = wb[::20,::20]
    c1 = ax.pcolormesh(X,Y,wb,vmin=-6,vmax=6,cmap="jet")
    cbar1 = plt.colorbar(c1,ax=ax)
    cbar1.set_label("Melting in m/yr")
    c2 = ax.pcolormesh(raster.x[::10],raster.y[::10],raster.values[0][::10,::10],cmap="magma")
    cbar2 = plt.colorbar(c2,ax=ax,orientation="horizontal")
    cbar2.set_label("WOA temperature at 750m")

    def build_bar(mapx, mapy, ax, width,title, xvals=['a','b','c'], yvals=[1,4,2], fcolors=[0,1]):
        ax_h = inset_axes(ax, width=width, \
                        height=width, \
                        loc=3, \
                        bbox_to_anchor=(mapx, mapy), \
                        bbox_transform=ax.transData, \
                        borderpad=0, \
                        axes_kwargs={'alpha': 0.35, 'visible': True})
        for x,y,c in zip(xvals, yvals, fcolors):
            ax_h.bar(c, y, label=str(x),color="black")
        ax_h.set_xticks(range(len(xvals)), xvals, fontsize=10, rotation=30)
        ax_h.set_yticks(yvals)
        ax_h.set_title(title)
        #ax_h.axis('off')
        return ax_h

    with open("data/glib_by_shelf.pickle","rb") as f:
        glib_by_shelf = pickle.load(f)

    with open("data/simple_shelf_thermals.pickle","rb") as f:
        glibheats = pickle.load(f)

    for k in tqdm(polygons.keys()):
        gons = []
        parts = polygons[k][1]
        polygon = polygons[k][0]
        #if len(parts)>1:
            ##parts.append(-1)
            #for l in range(0,len(parts)-1):
                #poly_path=Polygon(np.asarray(polygon.exterior.coords.xy)[:,parts[l]:parts[l+1]].T)
                #gons.append(poly_path)
        #else:
            #gons = [Polygon(np.asarray(polygon.exterior.coords.xy).T)]
        #p = PatchCollection(gons)
        exterior = np.asarray(polygon.exterior.coords.xy)
        min_i = np.argmin(np.sum(exterior**2,axis=0).shape)
        x = exterior[0][min_i]
        y = exterior[1][min_i]
        #ax.add_collection(p)
        if k in glib_by_shelf and k in glibheats and k in ["Filchner","Pine_Island","Amery","Fimbul","Ross_West"]:
            build_bar(x,y,ax,0.7,k,xvals=["HUB","AISF"],yvals=[glib_by_shelf[k],glibheats[k][3]])

    #icemask[icemask==1]=np.nan
    #plt.pcolormesh(bedmach.x,bedmach.y,icemask)
    #plt.savefig("this.png")
    plt.show()

def hub_schematic_figure():
    with open("data/bedmach.pickle","rb") as f:
        bedmach = pickle.load(f)
    icemask = bedmach.icemask_grounded_and_shelves.values[3400:6500,3400:6000]
    icemask = icemask[::-1,:]
    icemask[icemask==1]=np.nan
    bedvalues = bedmach.bed.values[3400:6500,3400:6000]
    bedvalues = bedvalues[::-1,:]

    fig, ax = plt.subplots(1,1)

    c = ax.imshow(bedvalues,vmin=-2500,vmax=0,cmap="terrain",origin="lower")
    plt.colorbar(c)
    ax.contour(bedvalues,[-600,-575],colors=["green","red"],origin="lower")
    ax.imshow(icemask,zorder=5,origin="lower")

    ax.set_xticks([],[])
    ax.set_yticks([],[])


    axins = ax.inset_axes([0.025, 0.5, 0.45, 0.45])
    axins.set_xticklabels([])
    axins.set_xticks([],[])
    axins.set_yticks([],[])
    axins.set_yticklabels([])
    axins.set_xlim(1300,1650)
    axins.set_ylim(2500,2750)
    axins.imshow(bedvalues,vmin=-2500,vmax=0,cmap="terrain",origin="lower")
    axins.imshow(icemask,zorder=5,origin="lower")
    axins.contour(bedvalues,[-600,-575],colors=["green","red"])
    patch, lines = ax.indicate_inset_zoom(axins, edgecolor="black")
    lines[0].set_visible(True)
    lines[1].set_visible(True)
    lines[2].set_visible(True)
    lines[3].set_visible(True)

    plt.show()
#hub_schematic_figure()
overview_figure()
