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
import cmocean
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib as mpl
import rasterio
import shapely
from scipy.ndimage import label

def grab_bottom(t,max_depth=500):
    tvalues = t.t_an.values
    depths = t.depth.values
    maxindex = np.argmin(np.abs(depths-500))
    bottom_values = np.empty(tvalues.shape[1:])
    for i in tqdm(range(tvalues.shape[1])):
        for j in range(tvalues.shape[2]):
            nans = np.where(~np.isnan(tvalues[:,i,j]))
            if len(nans)>0 and len(nans[0])>0:
                lastindex = nans[0][-1]
            else:
                lastindex =-1
            if lastindex>-1 and depths[lastindex]<500 :
                bottom_values[i,j] = tvalues[lastindex,i,j]
            else:
                bottom_values[i,j] = tvalues[maxindex,i,j]
    return bottom_values
                


def overview_figure(downscale=2):

    salfname,tempfname = "data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc"
    temp = xr.open_dataset(tempfname,decode_times=False)
    temp = temp.where(temp.lat<-60,drop=True)
    temp = temp.isel(time=0,drop=True)
    B = grab_bottom(temp)
    temp = temp.sel(depth=500,drop=True)
    temp.t_an.values = B

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


    fig,ax = plt.subplots(1,1,figsize=(20,12))
    ax.set_aspect('equal', 'box')

    icemask = bedmach.icemask_grounded_and_shelves.values
    bedmach.bed.values[icemask==0] = np.nan
    bedmach.bed.values[icemask==1] = np.nan
    icemask[icemask==1] = np.nan

    ax.pcolormesh(bedmach.x.values[::downscale],bedmach.y.values[::downscale],icemask[::downscale,::downscale],cmap="gray",vmin=-0.5,vmax=0.5)
    CS = ax.contour(bedmach.x.values[::downscale],bedmach.y.values[::downscale],bedmach.bed.values[::downscale,::downscale],[-1500],colors=["white","green"])
    depths = bedmach.bed.values[::downscale,::downscale]
    labim, num = label(depths>-1500)
    counts = []
    for i in range(1,num):
        counts.append(np.sum(labim==i))
    print(counts)
    countmax = np.argmax(counts)+1
    #print(counts[countmax])
    depths[labim!=countmax] = np.nan
    #depths[depths<-1500]=np.nan
    newcmap = cmocean.tools.crop(cmocean.cm.topo, -2500, 0, 0)
    cx = ax.contourf(bedmach.x.values[::downscale],bedmach.y.values[::downscale],depths,[-1500,-1250,-1000,-750,-500,-250],zorder=2,vmin=-1500,vmax=-250,cmap=newcmap)
    axins3 = inset_axes(
        ax,
        width="20%",  # width: 50% of parent_bbox width
        height="3%",  # height: 5%
        loc="upper right",
    )
    axins3.xaxis.set_ticks_position("bottom")
    cbar3 = fig.colorbar(cx, cax=axins3, orientation="horizontal",ticks=[-400,-900,-1400])
    cbar3.set_label("Continental shelf elevation (m)")



    for level in CS.collections:
        maxlength =0 
        maxlengthkp =0 
        for kp,path in reversed(list(enumerate(level.get_paths()))):
            length = np.max(path.vertices.shape)
            if length>maxlength:
                maxlength = length
                maxlengthkp = kp

        for kp,path in reversed(list(enumerate(level.get_paths()))):
            if kp!=maxlengthkp:
                del(level.get_paths()[kp])

    ax.set_xticks([],[])
    ax.set_yticks([],[])

    filename ='data/amundsilli.h5'
    is_wb = h5py.File(filename,'r')
    print(is_wb)
    wb = np.array(is_wb['/w_b'])

    x_wb = np.array(is_wb['/x'])
    y_wb = np.array(is_wb['/y'])
    wb = np.array(is_wb['/w_b'])

    extent = [np.min(is_wb['x']),np.max(is_wb['x']),np.min(is_wb['y']),np.max(is_wb['y'])]
    X,Y = np.meshgrid(x_wb[::downscale],y_wb[::downscale])
    wb = wb[::downscale,::downscale]
    c1 = ax.pcolormesh(X,Y,wb,zorder=3,vmin=-4,vmax=4,cmap=cmocean.cm.balance)
    ax.axis('off')
    axins1 = inset_axes(
        ax,
        width="20%",  # width: 50% of parent_bbox width
        height="3%",  # height: 5%
        loc="lower left",
    )
    axins1.xaxis.set_ticks_position("bottom")
    cbar1 = fig.colorbar(c1, cax=axins1, orientation="horizontal",ticks=[-4,-2,0,2,4])
    cbar1.set_label("Basal melt rate (m/yr)")


    c2 = ax.pcolormesh(raster.x[::downscale],raster.y[::downscale],raster.values[0][::downscale,::downscale],zorder=0,cmap=cmocean.cm.thermal,vmin=0,vmax=3)
    axins2 = inset_axes(
        ax,
        width="20%",  # width: 50% of parent_bbox width
        height="3%",  # height: 5%
        loc="lower right",
    )
 
    axins2.xaxis.set_ticks_position("bottom")
    cbar2 = fig.colorbar(c2, cax=axins2, orientation="horizontal",ticks=[0,1,2,3])
    cbar2.set_label("WOA temperature at 500m ($^\circ$C)")

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
        #if k in glib_by_shelf and k in glibheats and k in ["Filchner","Pine_Island","Amery","Fimbul","Ross_West"]:
            #build_bar(x,y,ax,0.7,k,xvals=["HUB","AISF"],yvals=[glib_by_shelf[k],glibheats[k][3]])

    #icemask[icemask==1]=np.nan
    #plt.pcolormesh(bedmach.x,bedmach.y,icemask)
    print("saving")
    fig.savefig("paperfigures/OverviewFigure.png",dpi=300)
    #plt.show()

def hub_schematic_figure():
    with open("data/bedmach.pickle","rb") as f:
        bedmach = pickle.load(f)
    icemask = bedmach.icemask_grounded_and_shelves.values[3400:6500,3400:6000-300]
    icemask = icemask[::-1,:]
    icemask[icemask==1]=np.nan


    bedvalues = bedmach.bed.values[3400:6500,3400:6000-300]
    bedvalues = bedvalues[::-1,:]

    fig, ax = plt.subplots(1,1)


    newcmap = cmocean.tools.crop(cmocean.cm.topo, -2500, 0, 0)
    c = ax.imshow(bedvalues,vmin=-2500,vmax=0,cmap=newcmap,origin="lower")
    cbax = plt.colorbar(c,aspect=40,shrink=0.5)
    tick_font_size = 16
    cbax.ax.tick_params(labelsize=tick_font_size)
    ax.contour(bedvalues,[-600,-575],colors=["green","red"],linestyles=["solid","dashed"],origin="lower",linewidths=3)

    ax.set_xticks([],[])
    ax.set_yticks([],[])


    mapins = inset_axes(ax, width="30%", height="30%", loc='lower left',
                   bbox_to_anchor=(0,0,1,1), bbox_transform=ax.transAxes)
    mapins.add_patch(Rectangle((3400,3400),2300,3100,facecolor='red',alpha=0.5))


    coarsefull = bedmach.icemask_grounded_and_shelves.values
    coarsefull[coarsefull==1]=np.nan
    mapins.imshow(coarsefull)
    mapins.set_xticks([],[])
    mapins.set_yticks([],[])

    mpl.rcParams['axes.linewidth'] =5
    axins = ax.inset_axes([0.025, 0.5, 0.45, 0.45])
    axins.spines['bottom'].set_color('white')
    axins.spines['top'].set_color('white')
    axins.spines['right'].set_color('white')
    axins.spines['left'].set_color('white')
    ax.imshow(icemask,zorder=5,origin="lower",cmap="Greys_r",vmin=-0.5,vmax=1)
    axins.set_xticklabels([])
    axins.set_xticks([],[])
    axins.set_yticks([],[])
    axins.set_yticklabels([])
    axins.set_xlim(1300,1650)
    axins.set_ylim(2500,2750)
    axins.imshow(bedvalues,vmin=-2500,vmax=0,cmap=newcmap,origin="lower")
    axins.imshow(icemask,zorder=5,origin="lower",cmap="Greys_r")
    axins.contour(bedvalues,[-600,-575],colors=["green","red"],linestyles=["solid","dashed"],linewidths=3)
    patch, lines = ax.indicate_inset_zoom(axins, edgecolor="black")
    lines[0].set_visible(True)
    lines[1].set_visible(True)
    lines[2].set_visible(True)
    lines[3].set_visible(True)

    plt.show()
#hub_schematic_figure()
overview_figure(downscale=2)
