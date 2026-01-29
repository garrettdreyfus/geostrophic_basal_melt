import xarray as xr
import pyproj
import gsw
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text
from tqdm import tqdm
import pickle
import rioxarray as riox
import cmocean
from matplotlib.patches import Rectangle
import matplotlib as mpl
import rasterio
from scipy.ndimage import label
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from cdw import pycnocline
from scipy.stats import pearsonr
import ipdb
import random
from sympy import Symbol
from sympy import solve,nsolve,re

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
    mpl.rcParams['savefig.dpi'] = 500
    with open("data/bedmach.pickle","rb") as f:
        bedmach = pickle.load(f)
    icemask = bedmach.icemask_grounded_and_shelves.values[3400:6500,3400:6000-300]
    icemask = icemask[::-1,:]
    icemask[icemask==1]=np.nan


    bedvalues = bedmach.bed.values[3400:6500,3400:6000-300]
    bedvalues = bedvalues[::-1,:]

    fig, ax = plt.subplots(1,1,figsize=(16,18))


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
    mapins.add_patch(Rectangle((3400-1100,3400-2000),2300,3100,facecolor='red',alpha=0.5))


    coarsefull = bedmach.icemask_grounded_and_shelves.values
    coarsefull[coarsefull==1]=np.nan
    mapins.imshow(coarsefull[2000:-2000,1100:-1100])
    mapins.set_xticks([],[])
    mapins.set_yticks([],[])

    mpl.rcParams['axes.linewidth'] =5
    axins = ax.inset_axes([0.025, 0.5, 0.45, 0.45],zorder=12)
    axins.spines['bottom'].set_color('white')
    axins.spines['top'].set_color('white')
    axins.spines['right'].set_color('white')
    axins.spines['left'].set_color('white')

    source_crs = 'epsg:3031' # Coordinate system of the file
    target_crs = 'epsg:4326' # Global lat-lon coordinate system
    converter = pyproj.Transformer.from_crs(source_crs, target_crs)
    X,Y=np.meshgrid(bedmach.coords["x"].values,bedmach.coords["y"].values)
    Xcrop,Ycrop = X[3400:6500,3400:6000-300],Y[3400:6500,3400:6000-300]
    lats,lons = converter.transform(Xcrop,Ycrop)
    Xi,Yi = np.meshgrid(range(np.shape(Xcrop)[1]),range(np.shape(Xcrop)[0]))
    CS = ax.contour(Xi,Yi[::-1,:],lons,4,colors="white", zorder=10)
    labels = ax.clabel(CS, CS.levels, inline=True, fmt=lonfmt, fontsize=16,manual=((1385.1508131411074, 336.9720182371744),(1543.1243307655654, 827.702774790029),(1858.143481797842, 1240.8565182021584)))
    CS = ax.contour(Xi,Yi[::-1,:],lats,5,colors="white",zorder=10)

    labels = ax.clabel(CS, CS.levels, inline=True, fmt=latfmt, fontsize=16,manual=((2077.4624486356074, 1391.0308273369785),(1648.2511185370836, 1889.8855571474253),(1295.6808102840082, 2452.214393506218)))
    for i in labels:
        print(i)
    #ax.clabel(CS, CS.levels, inline=True, fontsize=16)
 

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
#overview_figure(downscale=2)

def latfmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    if x>0:
        return f"{s} $^\circ$N"
    if x<0:
        return f"{s} $^\circ$S"
def lonfmt(x):
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    if x>0:
        return f"{s} $^\circ$E"
    if x<0:
        return f"{s} $^\circ$W"

def closestMethodologyFig(bedmap,grid,physical,baths,closest_points,sal,temp,shelves,debug=False,quant="glibheat",shelfkeys=None,point_i=55900):
    plt.figure(figsize=(18,5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
    ax,sideax = plt.subplot(gs[0]),plt.subplot(gs[1])
    print("temp from closest point")
    heats=[np.nan]*len(baths)
    stx = sal.coords["x"].values
    sty = sal.coords["y"].values
    projection = pyproj.Proj("epsg:3031")
    salvals,tempvals = sal.s_an.values[0,:,:,:],temp.t_an.values[0,:,:,:]
    d  = sal.depth.values
    lines = []
    bedvalues = bedmap.bed.values
    mask = np.zeros(salvals.shape[1:])

    icemask = np.empty_like(bedmap.icemask_grounded_and_shelves.values)
    icemask[:] = bedmap.icemask_grounded_and_shelves.values
    icemask[icemask==1]=np.nan

    mask[:]=np.inf
    for l in range(salvals.shape[1]):
        for k in range(salvals.shape[2]):
            if np.sum(~np.isnan(salvals[:,l,k]))>0 and np.max(d[~np.isnan(salvals[:,l,k])])>1500:
                mask[l,k] = 1
    l=point_i
    centroid = [bedmap.coords["x"].values[closest_points[l][1]],bedmap.coords["y"].values[closest_points[l][0]]]
    centroid_i =grid[l] 
    rdist = np.sqrt((sal.coords["x"]- centroid[0])**2 + (sal.coords["y"] - centroid[1])**2)
    rdist = rdist*mask
    closest=np.unravel_index(rdist.argmin(), rdist.shape)
    x = stx[closest[0],closest[1]]
    y = sty[closest[0],closest[1]]
    xC,yC = centroid
    print(x,y,xC,yC)
    X,Y=np.meshgrid(bedmap.coords["x"].values,bedmap.coords["y"].values)
    wym=100
    wyp=2200
    wxm=200
    wxp=200
    ax.set_xticks([],[])
    ax.set_yticks([],[])

    Xcrop = X[centroid_i[0]-wxm:centroid_i[0]+wxp,centroid_i[1]-wym:centroid_i[1]+wyp]
    print(Xcrop.shape)
    Ycrop = Y[centroid_i[0]-wxm:centroid_i[0]+wxp,centroid_i[1]-wym:centroid_i[1]+wyp]

    source_crs = 'epsg:3031' # Coordinate system of the file
    target_crs = 'epsg:4326' # Global lat-lon coordinate system
    converter = pyproj.Transformer.from_crs(source_crs, target_crs)
    lats,lons = converter.transform(Xcrop,Ycrop)

    def latfmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        if x>0:
            return f"{s} $^\circ$N"
        if x<0:
            return f"{s} $^\circ$S"
    def lonfmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        if x>0:
            return f"{s} $^\circ$E"
        if x<0:
            return f"{s} $^\circ$W"



    CS = ax.contour(Xcrop,Ycrop,lats,5,colors="white",zorder=10)
    ax.clabel(CS, CS.levels, inline=True, fmt=latfmt, fontsize=16)
    CS = ax.contour(Xcrop,Ycrop,lons,5,colors="white",zorder=10)
    ax.clabel(CS, CS.levels, inline=True, fmt=lonfmt, fontsize=16)
    


    bedcrop = bedvalues[centroid_i[0]-wxm:centroid_i[0]+wxp,centroid_i[1]-wym:centroid_i[1]+wyp]
    icecrop = icemask[centroid_i[0]-wxm:centroid_i[0]+wxp,centroid_i[1]-wym:centroid_i[1]+wyp]
    
   
    newcmap = cmocean.tools.crop(cmocean.cm.topo, -2500, 0, 0)
    im = ax.pcolormesh(Xcrop,Ycrop,bedcrop,vmin=-2500,vmax=0,cmap=newcmap)
    cbar = plt.colorbar(im,ax=ax,aspect=40,shrink=0.8,location = 'left',pad=0.02)
    cbar.ax.tick_params(labelsize=14)
    ax.pcolormesh(Xcrop,Ycrop,icecrop,zorder=7,cmap="Greys_r",vmin=-0.5,vmax=1)

    shelfmask = np.empty_like(bedmap.icemask_grounded_and_shelves.values)
    shelfmask[:] = bedmap.icemask_grounded_and_shelves.values
    shelfmask[shelfmask==0]=np.nan
    shelfcrop = shelfmask[centroid_i[0]-wxm:centroid_i[0]+wxp,centroid_i[1]-wym:centroid_i[1]+wyp]

    ax.pcolormesh(Xcrop,Ycrop,shelfcrop,zorder=5,cmap="Greys",alpha=0.5)

    ax.contour(Xcrop,Ycrop,bedcrop,[-abs(baths[l])+5],colors=["red"],linestyles=["solid"],zorder=1,linewidths=3)

    ax.scatter(physical[l][0],physical[l][1],s=200,linewidth=3,c="white",marker="*",zorder=10)
    ax.annotate("GL",(physical[l][0],physical[l][1]+2000),fontsize=24,color="white",zorder=10)
    ax.scatter(x,y,s=200,c="white",marker="x",linewidth=3,zorder=10)
    ax.scatter(xC,yC,s=200,c="white",marker="x",linewidth=3,zorder=10)
    ax.annotate("WOA",(x-45000,y-25000),fontsize=24,color="white",zorder=10)

    mapins = inset_axes(ax, width="30%", height="30%", loc='lower right',
                   bbox_to_anchor=(0.075,0,1,1), bbox_transform=ax.transAxes)
    mapins.add_patch(Rectangle((centroid_i[1]-wym,centroid_i[0]-wxm),wym+wyp,wxm+wxp,facecolor='red',alpha=0.5))


    coarsefull = bedmap.icemask_grounded_and_shelves.values
    coarsefull[coarsefull==1]=np.nan
    mapins.imshow(coarsefull)
    mapins.set_xticks([],[])
    mapins.set_yticks([],[])



    t = tempvals[:,closest[0],closest[1]]
    s = salvals[:,closest[0],closest[1]]
    lon,lat = projection(x,y,inverse=True)
    s = gsw.SA_from_SP(s,d,lon,lat)
    #FOR MIMOC MAKE PT
    #t = gsw.CT_from_pt(s,t)
    t = gsw.CT_from_t(s,t,d)

    tinterp,sinterp = interpolate.interp1d(d,np.asarray(t)),interpolate.interp1d(d,np.asarray(s))
    sideax.plot(t,-d)
    
    matplotlib.rcParams['axes.labelcolor'] = 'white'
    buffer = 0.125 # fractional axes coordinates
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    sideax.xaxis.label.set_color('black')
    sideax.yaxis.label.set_color('black')
    sideax.tick_params(axis='x', colors='black',labelsize=14)
    sideax.tick_params(axis='y', colors='black',labelsize=14)
    sideax.set_yticks([0,-500,-1000,-1500])
    sideax.set_xticks([-2,-1,0,1])
    sideax.set_xlabel("Temperature (C)",fontsize=18)
    sideax.set_ylabel("Depth (m)",fontsize=18)

    deltaH = pycnocline((tinterp,sinterp),-abs(baths[l]))
    sideax.axhline(-abs(baths[l])+abs(deltaH),c="blue",lw=3)

    sideax.axhline(-abs(baths[l]),c="red",lw=3)
    sideax.axhspan(-abs(baths[l]), -abs(baths[l])+100, color='red', alpha=0.4, lw=0)
    pyc = pycnocline((tinterp,sinterp),0)
    plt.tight_layout()

    plt.show()


def param_vs_melt_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels,xlim=30,ylim=30,colorthresh=5,textthresh=5,colorfield=None):
    print("deltaH: ", np.mean(cdws),np.std(cdws))
    print("thermals: ",np.mean(thermals),np.std(thermals))
    print("gprimes: ",np.mean(gprimes),np.std(gprimes))
    print("slopes: ", np.mean(slopes),np.std(slopes))
    melts = np.asarray(cdws*np.asarray(thermals)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))
    #melts = np.asarray(slopes)
    mys=np.asarray(mys)
    xs = np.asarray(([melts])).reshape((-1, 1))
    model = LinearRegression(fit_intercept=False).fit(xs, mys)
    r2 = model.score(xs,mys)
    rho0 = 1025
    rhoi = 910
    Cp = 4186
    If = 334000
    C = model.coef_
    W0 =  100000
    alpha =  (C/((rho0*Cp)/(rhoi*If*W0)))/(364*24*60*60)
    print('alpha: ', alpha)
    print('intercept: ', model.intercept_)
    plt.rc('axes', titlesize=24)     # fontsize of the axes title
    xs = np.asarray(([melts])).reshape((-1, 1))
    model = LinearRegression().fit(xs, mys)
    r2 = model.score(xs,mys)
    melts = model.predict(xs)
    ax = plt.gca()
    if colorfield == None:
        ax.scatter(melts[melts<colorthresh],mys[melts<colorthresh],c="blue")
        ax.scatter(melts[melts>colorthresh],mys[melts>colorthresh],c="red")
    else:
        cm = ax.scatter(melts,mys,c=np.asarray(colorfield),vmin=-1,vmax=1,cmap="RdBu_r")
        plt.colorbar(cm,ax=ax)

    markers, caps, bars = ax.errorbar(melts,mys,yerr=sigmas,ls='none')
    [bar.set_alpha(0.5) for bar in bars]
    ax.set_xlim(0,xlim)
    ax.set_ylim(0,ylim)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    texts = []
    for k in range(len(labels)):
        if melts[k]>textthresh:
            text=plt.annotate(labels[k],(melts[k],mys[k]))
            texts.append(text)
    #adjust_text(texts)
    ax.plot(range(30),range(30))
    ax.text(.05, .95, '$r^2=$'+str(str(round(pearsonr(melts.T.flatten(),mys)[0]**2,2))), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
    ax.set_xlabel(r"$\dot{m}_{\mathrm{pred}} (m/yr)$",fontsize=24)
    ax.set_ylabel(r'$\dot{m}_{\mathrm{obs}} (m/yr)$',fontsize=24)
    plt.show()


def hydro_vs_slope_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels,nozone=(1500,0.005),xlim="max",ylim="max"):
    mpl.rcParams['savefig.dpi'] = 500
    melts = np.asarray(cdws*np.asarray(thermals)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))
    mys=np.asarray(mys)
    xs = np.asarray(([melts])).reshape((-1, 1))
    model = LinearRegression().fit(xs, mys)
    tempterms = cdws*np.asarray(thermals)*np.asarray(fs)*np.asarray(gprimes)
    if xlim == "max":
        x = np.linspace(np.min(tempterms)*0.95,np.max(tempterms)*1.05,100)
        y = np.linspace(0,np.max(slopes)*1.05,100)
        X,Y = np.meshgrid(x,y)
        Z = np.multiply(X,Y)*model.coef_[0]+model.intercept_
        im = plt.pcolormesh(X,Y,Z,cmap="gnuplot",vmin=np.min(Z),vmax=33)
    else:
        x = np.linspace(np.min(tempterms)*0.95,xlim,100)
        y = np.linspace(0,ylim,100)
        plt.xlim((np.min(tempterms)*0.95,xlim))
        plt.ylim((0,ylim))
        X,Y = np.meshgrid(x,y)
        Z = np.multiply(X,Y)*model.coef_[0]+model.intercept_
        im = plt.pcolormesh(X,Y,Z,cmap="gnuplot",vmin=np.min(Z),vmax=4)
    cb = plt.colorbar(im)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(18)
    CS = plt.contour(X,Y,Z,levels=[1,2.5,5,10,15,20],colors="white")
    plt.clabel(CS, CS.levels, inline=True, fontsize=10)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel(r"Hydrographic terms $(C m^{2} s^{-1})$",fontsize=24)
    plt.ylabel(r'Ice shelf slope $(m^{-1})$',fontsize=24)
    plt.locator_params(axis='y', nbins=4)
    plt.locator_params(axis='x', nbins=4)
    plt.scatter(tempterms,slopes,c="white")
    for k in range(len(labels)):
        if tempterms[k]>nozone[0] or slopes[k]>nozone[1]:
            plt.annotate(labels[k],(tempterms[k],slopes[k]),c="white",fontsize=14)
    plt.show()

def singleparam_vs_melt_fig(quant,mys,sigmas,labels,xlabel):
    melts = np.asarray(quant)
    mys=np.asarray(mys)
    xs = np.asarray(([melts])).reshape((-1, 1))
    model = LinearRegression(fit_intercept=False).fit(xs, mys)
    r2 = model.score(xs,mys)
    plt.rc('axes', titlesize=24)     # fontsize of the axes title
    xs = np.asarray(([melts])).reshape((-1, 1))
    model = LinearRegression().fit(xs, mys)
    r2 = model.score(xs,mys)
    melts = model.predict(xs)
    ax = plt.gca()
    ax.scatter(quant,mys,c="blue")
    markers, caps, bars = ax.errorbar(quant,mys,yerr=sigmas,ls='none')
    [bar.set_alpha(0.5) for bar in bars]
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    texts = []
    for k in range(len(labels)):
        text=plt.annotate(labels[k],(quant[k],mys[k]))
        texts.append(text)
    #adjust_text(texts)
    print(pearsonr(melts.T.flatten(),mys)[0])
    ax.text(.05, .95, '$r^2=$'+str(pearsonr(melts.T.flatten(),mys)[0]), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
    ax.set_xlabel(xlabel,fontsize=24)
    ax.set_ylabel(r'$\dot{m}_{\mathrm{obs}} (m/yr)$',fontsize=24)
    plt.show()

def masslossparam(cdws,thermals,gprimes,slopes,fs,areas,ccoef,cint,wcoef,wint,thresh=2,mode="soft"):
    if mode == "harsh":
        melts = np.asarray(cdws*np.asarray(thermals)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))*wcoef+wint
        cmask = melts/thermals<thresh
        wmask = ~cmask
        melts[wmask] = (melts[wmask])*areas[wmask]
        melts[cmask] = (slopes[cmask]*areas[cmask])*ccoef + cint
        return melts
    if mode == "soft":
        def transition_func(x):
            return (np.tanh((x-thresh)*3)+1)/2
        warmmelts = np.asarray(cdws*np.asarray(thermals)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))*wcoef+wint
        #decider = warmmelts
        decider = warmmelts/thermals
        frac = transition_func(decider)
        warmmassloss = warmmelts*areas
        coldmassloss = (slopes*areas)*ccoef + cint
        plt.plot(np.mean(frac,axis=0))
        plt.show()
        print(np.min(frac),np.max(frac))
        print("more cold: ",np.sum(frac<0.5)/frac.size)
        print("more warm: ",np.sum(frac>=0.5)/frac.size)
        final = warmmassloss*frac+coldmassloss*(1-frac)
        return final

def shelf_class_fig(shelf_classnumber,labels,sigmas,areas,scalefactor,shelf_color,B0):
    fig,ax=plt.subplots(1,1,figsize=(7,9))

    sortlist = np.argsort(np.asarray(shelf_classnumber))
    sorted_nums = list(np.asarray(shelf_classnumber)[sortlist])
    count=0

    
    ipdb.set_trace()
    for c in sorted(np.unique(shelf_classnumber)):
        categorymask = shelf_classnumber==c
        Bcat = B0[categorymask]
        sigmacat = np.asarray(sigmas)[categorymask]
        sortmask = np.argsort(Bcat)
        counts = range(count,count+np.sum(shelf_classnumber==c))

        ax.errorbar(Bcat[sortmask],counts,xerr=np.asarray(sigmacat)[sortmask]*(1/(60*60*24*365))*(920.0)*areas[categorymask][sortmask]*(10**6)*34.5/1027*9.8*(7.8*10**(-4)),linestyle='',ecolor="red",alpha=0.5,capsize=5)
        ax.scatter(Bcat[sortmask],counts,c=np.asarray(shelf_color)[categorymask][sortmask],zorder=2)

        labelstrunc = np.asarray(labels)[categorymask][sortmask]
        for i in range(len(labelstrunc)):
            if np.sign(Bcat[sortmask][i])>0:
                ax.annotate(labelstrunc[i],(Bcat[sortmask][i]+2.5e-5,counts[i]-0.25),horizontalalignment='left')
            if np.sign(Bcat[sortmask][i])<0:
                ax.annotate(labelstrunc[i],(Bcat[sortmask][i]-2.5e-5,counts[i]-0.25),horizontalalignment='right')
            #ax.annotate(labelstrunc[i],(0.001,counts[i]))
        count+=np.sum(shelf_classnumber==c)

    plt.axhspan(0-0.5,sorted_nums.index(-0.5)-0.5,color="fuchsia",alpha=0.1)
    plt.axhspan(sorted_nums.index(-0.5)-0.5,sorted_nums.index(0)-0.5,color="plum",alpha=0.1)

    plt.axhspan(sorted_nums.index(0)-0.5,sorted_nums.index(0.5)-0.5,color="gray",alpha=0.1)

    plt.axhspan(sorted_nums.index(0.5)-0.5,sorted_nums.index(1)-0.5,color="bisque",alpha=0.1)
    plt.axhspan(sorted_nums.index(1)-0.5,len(sorted_nums)-0.5,color="orange",alpha=0.1)

    x1 = ((sorted_nums.index(-0.5)-0.5) + (0-0.5))/2
    x2 = ((sorted_nums.index(-0.5)-0.5) + (sorted_nums.index(0)-0.5))/2
    x3 = ((sorted_nums.index(0)-0.5) + (sorted_nums.index(0.5)-0.5))/2
    x4 = ((sorted_nums.index(0.5)-0.5) + (sorted_nums.index(1)-0.5))/2
    x5 = ((sorted_nums.index(1)-0.5) + len(sorted_nums))/2

    ax.set_yticks([x1,x2,x3,x4,x5])
    ax.set_yticklabels(['disconnected','likely disconnected','unknown or both','likely connected','connected'])
    # ax.set_xlim(-0.0015,0.0015)
    ax.set_xlabel("$B_{total}$",fontsize=18)
    ax.axvline(x=0,linestyle='--',color='gray')
    fig.subplots_adjust(left=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
    plt.savefig("/home/garrett/Downloads/b0class.svg")
    ipdb.set_trace()



def param_vs_coldmelt_fig(cdws,salts,raw_temps,thermals,gprimes,slopes,volumes,fs,areas,gldepths,hubdepths,mys,sigmas,labels,polynas,polynas_weighted,shelf_class,colorthresh=5,textthresh=5):
    melts = np.asarray(cdws*np.asarray(thermals)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))
    #melts = np.asarray(slopes)
    mys=np.asarray(mys)
    xs = np.asarray(([melts])).reshape((-1, 1))
    model = LinearRegression(fit_intercept=False).fit(xs, mys)
    r2 = model.score(xs,mys)
    rho0 = 1025
    rhoi = 910
    Cp = 4186
    If = 334000
    C = model.coef_
    W0 =  100000
    alpha =  (C/((rho0*Cp)/(rhoi*If*W0)))/(364*24*60*60)
    print('alpha: ', alpha)
    print("C warm:", C)
    print("int warm:", model.intercept_)
    plt.rc('axes', titlesize=24)     # fontsize of the axes title
    xs = np.asarray(([melts])).reshape((-1, 1))
    model = LinearRegression(fit_intercept=False).fit(xs, mys)
    r2 = model.score(xs,mys)

    wcoef = model.coef_
    wint = model.intercept_
    melts = model.predict(xs)
    ax = plt.gca()
    print(np.min(melts/thermals),np.max(melts/thermals),np.mean(melts/thermals),np.std(melts/thermals))
    print(np.asarray(labels)=="Getz")
    gigatonconv = 10**(-12)
    scalefactor = rhoi*gigatonconv*10**6

    Ronnei = labels.index("Ronne")
    Filchneri = labels.index("Filchner")
    #meanfrisratio = (mys[Ronnei]*areas[Ronnei]+mys[Filchneri]*areas[Filchneri])*scalefactor/(polynas[Ronnei]+polynas[Filchneri])
    #print("mreanfris",meanfrisratio)
    #meanfrismelt = (mys[Ronnei]*areas[Ronnei]+mys[Filchneri]*areas[Filchneri])/(areas[Ronnei]+areas[Filchneri])
    #mys[Ronnei]=meanfrismelt
    #mys[Filchneri]=meanfrismelt

    #sumfrispolyna = polynas[Ronnei] + polynas[Filchneri]
    #polynas[Ronnei]=sumfrispolyna
    #polynas[Filchneri]=sumfrispolyna

    #sumfrisarea = areas[Ronnei]+areas[Filchneri]
    #areas[Ronnei]=sumfrisarea
    #areas[Filchneri]=sumfrisarea

    meltflux = mys*(1/(60*60*24*365))*(920.0)
    Btotal = -(meltflux*areas*34.5-polynas)/1027*9.8*(7.8*10**(-4))

    N = np.sqrt( (9.8/1027)*(gprimes*1027/9.8)/50)

    Tpolyna = -1.9

    rhoanom = (3.9**2)*(1/(abs(hubdepths)))*(rho0/9.8)*((2*np.abs(Btotal))/(np.sqrt(areas)))**(2/3)*np.sign(Btotal)
    rhoanom[Btotal<0] = np.nan
    beta = gsw.beta(salts,-1.9)
    Spolyna = salts + rhoanom/beta

    Tf = gsw.CT_freezing(salts,np.abs(gldepths),0)

    Cp = 4186
    If = 334000
    Sm = salts/(1-(Cp/If)*(Tf-(Tpolyna)))

    rho_2 = gsw.rho(Spolyna,Tpolyna,0)
    rho_1 = gsw.rho((Spolyna+Sm)/2,(Tpolyna+Tf)/2,0)

    gprimegade = 9.8*(rho_2-rho_1)/((rho_1+rho_2)/2)

    # Tf = fpAtGl(zgl,Spolyna)
    meltgade =  gprimegade*slopes*(Tpolyna-(Tpolyna+Tf)/2)*abs(hubdepths)/fs



    # he = (3/(2*0.025))**(1/3)*(1/N)*(np.abs(B0)/(np.sqrt(areas)))**(1/3)*np.sign(B0)

    #ratio = gprimes*(mys*areas*scalefactor)/polynas

    #ratio = 
    rho0 = 1025
    #rhoanom = ((1/ce)**0.5) * ((np.nanmean(localdens[rho_1i]))/(9.8*tcline_height))*np.sign(B0)*np.abs(f*shelf_width*B0/P)**0.5
    #polynas[Ronnei]=meanfrispolyna
    #polynas[Filchneri]=meanfrispolyna
    #he[Ronnei]=-300
    #he[Filchneri]=-300


    #cmask = ratio<0.0005#0325
    cmask = np.asarray(shelf_class)<=0#0.0005

    #cmask[np.asarray(labels)=="Getz"]=0
    #print(len(slopes[cmask]),len(areas[cmask]),len(mys[cmask]))
    #areas = areas.T
    #slopes = slopes.T[0]
    #adjpolyna = (-np.asarray(polynas)*(34.5/1000)*365*24*60*60)/(volumes*500*500)
    #adjpolyna = (1/salts)
    #adjpolyna = (gsw.rho(salts+(-np.asarray(polynas)*(34.5/1000)*365*24*60*60)/(volumes*500*500)*641,raw_temps,0)-gsw.rho(32.0,-1.9,0))/np.mean([gsw.rho(salts+(-np.asarray(polynas)*(34.5/1000)*365*24*60*60)/(volumes*500*500)*641,raw_temps,0),gsw.rho(32,-1.9,0)])
    #glfreezing = (-1.9-gsw.CT_freezing(salts+(-np.asarray(polynas)*(34.5/1000)*365*24*60*60)/(volumes*500*500)*641,np.abs(gldepths),0))
    glfreezing = (-1.8-gsw.CT_freezing(salts,np.abs(gldepths),0))
    fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2)
    ax1.scatter(meltgade,mys)
    for k in range(len(labels)):
        if cmask[k] or True:
           text=ax1.annotate(labels[k],(meltgade[k],mys[k]))
    ax1.set_title("thickness")
    ax2.scatter(rhoanom+salts,mys)
    for k in range(len(labels)):
        if cmask[k] or True:
            text=ax2.annotate(labels[k],(rhoanom[k]+salts[k],mys[k]))
    ax2.set_title("polyna")
    ax3.scatter((slopes),mys)
    for k in range(len(labels)):
        if cmask[k] or True:
            text=ax3.annotate(labels[k],(slopes[k],mys[k]))
    ax3.set_title("slopes")
    ax4.scatter(glfreezing,mys)
    for k in range(len(labels)):
        if cmask[k] or True:
            text=ax4.annotate(labels[k],(glfreezing[k],mys[k]))

    ax4.set_title("freezing")
    ax5.scatter((cdws),mys)
    sigmas = np.asarray(sigmas)
    markers, caps, bars = ax5.errorbar((cdws),mys,yerr=sigmas,ls='none')
    [bar.set_alpha(0.5) for bar in bars]
    for k in range(len(labels)):
        if cmask[k] or True:
            text=ax5.annotate(labels[k],((cdws)[k],mys[k]))
    ax5.set_title("cdws")
    ax6.scatter(slopes[cmask],mys[cmask])
    for k in range(len(labels)):
        if cmask[k] or True:
            text=ax6.annotate(labels[k],(slopes[k],mys[k]))
    ax6.set_title("gprimes")
 
    plt.show()

    print(slopes)
    print(areas)

    fig, ax = plt.subplots(1,1)
    #coldprod = np.asarray(polynas)*np.asarray(slopes)*np.asarray(areas)*gsw.CT_freezing(34.9,np.abs(gldepths),0)
    #coldprod = np.asarray(polynas)*np.asarray(slopes)*gsw.CT_freezing(34.9,np.abs(gldepths),0)
    #coldprod = gsw.CT_freezing(34.9,np.abs(gldepths),0)*np.asarray(polynas)*np.asarray(slopes)
    #coldprod = (1.9-gsw.CT_freezing(34.9,np.abs(gldepths),0))*np.asarray(slopes)*np.log10(np.asarray(polynas)/np.asarray(volumes))
    #coldprod = (1.9-gsw.CT_freezing(34.9,np.abs(gldepths),0))*np.asarray(slopes)*(np.asarray(polynas)/np.asarray(volumes))
    polynas = polynas*(34.5/1000)*365*24*60*60
    print("polynas:",polynas/(volumes*500*500))
    #coldprod = ((34.35+(-np.asarray(polynas)/(volumes*500*500))*641)-34.2)*slopes
    #coldprod = adjpolyna*slopes*(np.abs(hubdepths))*np.asarray(fs)*gprimes*glfreezing

    coldprod = slopes*(np.abs(hubdepths))*np.asarray(fs)*glfreezing#*gprimechapman#*glfreezing#*gprimechapman
    #coldprod = (1/volumes)*slopes
    print("coldprod:",coldprod)
    #coldprod = (1.9-gsw.CT_freezing(34.9,np.abs(gldepths),0))*np.asarray(slopes)*np.asarray(polynas)/np.asarray(volumes)
    xs = np.asarray(([coldprod[cmask]*areas[cmask]])).reshape((-1, 1))
    #coldmodel = LinearRegression(fit_intercept=False).fit(xs, np.asarray(mys[cmask])*np.asarray(areas[cmask]))
    coldmodel = LinearRegression(fit_intercept=False).fit(xs, np.asarray(mys[cmask]*areas[cmask]))
    r2 = coldmodel.score(xs,np.asarray(mys[cmask]*areas[cmask]))

    alpha =  (coldmodel.coef_/((rho0*Cp)/(rhoi*If*W0)))/(364*24*60*60)
    print("C cold:", coldmodel.coef_)
    print("int cold:", coldmodel.intercept_)
    print("alpha cold:",alpha)
    ccoef = coldmodel.coef_
    cint = coldmodel.intercept_
    coldpred = coldprod[cmask]*coldmodel.coef_*scalefactor*areas[cmask]
    coldans = np.asarray(mys[cmask])*np.asarray(areas[cmask])*scalefactor
    warmpred = np.asarray(melts[~cmask])*np.asarray(areas[~cmask])*scalefactor
    warmans = np.asarray(mys[~cmask])*np.asarray(areas[~cmask])*scalefactor
    print("warmans sum: ,", np.sum(warmans))
    print("warmpred sum: ,", np.sum(warmpred))
    print("coldans sum: ,", np.sum(coldans))
    print("coldpred sum: ", np.sum(coldpred))
    print("total ans sum: ", np.sum(warmans)+np.sum(coldans))
    print("total pred sum: ", np.sum(warmpred)+np.sum(coldpred))
    ax.scatter(warmpred,warmans,c="red")
    ax.scatter(coldpred,coldans,c="blue")
    xs = np.concatenate((warmpred,coldpred))
    ys = np.concatenate((warmans,coldans))
    xs = np.asarray(([xs])).reshape((-1, 1))
    model = LinearRegression(fit_intercept=False).fit(xs, ys)
    r2 = pearsonr(xs.flatten(),ys.flatten()).statistic**2
    #markers, caps, bars = ax.errorbar(melts,mys,yerr=sigmas,ls='none')
    #[bar.set_alpha(0.5) for bar in bars]
    #ax.set_xlim(0,xlim)
    #ax.set_ylim(0,ylim)

    sigmas = np.asarray(sigmas)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    texts = []
    for k in range(len(labels)):
        if cmask[k]:
            1+1
            text=plt.annotate(labels[k],(coldprod[k]*areas[k]*coldmodel.coef_*scalefactor,mys[k]*areas[k]*scalefactor))
            #text=plt.annotate(labels[k],(coldprod[k]*coldmodel.coef_*scalefactor,mys[k]*areas[k]*scalefactor))
            texts.append(text)
        else:
            1+1
            text=plt.annotate(labels[k],(melts[k]*areas[k]*scalefactor,mys[k]*areas[k]*scalefactor))
            #texts.append(text)

    markers, caps, bars = ax.errorbar(coldpred,coldans,yerr=sigmas[cmask]*scalefactor*areas[cmask],ls='none')
    [bar.set_alpha(0.5) for bar in bars]
    markers, caps, bars = ax.errorbar(warmpred,warmans,yerr=sigmas[~cmask]*scalefactor*areas[~cmask],ls='none')
    [bar.set_alpha(0.5) for bar in bars]
    #adjust_text(texts)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(range(30),range(30))
    ax.text(.05, .95, '$r^2=$'+str(round(r2,2)), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
    ax.set_xlabel(r"$\dot{M}_{\mathrm{pred}} (Gt/yr)$",fontsize=24)
    ax.set_ylabel(r'$\dot{M}_{\mathrm{obs}} (Gt/yr)$',fontsize=24)
    ax.plot((0,np.max(warmans)),(0,np.max(warmans)))
    plt.show()
    return ccoef,cint,wcoef,wint



def cleanlog(cdws,salts,raw_temps,thermals,gprimes,front_spread,slopes,volumes,fs,areas,gldepths,hubdepths,mys,sigmas,labels,polynas,polynas_weighted,shelf_class,colorthresh=5,textthresh=5):

    areas = np.asarray(areas)

    wmask = np.asarray((shelf_class))>0#0.0005
    warm_mys=np.asarray(mys)[wmask]
    warmfull = np.asarray(cdws*np.asarray(thermals)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))*np.asarray(areas)
    warm = warmfull[wmask]
    warm_xs = np.asarray(([warm])).reshape((-1, 1))
    warm_model = LinearRegression(fit_intercept=False).fit(np.log10(warm_xs), np.log10(warm_mys*areas[wmask]))
    warm_melts = warm_model.predict(np.log10(warm_xs))

    cmask = np.asarray(shelf_class)<0#0.0005
    cold_mys=np.asarray(mys)[cmask]
    glfreezing = (-1.8-gsw.CT_freezing(salts,np.abs(gldepths),0))
    # coldfull = slopes*(np.abs(hubdepths))*np.asarray(fs)*glfreezing*np.asarray(areas)#*gprimechapman#*glfreezing#*gprimechapman
    coldfull = slopes*(np.abs(hubdepths))*np.asarray(fs)*np.asarray(areas)*(1/6)*glfreezing*np.sqrt(polynas)*front_spread#*gprimechapman#*glfreezing#*gprimechapman
    cold = coldfull[cmask]
    cold_xs = np.asarray(([cold])).reshape((-1, 1))
    cold_model = LinearRegression(fit_intercept=False).fit(np.log10(cold_xs), np.log10(cold_mys*areas[cmask]))
    cold_melts = cold_model.predict(np.log10(cold_xs))


    graymask = np.asarray(shelf_class)==0#0.0005
    graycold_xs = np.asarray(([coldfull[graymask]])).reshape((-1,1))
    graywarm_xs = np.asarray(([warmfull[graymask]])).reshape((-1,1))
    if len(graycold_xs)>0:
        gray_melts = (cold_model.predict(np.log10(graycold_xs)) + warm_model.predict(np.log10(graywarm_xs)))/2.0
    else:
        gray_melts = np.asarray([])
    gray_mys  = np.asarray(mys)[graymask]

    r2 = pearsonr(np.concatenate((gray_melts.flatten(),cold_melts.flatten(),warm_melts.flatten())),np.log10(np.concatenate((gray_mys.flatten()*areas[graymask],cold_mys.flatten()*areas[cmask],warm_mys.flatten()*areas[wmask])))).statistic**2

    fig, ax = plt.subplots(1,1)

    rhoi = 910
    gigatonconv = 10**(-12)
    scalefactor = rhoi*gigatonconv*10**6

    ax.scatter(10**(warm_melts.flatten())*scalefactor,scalefactor*warm_mys*areas[wmask],c="red")
    ax.scatter(10**(cold_melts.flatten())*scalefactor,scalefactor*cold_mys*areas[cmask],c="blue")
    ax.scatter(10**(gray_melts.flatten())*scalefactor,scalefactor*gray_mys*areas[graymask],c="gray")
    ax.set_yscale('log')
    ax.set_xscale('log')


    cmaskin = np.cumsum(cmask)-1
    wmaskin = np.cumsum(wmask)-1
    graymaskin = np.cumsum(graymask)-1

    sigmas = np.asarray(sigmas)
    for k in range(len(labels)):
        if cmask[k]:
            text=plt.annotate(labels[k],(10**(cold_melts.flatten()[cmaskin[k]])*scalefactor,(scalefactor*cold_mys*areas[cmask])[cmaskin[k]]))
        elif wmask[k]:
            text=plt.annotate(labels[k],(10**(warm_melts.flatten()[wmaskin[k]])*scalefactor,(scalefactor*warm_mys*areas[wmask])[wmaskin[k]]))
        elif graymask[k]:
            text=plt.annotate(labels[k],(10**(gray_melts.flatten()[graymaskin[k]])*scalefactor,(scalefactor*gray_mys*areas[graymask])[graymaskin[k]]))

    markers, caps, bars = ax.errorbar(10**(cold_melts.flatten())*scalefactor,(scalefactor*cold_mys*areas[cmask]),yerr=sigmas[cmask]*areas[cmask]*scalefactor,ls='none')
    [bar.set_alpha(0.3) for bar in bars]
    markers, caps, bars = ax.errorbar(10**(warm_melts.flatten())*scalefactor,(scalefactor*warm_mys*areas[wmask]),yerr=sigmas[wmask]*areas[wmask]*scalefactor,ls='none')
    [bar.set_alpha(0.3) for bar in bars]
    markers, caps, bars = ax.errorbar(10**(gray_melts.flatten())*scalefactor,(scalefactor*gray_mys*areas[graymask]),yerr=sigmas[graymask]*areas[graymask]*scalefactor,ls='none',ecolor="gray")
    [bar.set_alpha(0.3) for bar in bars]

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    ax.text(.05, .95, '$r^2=$'+str(round(r2,2)), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
    ax.set_xlabel(r"$\dot{M}_{\mathrm{pred}} (Gt/yr)$",fontsize=24)
    ax.set_ylabel(r'$\dot{M}_{\mathrm{obs}} (Gt/yr)$',fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

def clean(cdws,salts,raw_temps,thermals,gprimes,front_spread,h_min,h_max,slopes,volumes,fs,areas,gldepths,hubdepths,mys,sigmas,labels,polynas,polynas_weighted,shelf_class,colorthresh=5,textthresh=5):
    rho0 = 1025
    rhoi = 910
    Cp = 4186
    If = 334000
    W0 =  100000
 
    glfreezing = (-1.9-gsw.CT_freezing(salts,np.abs(gldepths)/2,0))/4
    # areas = np.asarray(areas)
    areas = areas*(10**6)
    wmask = np.asarray((shelf_class))>0#0.0005
    warm_mys=np.asarray(mys)[wmask]
    warmfull = np.asarray(cdws*np.asarray(thermals)*(1/6)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))*np.asarray(areas)
    warm = warmfull[wmask]
    warm_xs = np.asarray(([warm])).reshape((-1, 1))
    warm_model = LinearRegression(fit_intercept=False).fit(warm_xs, warm_mys*areas[wmask])
    warm_melts = warm_model.predict(warm_xs)


    ######## cold g'
    
    rhomin,rhomax = (rho0/9.8)*((1/fs)*(polynas*(9.8/1027)))**(1/2)/(h_max),(rho0/9.8)*((1/fs)*(polynas))**(1/2)/(h_min)
    rhomean = (rho0/9.8)*((1/fs)*(polynas))**(1/2)/(np.nanmean((h_max+h_min)/2))
    stratterm = rhomax-rhomin

    rho_s = gsw.beta(salts,-1.9,0)*rho0

    Spolyna = salts + rhomean/rho_s
    Tf = gsw.CT_freezing(salts,np.abs(gldepths)/2,0)
    Tpolyna = -1.9
    D = (1-(Cp/If)*(Tf-(Tpolyna))/4)
    gprimes_cold = (9.8/1027)*(Spolyna*(1-1/D)*rho_s + (stratterm/6))



    ##########3

    cmask = np.asarray(shelf_class)<0#0.0005
    cold_mys=np.asarray(mys)[cmask]
    coldfull = slopes*(np.abs(hubdepths))*np.asarray(fs)*np.asarray(areas)*glfreezing*gprimes_cold#*gprimechapman#*glfreezing#*gprimechapman
    cold = coldfull[cmask]
    cold_xs = np.asarray(([cold])).reshape((-1, 1))
    cold_model = LinearRegression(fit_intercept=False).fit(cold_xs, cold_mys*areas[cmask])
    cold_melts = cold_model.predict(cold_xs)

    warm_alpha =  (warm_model.coef_/((rho0*Cp)/(rhoi*If*W0)))/(364*24*60*60)
    cold_alpha =  (cold_model.coef_/((rho0*Cp)/(rhoi*If*W0)))/(364*24*60*60)
    print("warm alpha: ",warm_alpha)
    print("cold alpha: ",cold_alpha)


    ipdb.set_trace()


    graymask = np.asarray(shelf_class)==0#0.0005
    graycold_xs = np.asarray(([coldfull[graymask]])).reshape((-1,1))
    graywarm_xs = np.asarray(([warmfull[graymask]])).reshape((-1,1))
    if len(graycold_xs)>0:
        gray_melts = (cold_model.predict(graycold_xs) + warm_model.predict(graywarm_xs))/2.0
    else:
        gray_melts = np.asarray([])
    gray_mys  = np.asarray(mys)[graymask]

    r2 = pearsonr(np.concatenate((cold_melts.flatten(),warm_melts.flatten(),gray_melts.flatten())),np.concatenate((cold_mys.flatten()*areas[cmask],warm_mys.flatten()*areas[wmask],gray_mys.flatten()*areas[graymask]))).statistic**2
    fig, ax = plt.subplots(1,1)

    rhoi = 910
    gigatonconv = 10**(-12)
    scalefactor = rhoi*gigatonconv

    ax.scatter(warm_melts.flatten()*scalefactor,scalefactor*warm_mys*areas[wmask],c="red")
    ax.scatter(cold_melts.flatten()*scalefactor,scalefactor*cold_mys*areas[cmask],c="blue")
    ax.scatter(gray_melts.flatten()*scalefactor,scalefactor*gray_mys*areas[graymask],c="gray")

    cmaskin = np.cumsum(cmask)-1
    wmaskin = np.cumsum(wmask)-1
    graymaskin = np.cumsum(graymask)-1

    sigmas = np.asarray(sigmas)
    for k in range(len(labels)):
        if cmask[k]:
            text=plt.annotate(labels[k],(cold_melts.flatten()[cmaskin[k]]*scalefactor,(scalefactor*cold_mys*areas[cmask])[cmaskin[k]]))
        elif wmask[k]:
            text=plt.annotate(labels[k],(warm_melts.flatten()[wmaskin[k]]*scalefactor,(scalefactor*warm_mys*areas[wmask])[wmaskin[k]]))
        elif graymask[k]:
            text=plt.annotate(labels[k],(gray_melts.flatten()[graymaskin[k]]*scalefactor,(scalefactor*gray_mys*areas[graymask])[graymaskin[k]]))

    markers, caps, bars = ax.errorbar(cold_melts.flatten()*scalefactor,scalefactor*cold_mys*areas[cmask],yerr=sigmas[cmask]*areas[cmask]*scalefactor,ls='none')
    [bar.set_alpha(0.3) for bar in bars]                  
    markers, caps, bars = ax.errorbar(warm_melts.flatten()*scalefactor,scalefactor*warm_mys*areas[wmask],yerr=sigmas[wmask]*areas[wmask]*scalefactor,ls='none')
    [bar.set_alpha(0.3) for bar in bars]                  
    markers, caps, bars = ax.errorbar(gray_melts.flatten()*scalefactor,scalefactor*gray_mys*areas[graymask],yerr=sigmas[graymask]*areas[graymask]*scalefactor,ls='none',ecolor="gray")
    [bar.set_alpha(0.3) for bar in bars]

    ax.text(.05, .95, '$r^2=$'+str(round(r2,2)), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
    ax.set_xlabel(r"$\dot{M}_{\mathrm{pred}} (Gt/yr)$",fontsize=24)
    ax.set_ylabel(r'$\dot{M}_{\mathrm{obs}} (Gt/yr)$',fontsize=24)
    ax.set_xlim(0,140)
    ax.set_ylim(0,140)

    plt.show()

def clean_optimal(cdws,salts,raw_temps,thermals,gprimes,front_spread,slopes,volumes,fs,areas,gldepths,hubdepths,mys,sigmas,labels,polynas,polynas_weighted,shelf_class,colorthresh=5,textthresh=5):

    Ronnei = labels.index("Ronne")
    Filchneri = labels.index("Filchner")

    meanfrismelt = (mys[Ronnei]*areas[Ronnei]+mys[Filchneri]*areas[Filchneri])/(areas[Ronnei]+areas[Filchneri])
    mys[Ronnei]=meanfrismelt
    mys[Filchneri]=meanfrismelt

    sumfrispolyna = polynas[Ronnei] + polynas[Filchneri]

    sumfrisarea = areas[Ronnei]+areas[Filchneri]
    areas[Ronnei]=sumfrisarea
    areas[Filchneri]=sumfrisarea

    meanslope = (slopes[Ronnei] + slopes[Filchneri])/2
    slopes[Ronnei]=meanslope
    slopes[Filchneri]=meanslope



    # areas = np.asarray(areas)

    maxr = 0
    max_shelf_class = []
    r2s = []
    for i in tqdm(range(10000)):
        shelf_class = random.choices([-1,0,1],k=len(shelf_class))
        wmask = np.asarray((shelf_class))>0#0.0005
        warm_mys=np.asarray(mys)[wmask]
        warmfull = np.asarray(cdws*np.asarray(thermals)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))*np.asarray(areas)
        warm = warmfull[wmask]
        warm_xs = np.asarray(([warm])).reshape((-1, 1))
        warm_model = LinearRegression(fit_intercept=False).fit(warm_xs, warm_mys*areas[wmask])
        warm_melts = warm_model.predict(warm_xs)

        cmask = np.asarray(shelf_class)<0#0.0005
        cold_mys=np.asarray(mys)[cmask]
        glfreezing = (-1.9-gsw.CT_freezing(salts,np.abs(gldepths),0))
        coldfull = slopes*(np.abs(hubdepths))*np.asarray(fs)*glfreezing*np.asarray(areas)#*gprimechapman#*glfreezing#*gprimechapman
        cold = coldfull[cmask]
        cold_xs = np.asarray(([cold])).reshape((-1, 1))
        cold_model = LinearRegression(fit_intercept=False).fit(cold_xs, cold_mys*areas[cmask])
        cold_melts = cold_model.predict(cold_xs)


        graymask = np.asarray(shelf_class)==0#0.0005
        graycold_xs = np.asarray(([coldfull[graymask]])).reshape((-1,1))
        graywarm_xs = np.asarray(([warmfull[graymask]])).reshape((-1,1))
        gray_melts = (cold_model.predict(graycold_xs) + warm_model.predict(graywarm_xs))/2.0
        gray_mys  = np.asarray(mys)[graymask]

        r2 = pearsonr(np.concatenate((cold_melts.flatten(),warm_melts.flatten(),gray_melts.flatten())),np.concatenate((cold_mys.flatten()*areas[cmask],warm_mys.flatten()*areas[wmask],gray_mys.flatten()*areas[graymask]))).statistic**2
        r2s.append(r2)
        if r2>maxr:
            maxr=r2
            max_shelf_class = shelf_class

    plt.hist(r2s,bins=20)
    plt.show()
    shelf_class = max_shelf_class
    wmask = np.asarray((shelf_class))>0#0.0005
    warm_mys=np.asarray(mys)[wmask]
    warmfull = np.asarray(cdws*np.asarray(thermals)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))*np.asarray(areas)
    warm = warmfull[wmask]
    warm_xs = np.asarray(([warm])).reshape((-1, 1))
    warm_model = LinearRegression(fit_intercept=False).fit(warm_xs, warm_mys*areas[wmask])
    warm_melts = warm_model.predict(warm_xs)

    cmask = np.asarray(shelf_class)<0#0.0005
    cold_mys=np.asarray(mys)[cmask]
    glfreezing = (-1.9-gsw.CT_freezing(salts,np.abs(gldepths),0))
    coldfull = slopes*(np.abs(hubdepths))*np.asarray(fs)*glfreezing*np.asarray(areas)#*gprimechapman#*glfreezing#*gprimechapman
    cold = coldfull[cmask]
    cold_xs = np.asarray(([cold])).reshape((-1, 1))
    cold_model = LinearRegression(fit_intercept=False).fit(cold_xs, cold_mys*areas[cmask])
    cold_melts = cold_model.predict(cold_xs)


    graymask = np.asarray(shelf_class)==0#0.0005
    graycold_xs = np.asarray(([coldfull[graymask]])).reshape((-1,1))
    graywarm_xs = np.asarray(([warmfull[graymask]])).reshape((-1,1))
    gray_melts = (cold_model.predict(graycold_xs) + warm_model.predict(graywarm_xs))/2.0
    gray_mys  = np.asarray(mys)[graymask]

    r2 = pearsonr(np.concatenate((cold_melts.flatten(),warm_melts.flatten(),gray_melts.flatten())),np.concatenate((cold_mys.flatten()*areas[cmask],warm_mys.flatten()*areas[wmask],gray_mys.flatten()*areas[graymask]))).statistic**2
    if r2>maxr:
        maxr=r2
        max_shelf_class = shelf_class


    fig, ax = plt.subplots(1,1)

    rhoi = 910
    gigatonconv = 10**(-12)
    scalefactor = rhoi*gigatonconv*10**6

    ax.scatter(warm_melts.flatten()*scalefactor,scalefactor*warm_mys*areas[wmask],c="red")
    ax.scatter(cold_melts.flatten()*scalefactor,scalefactor*cold_mys*areas[cmask],c="blue")
    ax.scatter(gray_melts.flatten()*scalefactor,scalefactor*gray_mys*areas[graymask],c="gray")

    cmaskin = np.cumsum(cmask)-1
    wmaskin = np.cumsum(wmask)-1
    graymaskin = np.cumsum(graymask)-1

    sigmas = np.asarray(sigmas)
    for k in range(len(labels)):
        if cmask[k]:
            text=plt.annotate(labels[k],(cold_melts.flatten()[cmaskin[k]]*scalefactor,(scalefactor*cold_mys*areas[cmask])[cmaskin[k]]))
        elif wmask[k]:
            text=plt.annotate(labels[k],(warm_melts.flatten()[wmaskin[k]]*scalefactor,(scalefactor*warm_mys*areas[wmask])[wmaskin[k]]))
        elif graymask[k]:
            text=plt.annotate(labels[k],(gray_melts.flatten()[graymaskin[k]]*scalefactor,(scalefactor*gray_mys*areas[graymask])[graymaskin[k]]))

    markers, caps, bars = ax.errorbar(cold_melts.flatten()*scalefactor,scalefactor*cold_mys*areas[cmask],yerr=sigmas[cmask]*areas[cmask]*scalefactor,ls='none')
    [bar.set_alpha(0.3) for bar in bars]                  
    markers, caps, bars = ax.errorbar(warm_melts.flatten()*scalefactor,scalefactor*warm_mys*areas[wmask],yerr=sigmas[wmask]*areas[wmask]*scalefactor,ls='none')
    [bar.set_alpha(0.3) for bar in bars]                  
    markers, caps, bars = ax.errorbar(gray_melts.flatten()*scalefactor,scalefactor*gray_mys*areas[graymask],yerr=sigmas[graymask]*areas[graymask]*scalefactor,ls='none',ecolor="gray")
    [bar.set_alpha(0.3) for bar in bars]

    ax.text(.05, .95, '$r^2=$'+str(round(r2,2)), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
    ax.set_xlabel(r"$\dot{M}_{\mathrm{pred}} (Gt/yr)$",fontsize=24)
    ax.set_ylabel(r'$\dot{M}_{\mathrm{obs}} (Gt/yr)$',fontsize=24)
    ax.set_xlim(0,140)
    ax.set_ylim(0,140)

    plt.show()


def breakdown_cold(cdws,salts,raw_temps,thermals,gprimes,front_spread,h_min,h_max,slopes,volumes,fs,areas,gldepths,hubdepths,mys,sigmas,labels,polynas,polynas_weighted,shelf_class,colorthresh=5,textthresh=5):
    glfreezing = (gsw.CT_freezing(salts,0,0)-gsw.CT_freezing(salts,np.abs(gldepths)/2,0))/4

    rho0 = 1025
    rhoi = 910
    Cp = 4186
    If = 334000
    W0 =  100000
 
    # areas = np.asarray(areas)
    areas = areas*(10**6)
    wmask = np.asarray((shelf_class))>0#0.0005
    warm_mys=np.asarray(mys)[wmask]
    warmfull = np.asarray(cdws*np.asarray(thermals)*(1/6)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))*np.asarray(areas)
    warm = warmfull[wmask]
    warm_xs = np.asarray(([warm])).reshape((-1, 1))
    warm_model = LinearRegression(fit_intercept=False).fit(warm_xs, warm_mys*areas[wmask])
    warm_melts = warm_model.predict(warm_xs)

    cmask = np.asarray(shelf_class)<0#0.0005
    cold_mys=np.asarray(mys)[cmask]
    coldfull = slopes*(np.abs(hubdepths))*np.asarray(fs)*np.asarray(areas)*glfreezing*(np.sqrt(polynas)*front_spread)#*gprimechapman#*glfreezing#*gprimechapman
    cold = coldfull[cmask]
    cold_xs = np.asarray(([cold])).reshape((-1, 1))
    cold_model = LinearRegression(fit_intercept=False).fit(cold_xs, cold_mys*areas[cmask])
    cold_melts = cold_model.predict(cold_xs)

    rhomin,rhomax = (rho0/9.8)*((1/fs)*(polynas*(9.8/1027)))**(1/2)/(h_max),(rho0/9.8)*((1/fs)*(polynas))**(1/2)/(h_min)
    rhomean = (rho0/9.8)*((1/fs)*(polynas))**(1/2)/(np.nanmean((h_max+h_min)/2))
    stratterm = rhomax-rhomin

    rho_s = gsw.beta(salts,-1.9,0)*rho0

    Spolyna = salts + rhomean/rho_s

    Tf = gsw.CT_freezing(salts,np.abs(gldepths)/2,0)
    Tpolyna = -1.9
    D = (1-(Cp/If)*(Tf-(Tpolyna))/4)
    gprimes_cold = (9.8/1027)*(Spolyna*(1-1/D)*rho_s + (stratterm/6))




    bar_x = []
    for k in range(len(labels)):
        if cmask[k]:
            bar_x.append(labels[k])
    plt.close()
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.bar(bar_x,np.asarray(hubdepths)[cmask])
    ax1.set_title("H term")
    ax2.bar(bar_x,np.asarray(glfreezing)[cmask])
    ax2.set_title("Thermal term")
    ax3.bar(bar_x,(gprimes_cold)[cmask])
    ax3.set_title("g' term")
    ax4.bar(bar_x,np.asarray(slopes)[cmask])
    ax4.set_title("slope term")

    ax1.tick_params(axis='x', labelrotation=45)
    ax2.tick_params(axis='x', labelrotation=45)
    ax3.tick_params(axis='x', labelrotation=45)
    ax4.tick_params(axis='x', labelrotation=45)

    plt.show()

def clean_new(cdws,salts,raw_temps,thermals,gprimes,front_spread,h_min,h_max,slopes,volumes,fs,areas,gldepths,hubdepths,mys,sigmas,labels,polynas,polynas_weighted,shelf_class,colorthresh=5,textthresh=5):
    rho0 = 1025
    rhoi = 910
    Cp = 4186
    If = 334000
    W0 =  100000
 
    glfreezing = (-1.9-gsw.CT_freezing(salts,np.abs(gldepths)/2,0))/4
    # areas = np.asarray(areas)
    areas = areas*(10**6)
    wmask = np.asarray((shelf_class))>0#0.0005
    warm_mys=np.asarray(mys)[wmask]
    warmfull = np.asarray(cdws*np.asarray(thermals)*(1/6)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))*np.asarray(areas)
    warm = warmfull[wmask]
    warm_xs = np.asarray(([warm])).reshape((-1, 1))
    warm_model = LinearRegression(fit_intercept=False).fit(warm_xs, warm_mys*areas[wmask])
    warm_melts = warm_model.predict(warm_xs)

    warm_alpha =  (warm_model.coef_/((rho0*Cp)/(rhoi*If*W0)))/(364*24*60*60)
    # warm_alpha =  0.01/(364*24*60*60)

    ######## cold g'


    ##########3
    coldfull = []
    new_shelf_class = []

    for i in range(len(salts)):
        x = Symbol('x',real=True,positive=True)
        meltflux = x*(1/(60*60*24*365))*(920.0)
        Btotal = ((meltflux*areas[i]*34.5))/rho0*9.8*gsw.beta(34.5,-1.8,gldepths[i]/2)-polynas[i]

        meltscalefactor = ((areas[i]*34.5))/rho0*9.8*gsw.beta(34.5,-1.8,gldepths[i]/2)
        print(labels[i]," breakeven melt ",(24*60*60*365)/920*np.mean(polynas[i])/meltscalefactor, "warm melt: ",warmfull[i]*warm_alpha/areas[i])

        rhomin,rhomax = (rho0/9.8)*((1/fs[i])*(Btotal*(9.8/1027)))**(1/2)/(h_max[i]),(rho0/9.8)*((1/fs[i])*(Btotal))**(1/2)/(h_min[i])
        rhomean = (rho0/9.8)*((1/fs[i])*(Btotal))**(1/2)/(np.nanmean((h_max[i]+h_min[i])/2))
        stratterm = rhomax-rhomin

        rho_s = gsw.beta(salts[i],-1.9,0)*rho0

        Spolyna = salts[i] + rhomean/rho_s
        Tf = gsw.CT_freezing(salts[i],np.abs(gldepths[i])/2,0)
        Tpolyna = -1.9
        D = (1-(Cp/If)*(Tf-(Tpolyna))/4)
        gprimes_cold = (9.8/1027)*(Spolyna*(1-1/D)*rho_s + (stratterm/6))

        alpha = 1*((rho0*Cp)/(rhoi*If*W0))

        solveexpr = x - (alpha/(60*60*24*365))*slopes[i]*((Tpolyna-Tf)/4)*hubdepths[i]*fs[i]
        print(labels[i])
        # print(((warmfull[i]*34.5))/rho0*9.8*(7.8*10**(-4))-polynas[i])
        warmalpha = 0.16/((rho0*Cp)/(rhoi*If*W0))
        try:
            disconnectedmelt = float(re(nsolve(solveexpr,x,0)))*(60*60*24*365)
        except:
            disconnectedmelt = np.nan

        
        Bmelt = 34.5*((warmfull[i]*warm_alpha/areas[i])/(60*60*24*365))*920/rho0*9.8*(gsw.beta(34.5,-1.8,gldepths[i]/2))

        if warmfull[i]*warm_alpha/areas[i]-(24*60*60*365)/920*np.mean(polynas[i])/meltscalefactor> 0:
            connectedmelt = warmfull[i] * warm_alpha/areas[i]
        else:
            connectedmelt = np.nan

        if labels[i] == "Getz":
            ipdb.set_trace()
        print(labels[i]," | disconnected: ", disconnectedmelt, " | connected: ",connectedmelt)
        coldfull.append(disconnectedmelt)
        if ~np.isnan(connectedmelt):
            new_shelf_class.append(1)
        else:
            new_shelf_class.append(-1)

    shelf_class = new_shelf_class
    areas = areas*(10**6)
    wmask = np.asarray((shelf_class))>0#0.0005
    warm_mys=np.asarray(mys)[wmask]
    warmfull = np.asarray(cdws*np.asarray(thermals)*(1/6)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))*np.asarray(areas)
    warm = warmfull[wmask]
    warm_xs = np.asarray(([warm])).reshape((-1, 1))
    warm_model = LinearRegression(fit_intercept=False).fit(warm_xs, warm_mys*areas[wmask])
    warm_melts = warm_model.predict(warm_xs)

    warm_alpha =  (warm_model.coef_/((rho0*Cp)/(rhoi*If*W0)))/(364*24*60*60)


    cmask = np.asarray(shelf_class)<0#0.0005
    cold_mys=np.asarray(mys)[cmask]
    coldfull = np.asarray(coldfull)*areas
    coldfull = slopes*(np.abs(hubdepths))*np.asarray(fs)*np.asarray(areas)*glfreezing#*gprimes_cold#*gprimechapman#*glfreezing#*gprimechapman
    cold = coldfull[cmask]
    cold_xs = np.asarray(([cold])).reshape((-1, 1))
    cold_model = LinearRegression(fit_intercept=False).fit(cold_xs, cold_mys*areas[cmask])
    cold_melts = cold_model.predict(cold_xs)

    cold_alpha =  (cold_model.coef_/((rho0*Cp)/(rhoi*If*W0)))/(364*24*60*60)
    print("warm alpha: ",warm_alpha)
    print("cold alpha: ",cold_alpha)






    graymask = np.asarray(shelf_class)==0#0.0005
    graycold_xs = np.asarray(([coldfull[graymask]])).reshape((-1,1))
    graywarm_xs = np.asarray(([warmfull[graymask]])).reshape((-1,1))
    if len(graycold_xs)>0:
        gray_melts = (cold_model.predict(graycold_xs) + warm_model.predict(graywarm_xs))/2.0
    else:
        gray_melts = np.asarray([])
    gray_mys  = np.asarray(mys)[graymask]

    r2 = pearsonr(np.concatenate((cold_melts.flatten(),warm_melts.flatten(),gray_melts.flatten())),np.concatenate((cold_mys.flatten()*areas[cmask],warm_mys.flatten()*areas[wmask],gray_mys.flatten()*areas[graymask]))).statistic**2
    fig, ax = plt.subplots(1,1)

    rhoi = 910
    gigatonconv = 10**(-12)
    scalefactor = rhoi*gigatonconv

    ax.scatter(warm_melts.flatten()*scalefactor,scalefactor*warm_mys*areas[wmask],c="red")
    ax.scatter(cold_melts.flatten()*scalefactor,scalefactor*cold_mys*areas[cmask],c="blue")
    ax.scatter(gray_melts.flatten()*scalefactor,scalefactor*gray_mys*areas[graymask],c="gray")

    cmaskin = np.cumsum(cmask)-1
    wmaskin = np.cumsum(wmask)-1
    graymaskin = np.cumsum(graymask)-1

    sigmas = np.asarray(sigmas)
    for k in range(len(labels)):
        if cmask[k]:
            text=plt.annotate(labels[k],(cold_melts.flatten()[cmaskin[k]]*scalefactor,(scalefactor*cold_mys*areas[cmask])[cmaskin[k]]))
        elif wmask[k]:
            text=plt.annotate(labels[k],(warm_melts.flatten()[wmaskin[k]]*scalefactor,(scalefactor*warm_mys*areas[wmask])[wmaskin[k]]))
        elif graymask[k]:
            text=plt.annotate(labels[k],(gray_melts.flatten()[graymaskin[k]]*scalefactor,(scalefactor*gray_mys*areas[graymask])[graymaskin[k]]))

    markers, caps, bars = ax.errorbar(cold_melts.flatten()*scalefactor,scalefactor*cold_mys*areas[cmask],yerr=sigmas[cmask]*areas[cmask]*scalefactor,ls='none')
    [bar.set_alpha(0.3) for bar in bars]                  
    markers, caps, bars = ax.errorbar(warm_melts.flatten()*scalefactor,scalefactor*warm_mys*areas[wmask],yerr=sigmas[wmask]*areas[wmask]*scalefactor,ls='none')
    [bar.set_alpha(0.3) for bar in bars]                  
    markers, caps, bars = ax.errorbar(gray_melts.flatten()*scalefactor,scalefactor*gray_mys*areas[graymask],yerr=sigmas[graymask]*areas[graymask]*scalefactor,ls='none',ecolor="gray")
    [bar.set_alpha(0.3) for bar in bars]

    ax.text(.05, .95, '$r^2=$'+str(round(r2,2)), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
    ax.set_xlabel(r"$\dot{M}_{\mathrm{pred}} (Gt/yr)$",fontsize=24)
    ax.set_ylabel(r'$\dot{M}_{\mathrm{obs}} (Gt/yr)$',fontsize=24)
    # ax.set_xlim(0,140)
    # ax.set_ylim(0,140)

    plt.show()



 
