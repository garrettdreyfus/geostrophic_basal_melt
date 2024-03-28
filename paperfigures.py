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


def param_vs_melt_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels,xlim=30,ylim=30,colorthresh=5,textthresh=5):
    melts = np.asarray(cdws*np.asarray(thermals)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs))
    #melts = np.asarray(slopes)
    mys=np.asarray(mys)
    xs = np.asarray(([melts])).reshape((-1, 1))
    model = LinearRegression().fit(xs, mys)
    r2 = model.score(xs,mys)
    rho0 = 1025
    rhoi = 910
    Cp = 4186
    If = 334000
    C = model.coef_
    W0 =  100000
    alpha =  (C/((rho0*Cp)/(rhoi*If*W0)))/(364*24*60*60)
    print('alpha: ', alpha)
    plt.rc('axes', titlesize=24)     # fontsize of the axes title
    xs = np.asarray(([melts])).reshape((-1, 1))
    model = LinearRegression().fit(xs, mys)
    r2 = model.score(xs,mys)
    melts = model.predict(xs)
    ax = plt.gca()
    ax.scatter(melts[melts<colorthresh],mys[melts<colorthresh],c="blue")
    ax.scatter(melts[melts>colorthresh],mys[melts>colorthresh],c="red")
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
    ax.text(.05, .95, '$r^2=$'+str(round(r2,2)), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
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
    model = LinearRegression().fit(xs, mys)
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
    ax.text(.05, .95, '$r^2=$'+str(round(r2,2)), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
    ax.set_xlabel(xlabel,fontsize=24)
    ax.set_ylabel(r'$\dot{m}_{\mathrm{obs}} (m/yr)$',fontsize=24)
    plt.show()

