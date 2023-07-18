import bathtub as bt
import shapely
import pickle
import GLIB
import time,gsw,xarray, pyproj
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn import preprocessing
from sklearn import tree
import numpy as np
from sklearn.metrics import r2_score
import woa
import matplotlib.colors as colors
import cdw
from tqdm import tqdm
import xarray as xr
import winds as wind

# Create GLIB

writeBedMach = False
writeShelfNumbers = False
writeGLIB = False
writePolygons = False
writeGL =False
createWOA = False
createClosestWOA = False

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

#rignot_shelf_massloss =  bt.shelf_mass_loss("",polygons)

#with open("data/onlypositivemelt.pickle","wb") as f:
    #pickle.dump(rignot_shelf_massloss,f)

###############################################
if writeBedMach:
    print("Grabbing BedMachine Data")
    bedmach = bt.convert_bedmachine("data/BedMachine.nc",coarsenfact=1)
    with open("data/bedmach.pickle","wb") as f:
        pickle.dump(bedmach,f)

with open("data/bedmach.pickle","rb") as f:
    bedmach = pickle.load(f)

## Load these into memory just once for speed
bedvalues = bedmach.bed.values
icemask = bedmach.icemask_grounded_and_shelves.values

#GLIBregions = GLIB.generateGLIBsLabel(bedvalues,icemask)
##with open("data/bedmachGLIBregions.pickle","wb") as f:
    #pickle.dump(GLIBregions,f)
with open("data/bedmachGLIBregions.pickle","rb") as f:
    GLIBregions = pickle.load(f)
##################################################

##################################################
if writeGLIB:
    print("Calculating Glibs")
    GLIB = GLIB.generateGLIBs(bedvalues,icemask)

    with open("data/bedmachGLIB.pickle","wb") as f:
        pickle.dump(GLIB,f)


with open("data/bedmachGLIB.pickle","rb") as f:
    GLIB = pickle.load(f)

#cmap = ListedColormap ( np.random.rand ( 256,3))
#plt.imshow(GLIB,cmap=cmap)
#plt.show()



##################################################

##################################################

if writePolygons:
    bt.save_polygons()

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)


##################################################

##################################################

if writeShelfNumbers:
    with open("data/shelfnumbers.pickle","wb") as f:
        pickle.dump(bt.shelf_numbering(polygons,bedmach),f)
with open("data/shelfnumbers.pickle","rb") as f:
    shelf_number_labels, shelf_numbers = pickle.load(f)
    

################################################

################################################

if writeGL:
    physical, grid, depths, shelves, shelf_keys = bt.get_line_points(bedmach,polygons)
    with open("data/groundinglinepoints.pickle","wb") as f:
        pickle.dump([physical,grid,depths,shelves,shelf_keys],f)

with open("data/groundinglinepoints.pickle","rb") as f:
    physical,grid,depths,shelves,shelf_keys = pickle.load(f)

################################################

################################################

baths = []
for l in range(len(grid)):
    baths.append(GLIB[grid[l][0]][grid[l][1]])

################################################

################################################

if createWOA:
    sal,temp = woa.create_WOA(bedmach)
    with open("data/summerwoa.pickle","wb") as f:
        pickle.dump([sal,temp],f)

with open("data/summerwoa.pickle","rb") as f:
    sal,temp = pickle.load(f)

if createClosestWOA:
    zerobaths = np.full_like(baths,0)
    closest_points = cdw.closest_WOA_points(grid,zerobaths,bedmach,method="bfs")
    with open("data/closest_points_ZERO.pickle","wb") as f:
        pickle.dump(closest_points,f)
with open("data/closest_points.pickle","rb") as f:
    closest_points = pickle.load(f)


depths = []
glibs = []
for l in range(len(baths)):
    depths.append(bedvalues[grid[l][0],grid[l][1]])
    glibs.append(baths[l])
    if np.isnan(baths[l]):
        baths[l]=bedvalues[grid[l][0],grid[l][1]]

shelf_areas = bt.shelf_areas()
#physical_ice, _, _, _, shelf_keys_edge = bt.get_line_points(bedmach,polygons,mode="edge")

#with open("data/physical_ice.pickle","wb") as f:
    #pickle.dump((physical_ice,shelf_keys_edge),f)
with open("data/physical_ice.pickle","rb") as f:
    physical_ice,shelf_keys_edge = pickle.load(f)

#slopes = cdw.slopeFromClosestPoint(bedmach,physical_ice,grid,physical,depths,closest_points,shelf_keys,shelf_keys_edge)
#glibheats = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="glibheat",debug=False)
#layertemp = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="newtemp")
#hubsalts = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="glibheat")
#gprimes = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="gprime")
#cdwdepths = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="cdwdepth",debug=True)
#distances = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="distance")
#isopycnaldepths = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="thermocline",shelfkeys=shelf_keys)
#physical = np.asarray(physical)
#grid = np.asarray(grid)
#plt.scatter(grid.T[0][40000:80000],grid.T[1][40000:80000],c=range(grid.shape[0])[40000:80000])
#plt.colorbar()
#plt.scatter(grid.T[0][40000:80000],grid.T[1][40000:80000],c="red")
#for l in range(40000,80000,100):
    #plt.annotate(l,(grid.T[0,l],grid.T[1,l]))
#plt.show()
glibheats = cdw.closestMethodologyFig(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="glibheat",debug=False)
exit()
##with open("data/distances.pickle","wb") as f:
    #pickle.dump(distances,f)
with open("data/distances.pickle","rb") as f:
    distances = pickle.load(f)

#with open("data/layertemp.pickle","wb") as f:
    #pickle.dump(layertemp,f)
with open("data/layertemp.pickle","rb") as f:
    layertemp = pickle.load(f)


#with open("data/hub_shelf_thermals.pickle","wb") as f:
    #pickle.dump(glibheats,f)
with open("data/hub_shelf_thermals.pickle","rb") as f:
    glibheats = pickle.load(f)
#glibheats = layertemp
#with open("data/hub_shelf_salts.pickle","wb") as f:
    #pickle.dump(hubsalts,f)
with open("data/hub_shelf_salts.pickle","rb") as f:
    hubsalts = pickle.load(f)

#with open("data/cdwdepths.pickle","wb") as f:
    #pickle.dump(cdwdepths,f)
with open("data/cdwdepths.pickle","rb") as f:
    cdwdepths = pickle.load(f)

#with open("data/gprimes.pickle","wb") as f:
    #pickle.dump(gprimes,f)
with open("data/gprimes.pickle","rb") as f:
    gprimes = pickle.load(f)

#with open("data/isopycnaldepths.pickle","wb") as f:
    #pickle.dump(isopycnaldepths,f)
with open("data/isopycnaldepths.pickle","rb") as f:
    isopycnaldepths = pickle.load(f)

#slopes_by_shelf = cdw.slope_by_shelf(bedmach,polygons)

#with open("data/slopes_by_shelf.pickle","wb") as f:
    #pickle.dump(slopes_by_shelf,f)
with open("data/slopes_by_shelf.pickle","rb") as f:
    slopes_by_shelf = pickle.load(f)

#volumes_by_shelf = cdw.volumes_by_shelf(bedmach,polygons)

#with open("data/volumes_by_shelf.pickle","wb") as f:
    #pickle.dump(volumes_by_shelf,f)
with open("data/volumes_by_shelf.pickle","rb") as f:
    volumes_by_shelf = pickle.load(f)

#polynas = cdw.polyna_bathtub(bedmach,grid,GLIBregions)
#with open("data/polynasbt.pickle","wb") as f:
    #pickle.dump(polynas,f)
with open("data/polynas.pickle","rb") as f:
    polynas = pickle.load(f)

#polynas = cdw.polyna_bathtub(bedmach,grid,GLIBregions)
#with open("data/polynasbt.pickle","wb") as f:
    #pickle.dump(polynas,f)
#with open("data/polynas.pickle","rb") as f:
    #polynas = pickle.load(f)


#glibheats = np.asarray(glibheats)*np.asarray(isopycnaldepths)

projection = pyproj.Proj("epsg:3031")
fs = []
for x,y in physical:
        lon,lat = projection(x,y,inverse=True)
        fs.append(1/np.abs(gsw.f(lat)))

polynas=np.asarray(polynas)

fs_by_shelf = bt.shelf_sort(shelf_keys,fs)
hubheats_by_shelf = bt.shelf_sort(shelf_keys,np.asarray(glibheats))
salts_by_shelf = bt.shelf_sort(shelf_keys,hubsalts)
polyna_by_shelf = bt.shelf_sort(shelf_keys,polynas)
#slopes_by_shelf = bt.shelf_sort(shelf_keys,slopes)
cdws_by_shelf = bt.shelf_sort(shelf_keys,cdwdepths)
gprimes_by_shelf = bt.shelf_sort(shelf_keys,gprimes)
isopycnaldepth_by_shelf = bt.shelf_sort(shelf_keys,isopycnaldepths)
gldepths_by_shelf = bt.shelf_sort(shelf_keys,depths)
glibs_by_shelf = bt.shelf_sort(shelf_keys,glibs)
distances_by_shelf = bt.shelf_sort(shelf_keys,glibs)
hubdeltas_by_shelf = bt.shelf_sort(shelf_keys,np.asarray(depths)-np.asarray(baths))
winds_by_shelf = wind.AMPS_wind(polygons,"data/AMPS_winds.mat",icemask)
#rignot_shelf_massloss,rignot_shelf_areas,sigma = cdw.extract_rignot_massloss2019("data/rignot2019.xlsx")
#rignot_shelf_massloss,sigmas_by_shelf = cdw.extract_rignot_massloss2013("data/rignot2013.xlsx")
#rignot_shelf_massloss = bt.shelf_mass_loss('data/amundsilli.h5',polygons)
#with open("data/new_massloss.pickle","wb") as f:
    #pickle.dump(rignot_shelf_massloss,f)
#with open("data/new_massloss.pickle","rb") as f:
    #rignot_shelf_massloss = pickle.load(f)
rignot_shelf_massloss,sigmas_by_shelf =  cdw.extract_adusumilli("data/Adusumilli.csv")
#rignot_shelf_massloss =  bt.shelf_mass_loss("",polygons)

#with open("data/onlypositivemelt.pickle","wb") as f:
    #pickle.dump(rignot_shelf_massloss,f)
#with open("data/onlypositivemelt.pickle","rb") as f:
    #rignot_shelf_massloss = pickle.load(f)



#print(rignot_shelf_massloss.keys())


#with open("data/hub_shelf_massloss.pickle","wb") as f:
    #pickle.dump(slopes_by_shelf,f)
#with open("data/hub_shelf_massloss.pickle","rb") as f:
    #rignot_shelf_massloss = pickle.load(f)
#
#
##with open("data/cdwdiff_hub_shelf_massloss.pickle","wb") as f:
    ##pickle.dump(slopes_by_shelf,f)
#with open("data/cdwdiff_hub_shelf_massloss.pickle","rb") as f:
    #slopes_by_shelf = pickle.load(f)

thermals=[]
cdws = []
isopycnals = []
ys = []
gldepths = []
glibshelf=[]
gprimes=[]
distances = []
hubdeltas = []
polynas = []
gllengths=[]
winds = []
salts = []


znews= [] 
bars = []
areas = []
mys = []
slopes = []
volumes = []
fs = []
sigmas = []
#
##
##with open("data/MSDS.pickle","rb") as f:
    ##MSDS = pickle.load(f)
#
labels = []
for k in slopes_by_shelf.keys():
    if k in rignot_shelf_massloss and ~np.isnan(rignot_shelf_massloss[k]):
        x,y = (polygons[k][0].centroid.x,polygons[k][0].centroid.y)
        slopes.append(slopes_by_shelf[k])
        volumes.append(volumes_by_shelf[k])
        labels.append(k)
        fs.append(np.nanmean(fs_by_shelf[k]))
        thermals.append(np.nanmean(hubheats_by_shelf[k]))
        cdws.append(np.nanmean(cdws_by_shelf[k]))
        isopycnals.append(np.nanmean(isopycnaldepth_by_shelf[k]))
        gldepths.append(np.nanmean(gldepths_by_shelf[k]))
        gprimes.append(np.nanmean(gprimes_by_shelf[k]))
        glibshelf.append(np.nanmean(glibs_by_shelf[k]))
        distances.append(np.nanmean(distances_by_shelf[k]))
        hubdeltas.append(np.nanmean(hubdeltas_by_shelf[k]))
        polynas.append(np.nanmean(polyna_by_shelf[k]))
        salts.append(np.nanmean(salts_by_shelf[k]))
        winds.append(np.nanmean(winds_by_shelf[k]))
        sigmas.append(sigmas_by_shelf[k])
        #bars.append(sigmas[k])
        areas.append(shelf_areas[k])
        mys.append(rignot_shelf_massloss[k])

#plt.scatter(np.asarray(thermals)*np.asarray(cdws)*slopes,mys,c=gldepths)
#plt.errorbar(np.asarray(thermals)*np.asarray(cdws)*slopes,mys,yerr=sigmas,ls='none')
#plt.show()

# plt.scatter(np.asarray(thermals),mys)
# for k in range(len(labels)):
#     plt.annotate(labels[k],(np.asarray(thermals[k]),mys[k]))
#
# plt.show()
#
# plt.scatter(np.asarray(thermals)*np.asarray(cdws)*np.asarray(gprimes)*fs,mys)
# for k in range(len(labels)):
#     plt.annotate(labels[k],(np.asarray(thermals)[k]*np.asarray(cdws)[k]*np.asarray(gprimes)[k]*fs[k],mys[k]))
# plt.show()
#
#
# plt.scatter(np.asarray(thermals)*np.asarray(cdws)*np.asarray(gprimes)*fs*np.asarray(slopes),mys)
# for k in range(len(labels)):
#     plt.annotate(labels[k],(slopes[k]*np.asarray(thermals)[k]*np.asarray(cdws)[k]*np.asarray(gprimes)[k]*fs[k],mys[k]))
# plt.show()
#




if False:
    plt.scatter(np.asarray(thermals),mys)
    plt.errorbar(np.asarray(thermals),mys,yerr=sigmas,ls='none')
    for k in range(len(labels)):
        plt.annotate(labels[k],(np.asarray(thermals[k]),mys[k]))
    plt.xlabel("t")
    plt.show()

    plt.scatter(np.asarray(thermals)*np.asarray(isopycnals),mys)
    plt.errorbar(np.asarray(thermals)*np.asarray(isopycnals),mys,yerr=sigmas,ls='none')
    for k in range(len(labels)):
        plt.annotate(labels[k],(np.asarray(thermals[k])*isopycnals[k],mys[k]))
    plt.xlabel("t * delta H")
    plt.show()

    plt.scatter(np.asarray(thermals)*np.asarray(isopycnals)*slopes,mys)
    plt.errorbar(np.asarray(thermals)*np.asarray(isopycnals)*slopes,mys,yerr=sigmas,ls='none')
    plt.xlabel("t * delta H * slope")
    for k in range(len(labels)):
        plt.annotate(labels[k],(np.asarray(thermals[k])*isopycnals[k]*slopes[k],mys[k]))
    plt.show()

    plt.scatter(np.asarray(thermals)*fs*np.asarray(isopycnals)*slopes,mys)
    plt.errorbar(np.asarray(thermals)*fs*np.asarray(isopycnals)*slopes,mys,yerr=sigmas,ls='none')
    plt.xlabel("t * delta H * slope /f")
    for k in range(len(labels)):
        plt.annotate(labels[k],(np.asarray(thermals[k])*fs[k]*isopycnals[k]*slopes[k],mys[k]))
    plt.show()



#
# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
# ax1.scatter(np.asarray(thermals),mys,c=salts)
# ax1.set_xlabel("Avg temp at HUB")
# ax1.set_ylabel("Melt (m/yr)")
# ax2.scatter(np.asarray(cdws),mys,c=thermals)
# ax2.set_xlabel("Avg depth of thermocline above HUB * delta t")
# ax3.scatter(np.asarray(slopes),mys,c=thermals)
# ax3.set_xlabel("Avg depth of thermocline above HUB * delta t")
#plt.plot(range(30),range(30))
#plt.show()
areas = np.asarray(areas)
melts = cdws*np.asarray(thermals)*np.asarray(fs)*(np.asarray(gprimes))*slopes
melts=melts
mys=mys
#melts = np.asarray(thermals)*slopes
rho0 = 1025
Cp = 3850
spy = (3.154*10**7)
melten = 3.34*10**5
kgtom = 920
C= 0.00002
#plt.scatter(melts*(rho0*spy*Cp*C)/(melten*kgtom),mys)
plt.rc('axes', titlesize=24)     # fontsize of the axes title
xs = np.asarray(([melts])).reshape((-1, 1))
model = LinearRegression().fit(xs, mys)
r2 = model.score(xs,mys)
melts = model.predict(xs)
plt.scatter(melts,mys)
plt.errorbar(melts,mys,yerr=sigmas,ls='none')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.plot(range(30),range(30))
plt.text(.05, .95, '$r^2=$'+str(round(r2,2)), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
plt.xlabel(r"$\dot{m}_{pred} (m/yr)$",fontsize=18)
plt.ylabel(r'$\dot{m}_{obs} (m/yr)$',fontsize=18)
for k in range(len(labels)):
     plt.annotate(labels[k],(melts[k],mys[k]))
plt.show()
xs=model.predict(xs)

if False:
    oldxs = xs
    xs = model.predict(xs)
    tempterms = cdws*np.asarray(thermals)*np.asarray(fs)*np.asarray(gprimes)
    x = np.linspace(np.min(tempterms)*0.95,np.max(tempterms)*1.05,20)
    y = np.linspace(np.min(slopes)*0.95,np.max(slopes)*1.05,20)
    X,Y = np.meshgrid(x,y)
    Z = np.multiply(X,Y)*model.coef_[0]+model.intercept_
    im = plt.pcolormesh(X,Y,Z,cmap="gnuplot",vmin=np.min(Z),vmax=np.max(Z))
    cbar = plt.colorbar(im)
    CS = plt.contour(X,Y,Z,levels=[1,2.5,5,10,15,20],colors="white")
    plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(r"$(T_{CDW}-T_{f})*(H_{PYC}-H_{HUB})*\frac{g'}{f}$",fontsize=18)
    plt.ylabel(r'$S_{ice}$',fontsize=18)
    plt.scatter(tempterms,slopes,c="white")
    for k in range(len(labels)):
        plt.annotate(labels[k],(tempterms[k],slopes[k]),c="white")
    plt.show()

if False:
    xs = np.asarray(([-np.asarray(cdws)*np.asarray(thermals)*np.asarray(slopes),np.asarray(thermals),\
       np.asarrayTrue(polynas),np.asarray(winds),np.asarray(gldepths)\
    ]))
    #scaler = preprocessing.StandardScaler().fit(xs.T)
    #xs = scaler.transform(xs.T)
    xs = xs.T
    model = DecisionTreeRegressor(max_depth=3,min_samples_leaf=10).fit(xs, np.asarray(mys))
    export_graphviz(Truemodel,out_file="tree.dot",feature_names = ["full equation","thermal","polynas","winds","gldepths"])
    print(model.feature_importances_)
    xs = model.predict(xs)
    print("r2,", r2_score(mys,xs))
    print(3)

#
#
#plt.errorbar(xs,mys,yerr=bars,fmt="o")
plt.scatter(xs,mys,c=polynas)
plt.errorbar(xs,mys,yerr=sigmas,ls="none")
for k in range(len(labels)):
    plt.annotate(labels[k],(xs[k],mys[k]))
plt.plot(range(30),range(30))
##plt.colorbar()
plt.show()

xs = np.asarray(xs)
polynas = np.asarray(polynas)
mys = np.asarray(mys)
labels = np.asarray(labels)
gldepths = np.asarray(gldepths)
glibs = np.asarray(glibshelf)
sigmas = np.asarray(sigmas)
slopes = np.asarray(slopes)
areas = np.asarray(areas)
winds = np.asarray(winds)
volumes = np.asarray(volumes)
thermals = np.asarray(thermals)
hubdeltas = np.asarray(hubdeltas)

# plt.scatter(polynas[xs<2.5]/areas[xs<2.5],mys[xs<2.5],c=polynas[xs<2.5])
# plt.errorbar(xs[xs<2.5]/areas[xs<2.5],mys[xs<2.5],yerr=sigmas[xs<2.5],ls="none")
# for k in range(len(labels[xs<2.5])):
#     plt.annotate(labels[xs<2.5][k],(polynas[xs<2.5][k]/areas[xs<2.5][k],mys[xs<2.5][k]))
thresh = 3
freezingtemps = gsw.CT_freezing(34.7,np.abs(gldepths),0)
newxs = ((volumes)*(slopes))[xs<thresh]
plt.scatter(newxs,mys[xs<thresh],c=xs[xs<thresh],cmap="jet")
plt.errorbar(newxs,mys[xs<thresh],yerr=sigmas[xs<thresh],ls="none")
for k in range(len(labels[xs<thresh])):
    plt.annotate(labels[xs<thresh][k],(newxs[k],mys[xs<thresh][k]))
# ##plt.colorbar()
plt.show()
#
#
