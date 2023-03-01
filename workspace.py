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


##################################################

##################################################
if writeGLIB:
    print("Calculating Glibs")
    GLIB = GLIB.generateGLIBs(bedvalues,icemask)

    with open("data/bedmachGLIB.pickle","wb") as f:
        pickle.dump(GLIB,f)


with open("data/bedmachGLIB.pickle","rb") as f:
    GLIB = pickle.load(f)


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
    closest_points = cdw.closest_WOA_points(grid,baths,bedmach)
    with open("data/closest_points.pickle","wb") as f:
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
#glibheats = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="glibheat")
#hubsalts = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="glibheat")
#cdwdepths = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="cdwdepth")
#distances = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="distance")
#isopycnaldepths = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="thermocline",shelfkeys=shelf_keys)
physical = np.asarray(physical)
##with open("data/distances.pickle","wb") as f:
    #pickle.dump(distances,f)
with open("data/distances.pickle","rb") as f:
    distances = pickle.load(f)
#with open("data/hub_shelf_thermals.pickle","wb") as f:
    #pickle.dump(glibheats,f)
with open("data/hub_shelf_thermals.pickle","rb") as f:
    glibheats = pickle.load(f)

#with open("data/hub_shelf_salts.pickle","wb") as f:
    #pickle.dump(hubsalts,f)
with open("data/hub_shelf_salts.pickle","rb") as f:
    hubsalts = pickle.load(f)

##with open("data/cdwdepths.pickle","wb") as f:
    #pickle.dump(cdwdepths,f)
with open("data/cdwdepths.pickle","rb") as f:
    cdwdepths = pickle.load(f)
#with open("data/isopycnaldepths.pickle","wb") as f:
    #pickle.dump(isopycnaldepths,f)
with open("data/isopycnaldepths.pickle","rb") as f:
    isopycnaldepths = pickle.load(f)

#slopes_by_shelf = cdw.slope_by_shelf(bedmach,polygons)

#with open("data/slopes_by_shelf.pickle","wb") as f:
    #pickle.dump(slopes_by_shelf,f)
with open("data/slopes_by_shelf.pickle","rb") as f:
    slopes_by_shelf = pickle.load(f)

#polyna_by_shelf = cdw.polyna_dataset(polygons)
#with open("data/polyna_by_shelf.pickle","wb") as f:
    #pickle.dump(polyna_by_shelf,f)
with open("data/polyna_by_shelf.pickle","rb") as f:
    polyna_by_shelf = pickle.load(f)
#glibheats = np.asarray(glibheats)*np.asarray(isopycnaldepths)

hubheats_by_shelf = bt.shelf_sort(shelf_keys,glibheats)
salts_by_shelf = bt.shelf_sort(shelf_keys,hubsalts)
#slopes_by_shelf = bt.shelf_sort(shelf_keys,slopes)
cdws_by_shelf = bt.shelf_sort(shelf_keys,cdwdepths)
isopycnaldepth_by_shelf = bt.shelf_sort(shelf_keys,isopycnaldepths)
gldepths_by_shelf = bt.shelf_sort(shelf_keys,depths)
glibs_by_shelf = bt.shelf_sort(shelf_keys,glibs)
distances_by_shelf = bt.shelf_sort(shelf_keys,glibs)
hubdeltas_by_shelf = bt.shelf_sort(shelf_keys,np.asarray(depths)-np.asarray(baths))
winds_by_shelf = wind.AMPS_wind(polygons,"data/AMPS_winds.mat",icemask)
#rignot_shelf_massloss,rignot_shelf_areas,sigma = cdw.extract_rignot_massloss2019("data/rignot2019.xlsx")
#rignot_shelf_massloss,sigmas = cdw.extract_rignot_massloss2013("data/rignot2013.xlsx")
#rignot_shelf_massloss = bt.shelf_mass_loss('data/amundsilli.h5',polygons)
#with open("data/new_massloss.pickle","wb") as f:
    #pickle.dump(rignot_shelf_massloss,f)
#with open("data/new_massloss.pickle","rb") as f:
    #rignot_shelf_massloss = pickle.load(f)

rignot_shelf_massloss,sigmas_by_shelf =  cdw.extract_adusumilli("data/Adusumilli.csv")
print(rignot_shelf_massloss.keys())


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
distances = []
hubdeltas = []
polynas = []
winds = []
salts = []


znews= [] 
bars = []
areas = []
mys = []
slopes = []
fs = []
sigmas = []
#
##
##with open("data/MSDS.pickle","rb") as f:
    ##MSDS = pickle.load(f)
#
labels = []
projection = pyproj.Proj("epsg:3031")
for k in slopes_by_shelf.keys():
    if k in rignot_shelf_massloss and ~np.isnan(rignot_shelf_massloss[k]):
        x,y = (polygons[k][0].centroid.x,polygons[k][0].centroid.y)
        lon,lat = projection(x,y,inverse=True)
        fs.append(1/np.abs(gsw.f(lat)))
        slopes.append(slopes_by_shelf[k])
        labels.append(k)
        thermals.append(np.nanmean(hubheats_by_shelf[k]))
        cdws.append(np.nanmean(cdws_by_shelf[k]))
        isopycnals.append(np.nanmean(isopycnaldepth_by_shelf[k]))
        gldepths.append(np.nanmean(gldepths_by_shelf[k]))
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

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
ax1.scatter(np.asarray(thermals),mys,c=salts)
ax1.set_xlabel("Avg temp at HUB")
ax1.set_ylabel("Melt (m/yr)")
ax2.scatter(np.asarray(cdws),mys,c=thermals)
ax2.set_xlabel("Avg depth of thermocline above HUB * delta t")
ax3.scatter(np.asarray(slopes),mys,c=thermals)
ax3.set_xlabel("Avg depth of thermocline above HUB * delta t")
ax4.scatter(np.asarray(thermals)*fs*np.asarray(cdws)*slopes,mys,c=gldepths)
ax4.errorbar(np.asarray(thermals)*fs*np.asarray(cdws)*slopes,mys,yerr=sigmas,ls='none')
ax4.set_xlabel("Avg depth of thermocline * avg temp above HUB * delta t")
for k in range(len(labels)):
    ax1.annotate(labels[k],(thermals[k],mys[k]))
    ax2.annotate(labels[k],(np.asarray(cdws[k]),mys[k]))
    ax3.annotate(labels[k],(np.asarray(slopes[k]),mys[k]))
    ax4.annotate(labels[k],(np.asarray(thermals[k])*fs[k]*cdws[k]*slopes[k],mys[k]))
plt.show()
if True:
    ##xs = np.asarray([np.asarray(np.sqrt(areas)),np.asarray(thermals)**2])
    #xs = np.asarray(([np.asarray(thermals)*np.abs(thermals),np.asarray(np.sqrt(areas)),(np.asarray(thermals)**2)*np.asarray(np.sqrt(areas))]))
    print(thermals)
    #xs = np.asarray(([np.exp(thermals),np.exp(thermals)])).T
    xs = np.asarray(([-np.asarray(cdws)*slopes*np.asarray(thermals)*fs,np.asarray(cdws)*slopes*np.asarray(thermals)*fs]))
    #xs = np.asarray(([np.asarray(thermals),thermals]))
    scaler = preprocessing.StandardScaler().fit(xs.T)
    xs = scaler.transform(xs.T)
    print(mys)
    model = LinearRegression().fit(xs, mys)
    #r_sq = model.coef_#score(xs.reshape((-1, 1)), ys)
    print(model.coef_)
    #xs = np.asarray(xs)/r_sq
    xs = model.predict(xs)
    print("RMSE",np.sqrt(np.mean((mys-xs)**2)))
    print("RMSE",np.sqrt(np.mean(((mys-xs)/mys)**2)))
    print("r2,", r2_score(mys,xs))

if False:
    xs = np.asarray(([-np.asarray(cdws)*np.asarray(thermals)*np.asarray(slopes),np.asarray(thermals),\
       np.asarray(polynas),np.asarray(winds),np.asarray(gldepths)\
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
plt.scatter(xs,mys)
plt.errorbar(xs,mys,yerr=sigmas,ls="none")
for k in range(len(labels)):
    plt.annotate(labels[k],(xs[k],mys[k]))
plt.plot(range(30),range(30))
##plt.colorbar()
plt.show()
#
#
