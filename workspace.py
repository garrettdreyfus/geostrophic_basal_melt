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
    with open("data/woawithbed.pickle","wb") as f:
        pickle.dump([sal,temp],f)

with open("data/woawithbed.pickle","rb") as f:
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
#glibheats = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="glibheat")
#cdwdepths = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="cdwdepth")
#distances = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="distance")
#isopycnaldepths = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="isopycnaldepth",shelfkeys=shelf_keys)
physical = np.asarray(physical)
##with open("data/distances.pickle","wb") as f:
    #pickle.dump(distances,f)
with open("data/distances.pickle","rb") as f:
    distances = pickle.load(f)
#with open("data/hub_shelf_thermals.pickle","wb") as f:
    #pickle.dump(glibheats,f)
with open("data/hub_shelf_thermals.pickle","rb") as f:
    glibheats = pickle.load(f)
#with open("data/cdwdepths.pickle","wb") as f:
    #pickle.dump(cdwdepths,f)
with open("data/cdwdepths.pickle","rb") as f:
    cdwdepths = pickle.load(f)
#with open("data/isopycnaldepths.pickle","wb") as f:
    #pickle.dump(isopycnaldepths,f)
with open("data/isopycnaldepths.pickle","rb") as f:
    isopycnaldepths = pickle.load(f)

#glibheats = np.asarray(glibheats)*np.asarray(isopycnaldepths)

slopes_by_shelf = bt.shelf_sort(shelf_keys,glibheats)
cdws_by_shelf = bt.shelf_sort(shelf_keys,cdwdepths)
isopycnaldepth_by_shelf = bt.shelf_sort(shelf_keys,isopycnaldepths)
gldepths_by_shelf = bt.shelf_sort(shelf_keys,depths)
glibs_by_shelf = bt.shelf_sort(shelf_keys,glibs)
distances_by_shelf = bt.shelf_sort(shelf_keys,glibs)
hubdeltas_by_shelf = bt.shelf_sort(shelf_keys,np.asarray(depths)-np.asarray(baths))
#rignot_shelf_massloss,rignot_shelf_areas,sigma = cdw.extract_rignot_massloss2019("data/rignot2019.xlsx")
rignot_shelf_massloss,sigmas = cdw.extract_rignot_massloss2013("data/rignot2013.xlsx")
#rignot_shelf_massloss = bt.shelf_mass_loss('data/amundsilli.h5',polygons)
#with open("data/new_massloss.pickle","wb") as f:
    #pickle.dump(rignot_shelf_massloss,f)
with open("data/new_massloss.pickle","rb") as f:
    rignot_shelf_massloss = pickle.load(f)
with open("data/polyna_by_shelf.pickle","rb") as f:
    polyna_by_shelf = pickle.load(f)

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


znews= [] 
bars = []
areas = []
mys = []
fs = []
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

        labels.append(k)
        thermals.append(np.nanmean(slopes_by_shelf[k]))
        cdws.append(np.nanmean(cdws_by_shelf[k]))
        isopycnals.append(np.nanmean(isopycnaldepth_by_shelf[k]))
        gldepths.append(np.nanmean(gldepths_by_shelf[k]))
        glibshelf.append(np.nanmean(glibs_by_shelf[k]))
        distances.append(np.nanmean(distances_by_shelf[k]))
        hubdeltas.append(np.nanmean(hubdeltas_by_shelf[k]))
        polynas.append(np.nanmean(polyna_by_shelf[k]))
        #bars.append(sigmas[k])
        areas.append(shelf_areas[k])
        mys.append(rignot_shelf_massloss[k])

fig, (ax1,ax2,ax3) = plt.subplots(1,3)
ax1.scatter(np.asarray(thermals),mys,c=glibshelf)
ax1.set_xlabel("Avg temp at HUB")
ax1.set_ylabel("Melt (m/yr)")
ax2.scatter(-np.asarray(cdws),mys,c=glibshelf)
ax2.set_xlabel("Avg depth of thermocline above HUB * delta t")
ax3.scatter(np.asarray(thermals)*-np.asarray(cdws),mys,c=glibshelf)
ax3.set_xlabel("Avg depth of thermocline * avg temp above HUB * delta t")
print(r2_score(mys,np.asarray(cdws)*thermals))
for k in range(len(labels)):
    ax1.annotate(labels[k],(thermals[k],mys[k]))
    ax2.annotate(labels[k],(-cdws[k],mys[k]))
    ax3.annotate(labels[k],(-np.asarray(thermals[k])*cdws[k],mys[k]))
plt.show()
plt.scatter(np.asarray(cdws),np.asarray(thermals),c=np.asarray(thermals))
#plt.scatter(np.asarray(thermals),mys,c=np.asarray(isopycnals))
plt.show()
#plt.scatter(np.exp(np.asarray(thermals)),mys)
#plt.show()

if True:
    ##xs = np.asarray([np.asarray(np.sqrt(areas)),np.asarray(thermals)**2])
    #xs = np.asarray(([np.asarray(thermals)*np.abs(thermals),np.asarray(np.sqrt(areas)),(np.asarray(thermals)**2)*np.asarray(np.sqrt(areas))]))
    print(thermals)
    #xs = np.asarray(([np.exp(thermals),np.exp(thermals)])).T
    xs = np.asarray(([np.asarray(cdws)*np.asarray(thermals),np.asarray(cdws)*np.asarray(thermals)]))
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
    xs = np.asarray(([np.asarray(isopycnals)*np.asarray(thermals),np.asarray(thermals),\
       np.asarray(hubdeltas),np.asarray(hubdeltas)/np.asarray(distances),np.asarray(polynas)\
    ]))
    scaler = preprocessing.StandardScaler().fit(xs.T)
    xs = scaler.transform(xs.T)
    model = RandomForestRegressor(max_depth=10).fit(xs, np.asarray(mys))
    print(model.feature_importances_)
    print(2)
    xs = model.predict(xs)
    print("r2,", r2_score(mys,xs))
    print(3)

#
#
#plt.errorbar(xs,mys,yerr=bars,fmt="o")
plt.scatter(xs,mys)
plt.plot(range(30),range(30))
##plt.colorbar()
plt.show()
#
#
