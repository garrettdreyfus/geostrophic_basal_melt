import bathtub as bt
import shapely
import pickle
import GLIB
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
#isopycnaldepths = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="isopycnaldepth")
physical = np.asarray(physical)
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
#rignot_shelf_massloss,rignot_shelf_areas,sigma = cdw.extract_rignot_massloss2019("data/rignot2019.xlsx")
#rignot_shelf_massloss,sigmas = cdw.extract_rignot_massloss2013("data/rignot2013.xlsx")
#rignot_shelf_massloss = bt.shelf_mass_loss('data/amundsilli.h5',polygons)
#with open("data/new_massloss.pickle","wb") as f:
    #pickle.dump(rignot_shelf_massloss,f)
with open("data/new_massloss.pickle","rb") as f:
    rignot_shelf_massloss = pickle.load(f)

#with open("data/hub_shelf_massloss.pickle","wb") as f:
    #pickle.dump(slopes_by_shelf,f)
#with open("data/hub_shelf_massloss.pickle","rb") as f:
    #slopes_by_shelf = pickle.load(f)
#
#
##with open("data/cdwdiff_hub_shelf_massloss.pickle","wb") as f:
    ##pickle.dump(slopes_by_shelf,f)
#with open("data/cdwdiff_hub_shelf_massloss.pickle","rb") as f:
    #slopes_by_shelf = pickle.load(f)

thermals=[]
cdws = []
isopycnals = []
msds = []
ys = []
gldepths = []
glibshelf=[]


znews= [] 
bars = []
areas = []
mys = []
#
##
##with open("data/MSDS.pickle","rb") as f:
    ##MSDS = pickle.load(f)
#
labels = []
for k in slopes_by_shelf.keys():
    if k in rignot_shelf_massloss and ~np.isnan(rignot_shelf_massloss[k]):
        labels.append(k)
        thermals.append(np.nanmean(slopes_by_shelf[k]))
        cdws.append(np.nanmean(cdws_by_shelf[k]))
        isopycnals.append(np.nanmean(isopycnaldepth_by_shelf[k]))
        gldepths.append(np.nanmean(gldepths_by_shelf[k]))
        glibshelf.append(np.nanmean(glibs_by_shelf[k]))
        #bars.append(sigmas[k])
        areas.append(shelf_areas[k])
        mys.append(rignot_shelf_massloss[k])
plt.scatter(np.asarray(isopycnals)*np.asarray(thermals),mys,c=np.asarray(thermals))
#plt.scatter(np.asarray(isopycnals),np.asarray(thermals),c=np.asarray(thermals))
#plt.scatter(np.asarray(thermals),mys,c=np.asarray(isopycnals))
plt.show()
plt.scatter(np.exp(np.asarray(thermals)),mys)
plt.show()
#
#plt.xlabel("Average degrees C above freezing within 200m above HUB")
#plt.ylabel("Rignot 2019 massloss divided by grounding line length")
##plt.scatter(xs,ys,c=zs,cmap="jet")
#print("here")
if True:
    ##xs = np.asarray([np.asarray(np.sqrt(areas)),np.asarray(thermals)**2])
    #xs = np.asarray(([np.asarray(thermals)*np.abs(thermals),np.asarray(np.sqrt(areas)),(np.asarray(thermals)**2)*np.asarray(np.sqrt(areas))]))
    print(thermals)
    #xs = np.asarray(([np.exp(thermals),np.exp(thermals)])).T
    xs = np.asarray(([np.asarray(thermals)*np.asarray(isopycnals),thermals*np.asarray(isopycnals)]))
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
    w = np.asarray(np.sqrt(areas))
    s = np.arange(len(areas))
    labels=np.asarray(labels)[s]
    #bars=np.asarray(bars)[s]
    mys=np.asarray(mys)[s]
    xs = np.asarray(([np.asarray(thermals)[s]**(2),np.sqrt(areas)[s],np.asarray(np.asarray(winds)*-1*np.abs(winds))[s],polyna[s]]))
    scaler = preprocessing.StandardScaler().fit(xs.T)
    xs = scaler.transform(xs.T)
    print(xs)
    print(xs.shape)
    #model = RandomForestRegressor(max_depth=5).fit(xs[:,::3].T, np.asarray(mys)[::3])
    model = RandomForestRegressor(max_depth=3).fit(xs[::2,:], np.asarray(mys)[::2])
    print(model.feature_importances_)
    print(2)
    #r_sq = model.coef_#score(xs.reshape((-1, 1)), ys)
    #xs = np.asarray(xs)/r_sq
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
