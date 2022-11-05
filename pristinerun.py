import bathtub as bt
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

# Create GLIB

writeBedMach = False
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

################################################

################################################

if writeGL:
    physical, grid, depths, shelves, shelf_keys = bt.get_line_points(bedmach,polygons)
    with open("data/groundinglinepoints.pickle","wb") as f:
        pickle.dump([physical,grid,depths,shelves,shelf_keys],f)

with open("data/groundinglinepoints.pickle","rb") as f:
    physical,grid,depths,shelves,shelf_keys = pickle.load(f)

# randomcmap = ListedColormap(np.random.rand ( 256,3))
# plt.imshow(GLIB,cmap=randomcmap)
# grid = np.asarray(grid)
# print(grid.shape)
# shelf_keys =np.asarray(shelf_keys)
# plt.scatter(grid[shelf_keys=="Frost"][1],grid[shelf_keys=="Frost"][0],c="red",s=100)
# plt.scatter(grid.T[1][shelf_keys=="Frost"],grid.T[0][shelf_keys=="Frost"],c="red",s=100)
# plt.show()


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
glibheats = np.asarray([0]*len(physical))# cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys)
glibheats = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys)
#glibheats = cdw.tempFromClosestPointSimple(bedmach,grid,physical,depths,physical,sal,temp,shelf_keys)
physical = np.asarray(physical)
slopes_by_shelf = bt.shelf_sort(shelf_keys,glibheats)
gldepths_by_shelf = bt.shelf_sort(shelf_keys,depths)
glibs_by_shelf = bt.shelf_sort(shelf_keys,glibs)
#rignot_shelf_massloss,rignot_shelf_areas,sigma = cdw.extract_rignot_massloss2019("data/rignot2019.xlsx")
rignot_shelf_massloss,sigmas = cdw.extract_rignot_massloss2013("data/rignot2013.xlsx")

#with open("data/hub_shelf_massloss.pickle","wb") as f:
    #pickle.dump(slopes_by_shelf,f)
with open("data/hub_shelf_massloss.pickle","rb") as f:
    slopes_by_shelf = pickle.load(f)


#with open("data/cdwdiff_hub_shelf_massloss.pickle","wb") as f:
    #pickle.dump(slopes_by_shelf,f)
#with open("data/cdwdiff_hub_shelf_massloss.pickle","rb") as f:
    #slopes_by_shelf = pickle.load(f)



thermals=[]
msds = []
ys = []
gldepths = []
glibshelf=[]


znews= [] 
bars = []
areas = []

mys = []

#
#with open("data/MSDS.pickle","rb") as f:
    #MSDS = pickle.load(f)

labels = []
for k in slopes_by_shelf.keys():
    if k in rignot_shelf_massloss:
        labels.append(k)
        thermals.append(np.nanmean(slopes_by_shelf[k]))
        gldepths.append(np.nanmean(gldepths_by_shelf[k]))
        glibshelf.append(np.nanmean(glibs_by_shelf[k]))
        bars.append(sigmas[k])
        areas.append(shelf_areas[k])
        mys.append(rignot_shelf_massloss[k])

plt.xlabel("Average degrees C above freezing within 200m above HUB")
plt.ylabel("Rignot 2019 massloss divided by grounding line length")
#plt.scatter(xs,ys,c=zs,cmap="jet")
print("here")
if True:
    #xs = np.asarray([np.asarray(np.sqrt(areas)),np.asarray(thermals)**2])
    xs = np.asarray(([np.asarray(thermals)**2,np.asarray(np.sqrt(areas)),(np.asarray(thermals)**2)*np.asarray(np.sqrt(areas))]))
    scaler = preprocessing.StandardScaler().fit(xs.T)
    xs = scaler.transform(xs.T)
    print(xs)
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
    s = np.argsort(w)
    labels=np.asarray(labels)[s]
    bars=np.asarray(bars)[s]
    mys=np.asarray(mys)[s]
    xs = np.asarray(([np.asarray(thermals)[s]**(2),areas]))
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
    print(3)
if False:
    xs = np.asarray(([np.asarray(np.sqrt(areas)),np.asarray(thermals)**(3/2)]))
    print(xs.shape)
    model = DecisionTreeRegressor(max_depth=2).fit(xs.T, mys)
    tree.plot_tree(model)
    plt.show()
    print(2)
    #r_sq = model.coef_#score(xs.reshape((-1, 1)), ys)
    #xs = np.asarray(xs)/r_sq
    xs = model.predict(xs.T)
    print(3)
print(xs)


plt.errorbar(xs,mys,yerr=bars,fmt="o")
plt.plot(range(30),range(30))
for i in range(len(xs)):
    plt.annotate(labels[i],(xs[i],mys[i]))
#plt.colorbar()
plt.show()


