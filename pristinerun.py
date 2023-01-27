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
    closest_points = cdw.closest_WOA_points(grid,baths,bedmach,method="simple")
    with open("data/closest_points.pickle","wb") as f:
        pickle.dump(closest_points,f)
with open("data/closest_points.pickle","rb") as f:
    closest_points = pickle.load(f)


depths = []
glibs = []
for l in range(len(baths)):
    depths.append(bedvalues[grid[l][0],grid[l][1]])
    glibs.append(baths[l])
    #if np.isnan(baths[l]):
        #baths[l]=bedvalues[grid[l][0],grid[l][1]]

with open("data/MSDGLIB.pickle","rb") as f:
    glibs = pickle.load(f)


areas = bt.shelf_areas()

baths=glibs
#
glibheats =  cdw.tempFromClosestPoint(bedmach,grid,physical,baths,closest_points,sal,temp)
glibheats_by_shelf = bt.shelf_sort(shelf_keys,glibheats)

glheats =  cdw.tempFromClosestPoint(bedmach,grid,physical,depths,grid,sal,temp)
glheats_by_shelf = bt.shelf_sort(shelf_keys,glheats)
with open("data/shelf_massloss.pickle","wb") as f:
    pickle.dump([glibheats_by_shelf,glheats_by_shelf],f)
with open("data/shelf_massloss.pickle","rb") as f:
    hubheats_by_shelf,glheats_by_shelf = pickle.load(f)

physical = np.asarray(physical)
gldepths_by_shelf = bt.shelf_sort(shelf_keys,depths)
glibs_by_shelf = bt.shelf_sort(shelf_keys,glibs)
rignot_shelf_massloss,rignot_shelf_areas,sigma = cdw.extract_rignot_massloss("data/rignot2019.xlsx")


hubheats=[]
glheats = []
ys = []
zs = []
znews= [] 
melts = []

with open("data/MSDS.pickle","rb") as f:
    MSDS = pickle.load(f)

winds_by_shelf = winds.AMPS_wind(polygons,"data/AMPS_winds.mat",icemask)

fig, (ax1,ax2) = plt.subplots(1,2)
for k in hubheats_by_shelf.keys():
    if k in rignot_shelf_massloss:
        hubheat = np.nanmean(hubheats_by_shelf[k])
        glheat = np.nanmean(glheats_by_shelf[k])
        melt = rignot_shelf_massloss[k]/np.sqrt(areas[k])
        hubheats.append(hubheat)
        glheats.append(glheat)
        melts.append(melt)
        ax1.annotate(k,(hubheat,melt))
        ax2.annotate(k,(glheat,melt))

ax1.scatter(hubheats,melts)
ax2.scatter(glheats,melts)
plt.show()
