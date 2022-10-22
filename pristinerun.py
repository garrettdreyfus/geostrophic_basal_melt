import bathtub as bt
import pickle
import GLIB
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
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

areas = bt.shelf_areas()
#glibheats =  cdw.tempFromClosestPoint(bedmach,grid,physical,depths,closest_points,sal,temp,debug=True)
physical = np.asarray(physical)
#slopes_by_shelf = bt.shelf_sort(shelf_keys,glibheats)
gldepths_by_shelf = bt.shelf_sort(shelf_keys,depths)
glibs_by_shelf = bt.shelf_sort(shelf_keys,glibs)
rignot_shelf_massloss,rignot_shelf_areas,sigma = cdw.extract_rignot_massloss("data/rignot2019.xlsx")

#with open("data/hub_shelf_massloss.pickle","wb") as f:
    #pickle.dump(slopes_by_shelf,f)
with open("data/hub_shelf_massloss.pickle","rb") as f:
    slopes_by_shelf = pickle.load(f)



xs=[]
ys = []
zs = []
znews= [] 
#
#with open("data/MSDS.pickle","rb") as f:
    #MSDS = pickle.load(f)
for k in slopes_by_shelf.keys():
    if k in rignot_shelf_massloss:
        x = np.nanmean(slopes_by_shelf[k])
        z = np.nanmean(gldepths_by_shelf[k])
        c = (rignot_shelf_massloss[k])/np.sqrt(areas[k])#len(slopes_by_shelf[k])#/rignot_shelf_areas[k]
        xs.append(x)
        ys.append(c)
        zs.append(np.sqrt(areas[k]))
        znews.append(np.nanmean(glibs_by_shelf[k]))
        plt.annotate(k,(x,c))

plt.xlabel("Average degrees C above freezing within 200m above HUB")
plt.ylabel("Rignot 2019 massloss divided by grounding line length")
plt.scatter(xs,ys,c=zs,cmap="jet")
plt.colorbar()
plt.show()


