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
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import r2_score
import woa
import cdw
from scipy import interpolate
from tqdm import tqdm
import xarray as xr
import winds as wind

# Create GLIB

writeBedMach = False
writeGLIB = False
writePolygons = False
writeGL = False

createWOA = False
createClosestWOA = False




with open("data/bedmach.pickle","rb") as f:
    bedmach = pickle.load(f)

## Load these into memory just once for speed
bedvalues = bedmach.bed.values
icemask = bedmach.icemask_grounded_and_shelves.values
##################################################

with open("data/bedmachGLIB.pickle","rb") as f:
    GLIB = pickle.load(f)
with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)
################################################

################################################

if writeGL:
    physical, grid, depths, shelves, shelf_keys = bt.get_line_points(bedmach,polygons,True,"edge")
    with open("data/iceedgepoints.pickle","wb") as f:
        pickle.dump([physical,grid,depths,shelves,shelf_keys],f)

with open("data/iceedgepoints.pickle","rb") as f:
    physical,grid,depths,shelves,shelf_keys = pickle.load(f)

with open("data/woawithbed.pickle","rb") as f:
    sal,temp = pickle.load(f)

with open("data/shelfnumbers.pickle","rb") as f:
    shelf_number_labels, shelf_numbers = pickle.load(f)

depths = []
glibs = []
baths = []
for l in range(len(physical)):
    baths.append(GLIB[grid[l][0],grid[l][1]])
    depths.append(bedvalues[grid[l][0],grid[l][1]])
    glibs.append(baths[l])

shelf_areas = bt.shelf_areas()
glibheats = np.asarray([0]*len(physical))# cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys)
#glibheats = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys)

#glibheats = cdw.tempProfileByShelf(bedmach,grid,physical,depths,physical,sal,temp,shelf_keys)

# # #physical = np.asarray(physical)

#with open("data/simple_shelf_thermals.pickle","wb") as f:
    #pickle.dump(glibheats,f)
with open("data/simple_shelf_thermals.pickle","rb") as f:
    glibheats = pickle.load(f)

def explore_graph(glibheats,gbs,k):
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.plot(glibheats[k][0],-glibheats[k][2])
    ax1.axhline(gbs[k])
    ax2.plot(glibheats[k][1],-glibheats[k][2])
    ax2.axhline(gbs[k])
    plt.show()

with open("data/new_massloss.pickle","rb") as f:
    rignot_shelf_massloss = pickle.load(f)

with open("data/slopes_by_shelf.pickle","rb") as f:
    slopes_by_shelf = pickle.load(f)

#rignot_shelf_massloss = bt.shelf_mass_loss('data/amundsilli.h5',polygons)
#glib_by_shelf = cdw.GLIB_by_shelf(GLIB,bedmach,polygons)

#with open("data/glib_by_shelf.pickle","wb") as f:
    #pickle.dump(glib_by_shelf,f)
with open("data/glib_by_shelf.pickle","rb") as f:
    glib_by_shelf = pickle.load(f)

tforce = []
tforceg = []
slopes = []
r2013 = []
#rignot_shelf_massloss,sigmas = cdw.extract_rignot_massloss2013("data/rignot2013.xlsx")
labels = []
draft = bedmach.surface.values - bedmach.thickness.values
for k in tqdm(glibheats.keys()):
    if k in rignot_shelf_massloss.keys() and ~np.isnan(rignot_shelf_massloss[k]):
        tinterp,sinterp = interpolate.interp1d(glibheats[k][2],np.asarray(glibheats[k][0])),interpolate.interp1d(glibheats[k][2],np.asarray(glibheats[k][1]))
        icemean = []
        glibmean = []
        drafts_so_far = []
        drafts = draft[shelf_numbers==shelf_number_labels[k]].round()
        drafts_u,draft_counts = np.unique(drafts,return_counts=True)
        heats = []
        for draft_depth in drafts_u :
            icemean.append(cdw.heat_content((tinterp,sinterp),min(abs(glibheats[k][3]),abs(draft_depth)),10))
            glibmean.append(cdw.heat_content((tinterp,sinterp),min(abs(glib_by_shelf[k]),abs(draft_depth)),10))
        if np.sum(~np.isnan(icemean))>0 and np.sum(~np.isnan(glibmean))>0:
            tforce.append((np.asarray(icemean)[~np.isnan(icemean)].dot(draft_counts[~np.isnan(icemean)]))/np.sum(draft_counts[~np.isnan(icemean)]))
            tforceg.append((np.asarray(glibmean)[~np.isnan(glibmean)].dot(draft_counts[~np.isnan(glibmean)]))/np.sum(draft_counts[~np.isnan(glibmean)]))
            r2013.append(rignot_shelf_massloss[k])
            labels.append(k)
            slopes.append(slopes_by_shelf[k])

plt.scatter(np.asarray(tforce)*np.abs(tforce)*np.asarray(slopes),r2013)
for l in range(len(labels)):
    plt.annotate(labels[l],(tforce[l]*np.abs(tforce)*slopes[l],r2013[l]))
plt.show()

tforce = np.asarray(tforce)
print(tforce)
reg1 = LinearRegression().fit(np.asarray([tforce*np.abs(tforce)*slopes,tforce*np.abs(tforce)*slopes]).T, r2013)

print(reg1.score(np.asarray([tforce*np.abs(tforce)*slopes,tforce*np.abs(tforce)*slopes]).T, r2013))
tforce_est = reg1.predict(np.asarray([tforce*np.abs(tforce)*slopes,tforce*np.abs(tforce)*slopes]).T)

fig, (ax1) = plt.subplots(1,1)
r2_score(r2013,tforce_est)
ax1.scatter(tforce_est,r2013)
plt.show()


print(np.mean(np.abs(tforce_est-r2013)))
print(np.mean(np.abs(tforceg_est-r2013)))
plt.scatter(np.abs(tforce_est-r2013),np.abs(tforceg_est-r2013))
plt.plot(range(20),range(20))
plt.show()

slopes_by_shelf = bt.shelf_sort(shelf_keys,glibheats)
gldepths_by_shelf = bt.shelf_sort(shelf_keys,depths)
glibs_by_shelf = bt.shelf_sort(shelf_keys,glibs)
#rignot_shelf_massloss,rignot_shelf_areas,sigma = cdw.extract_rignot_massloss2019("data/rignot2019.xlsx")
rignot_shelf_massloss,sigmas = cdw.extract_rignot_massloss2013("data/rignot2013.xlsx")
