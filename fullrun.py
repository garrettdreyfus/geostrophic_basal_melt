import bathtub as bt
import pickle
import HUB
import gsw,xarray, pyproj
import matplotlib.pyplot as plt
import numpy as np
import woa
import matplotlib.colors as colors
import paperfigures as pf
import cdw

#These flags just make it easy to turn off and steps of the analysis 
writeBedMach = True
writeShelfNumbers = True
writeHUB = True
writePolygons = True
writeGL = True
createWOA = True
createGISS = True
createClosestShelfPoints = True
createQuants = True
createSlopes = True

##################################################
#Create Polygon objects from MEASURES files
##################################################

if writePolygons:
    bt.save_polygons()

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

###############################################
# This is a nuisance but I began the analysis with bedmap data,
#     this basically structures bedmachine with bedmap type field names
#     and then saves it
############

if writeBedMach:
    print("Grabbing BedMachine Data")
    bedmach = bt.convert_bedmachine("data/BedMachine.nc",coarsenfact=1)
    with open("data/bedmach.pickle","wb") as f:
        pickle.dump(bedmach,f)

with open("data/bedmach.pickle","rb") as f:
    bedmach = pickle.load(f)


##################################################
# Calculate HUB values
##################################################
bedvalues = bedmach.bed.values
icemask = bedmach.icemask_grounded_and_shelves.values
if writeHUB:

    print("Calculating Hubs")
    HUB = HUB.generateHUBs(bedvalues,icemask)

    with open("data/bedmachHUB.pickle","wb") as f:
        pickle.dump(HUB,f)


with open("data/bedmachHUB.pickle","rb") as f:
    HUB = pickle.load(f)


################################################
# This finds grounding lines points, stores their physical (x,y) coordinates as well as their grid indices
#     and the closest ice shelf they are affiliated with 
################################################

if writeGL:
    physical, grid, depths, shelves, shelf_keys = bt.get_line_points(bedmach,polygons)
    with open("data/groundinglinepoints.pickle","wb") as f:
        pickle.dump([physical,grid,depths,shelves,shelf_keys],f)

with open("data/groundinglinepoints.pickle","rb") as f:
    physical,grid,depths,shelves,shelf_keys = pickle.load(f)

################################################
## Use the grid indices to get the HUBS for each grounding line point
################################################

baths = []
for l in range(len(grid)):
    baths.append(HUB[grid[l][0]][grid[l][1]])

################################################
# Extract the WOA salinity and temperature fields
################################################
if createWOA:
    sal,temp = woa.create_WOA(bedmach)
    with open("data/woa.pickle","wb") as f:
        pickle.dump([sal,temp],f)

with open("data/woa.pickle","rb") as f:
    sal,temp = pickle.load(f)

################################################
## This is the real doozy. This calculates the closest point on the shelf to every grounding line point
##      # using breadth first search
##      # but... on the bedmachine grid.
##      # I have parallelized it but it still takes a week on my computer 
##      # but... this takes a lonngnggggg time. IF you want it precalculated email me!
################################################
if createClosestShelfPoints:
    closest_points = cdw.closest_shelfbreak_points_bfs(grid,zerobaths,bedmach,method="bfs")
    with open("data/closest_points.pickle","wb") as f:
        pickle.dump(closest_points,f)
with open("data/closest_points.pickle","rb") as f:
    closest_points = pickle.load(f)


################################################
## Once you have the points on the shelfbreak from the previous function
## it is trivial to just get the closest point from whatever hydrography you want
## using a simple euclidean distance
################################################
if createClosestHydro:
    closest_hydro = cdw.closestHydro(bedmach,grid,physical,closest_points,sal,temp,shelf_keys)
    with open("data/closest_hydro_woathree.pickle","wb") as f:
        pickle.dump(closest_hydro,f)
with open("data/closest_hydro_woathree.pickle","rb") as f:
    closest_hydro = pickle.load(f)


################################################
## From the closest hydrography points we can now calculate the thermal forcing,
## cdw depths (really a delta pyc-HUB) and gprimes
################################################
if createQuants:
    hubheats,cdwdepths,gprimes = cdw.parameterization_quantities(bedmach,grid,physical,hubs,closest_hydro,sal,temp,shelf_keys,quant="hubheat",debug=False)

    with open("data/stats_woa.pickle","wb") as f:
        pickle.dump((hubheats,cdwdepths,gprimes),f)
with open("data/stats_woa.pickle","rb") as f:
    (hubheats,cdwdepths,gprimes) = pickle.load(f)

if createSlopes
    # slopes_by_shelf = cdw.slope_by_shelf(bedmach,polygons)
    # with open("data/slopes_by_shelf.pickle","wb") as f:
    #     pickle.dump(slopes_by_shelf,f)
    with open("data/slopes_by_shelf.pickle","rb") as f:
        slopes_by_shelf = pickle.load(f)

#calculate 1/f
print("hello")
projection = pyproj.Proj("epsg:3031")
fs = []
for x,y in physical:
        lon,lat = projection(x,y,inverse=True)
        fs.append(1/np.abs(gsw.f(lat)))

#Sort points by shelf for averaging
fs_by_shelf = bt.shelf_sort(shelf_keys,fs)
hubheats_by_shelf = bt.shelf_sort(shelf_keys,hubheats)
cdws_by_shelf = bt.shelf_sort(shelf_keys,cdwdepths)
gprimes_by_shelf = bt.shelf_sort(shelf_keys,gprimes)
hubs_by_shelf = bt.shelf_sort(shelf_keys,hubs)
rignot_shelf_massloss,shelf_areas,sigmas_by_shelf =  cdw.extract_adusumilli("data/Adusumilli.csv")


# move data from shelf based dictionaries to vectorized arrays 
thermals=[]
cdws = []
hubshelf=[]
gprimes=[]
bars = []
areas = []
mys = []
slopes = []
fs = []
sigmas = []
labels = []

for k in slopes_by_shelf.keys():
    if (k in rignot_shelf_massloss and ~np.isnan(rignot_shelf_massloss[k]) and k!="George_VI" and ~np.isnan(slopes_by_shelf[k]))or k =="Amery" :
        x,y = (polygons[k][0].centroid.x,polygons[k][0].centroid.y)
        slopes.append(list([slopes_by_shelf[k]])*np.shape(hubheats_by_shelf[k])[1])
        labels.append(k)
        fs.append(list([np.nanmean(fs_by_shelf[k])])*np.shape(hubheats_by_shelf[k])[1])
        thermals.append(np.nanmean(hubheats_by_shelf[k],axis=0))
        cdws.append(np.nanmean(cdws_by_shelf[k],axis=0))
        gprimes.append(np.nanmean(gprimes_by_shelf[k],axis=0))
        hubshelf.append(np.nanmean(hubs_by_shelf[k]))
        if k == "Amery":
            sigmas.append(0.7)
            areas.append(list([60228])*np.shape(hubheats_by_shelf[k])[1])
            mys.append(0.8)
        else:
            sigmas.append(sigmas_by_shelf[k])
            areas.append(list([shelf_areas[k]])*np.shape(hubheats_by_shelf[k])[1])
            mys.append(rignot_shelf_massloss[k])


# pass to plotting functions
areas = np.asarray(areas)
slopes = np.asarray(slopes)
hubshelf = np.asarray(hubshelf)
cdws = np.asarray(cdws)
pf.closestMethodologyFig(bedmach,grid,physical,baths,closest_points,sal,temp,shelves)
pf.hydro_vs_slope_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels)
pf.hydro_vs_slope_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels,xlim=1500,ylim=0.005,nozone=(-1000,-1000))
pf.param_vs_melt_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels,xlim=5,ylim=5,textthresh=0,colorthresh=5)
pf.param_vs_melt_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels)
exit()


