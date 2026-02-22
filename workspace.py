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
import ipdb
import pandas as pd
from shelf_utils import *

#These flags just make it easy to turn off and steps of the analysis 
writeBedMach = False
writeShelfNumbers = False
writeHUB = False
writePolygons = False
writeGL = False
createWOA = False
createGISS = False
createClosestShelfPoints = False
createClosestHydro = False
createQuants = False
createSlopes = False
createDrafts = False
createFrontThick = False

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

hubs = []
for l in range(len(grid)):
    hubs.append(HUB[grid[l][0]][grid[l][1]])

################################################
# Extract the WOA salinity and temperature fields
################################################
if createWOA:
    sal,temp = woa.create_WOA(bedmach)
    with open("data/woanew.pickle","wb") as f:
        pickle.dump([sal,temp],f)

with open("data/woanew.pickle","rb") as f:
    sal,temp = pickle.load(f)

################################################
## This is the real doozy. This calculates the closest point on the shelf to every grounding line point
##      # using breadth first search
##      # but... on the bedmachine grid.
##      # I have parallelized it but it still takes a week on my computer 
##      # but... this takes a lonngnggggg time. IF you want it precalculated email me!
################################################
if createClosestShelfPoints:
    closest_points = cdw.closest_shelfbreak_points_bfs(grid,zerohubs,bedmach,method="bfs")
    with open("data/closest_points.pickle","wb") as f:
        pickle.dump(closest_points,f)
with open("data/closest_points.pickle","rb") as f:
    closest_points = pickle.load(f)


################################################
## Once you have the points on the shelfbreak from the previous function
## it is trivial to just get the closest point from whatever hydrography you want
## using a simple euclidean distance (this is assumes the hydrography gets pretty close to the 2000m isobath
################################################
if createClosestHydro:
    closest_hydro = cdw.closestHydro(bedmach,grid,physical,closest_points,sal,temp,shelf_keys)
    with open("data/closest_hydro_woanew.pickle","wb") as f:
        pickle.dump(closest_hydro,f)
with open("data/closest_hydro_woanew.pickle","rb") as f:
    closest_hydro = pickle.load(f)

################################################
## Gets the slope of every ice shelf
################################################
if createSlopes:
    slopes_by_shelf = cdw.slope_by_shelf(bedmach,polygons,method="simple")
    with open("data/slopes_by_shelf.pickle","wb") as f:
        pickle.dump(slopes_by_shelf,f)
with open("data/slopes_by_shelf.pickle","rb") as f:
    slopes_by_shelf = pickle.load(f)

# This gets the average draft for every ice shelf
if createDrafts:
    drafts_by_shelf = cdw.draft_by_shelf(bedmach,polygons)
    with open("data/drafts_by_shelf.pickle","wb") as f:
        pickle.dump(drafts_by_shelf,f)
with open("data/drafts_by_shelf.pickle","rb") as f:
    drafts_by_shelf = pickle.load(f)



################################################
## From the closest hydrography points we can now calculate the thermal forcing,
## cdw depths (really a delta pyc-HUB) and gprimes
## you can ignore salts and raw_temps for the most part
################################################
if createQuants:
    out = cdw.parameterization_quantities(bedmach,grid,physical,hubs,closest_hydro,sal,temp,shelf_keys,quant="hubheat",debug=False)
    with open("data/new_stats_woa.pickle","wb") as f:
        pickle.dump(out,f)
with open("data/new_stats_woa.pickle","rb") as f:
    (salts,raw_temps,hubheats,cdwdepths,gprimes) = pickle.load(f)
    

################################################
# Kinda crazy move by me but I calculate 1/f for every grounding line point
######################################
projection = pyproj.Proj("epsg:3031")
fs = []
for x,y in physical:
        lon,lat = projection(x,y,inverse=True)
        fs.append(1/np.abs(gsw.f(lat)))


################################################
# Read in polynya data generated in polynya.py
######################################
with open("data/polyna_by_shelf_2024_nakata.pickle","rb") as f:
    polyna_by_shelf,polyna_by_shelf_weighted = pickle.load(f)


################################################
# Read in depth at ice shelf front and entrance thickness
######################################

if createFrontThick:
    front_thick, front_depth = bt.front_thickness(bedmach,polygons)
    with open("data/front_thick_by_shelf.pickle","wb") as f:
        pickle.dump((front_thick,front_depth),f)
with open("data/front_thick_by_shelf.pickle","rb") as f:
    front_thick, front_depth = pickle.load(f)


################################################
# Sort grounding line points by shelf for averaging
######################################
fs_by_shelf = bt.shelf_sort(shelf_keys,fs)
hubheats_by_shelf = bt.shelf_sort(shelf_keys,hubheats)
depths_by_shelf = bt.shelf_sort(shelf_keys,depths)
salts_by_shelf = bt.shelf_sort(shelf_keys,salts)
raw_temps_by_shelf = bt.shelf_sort(shelf_keys,raw_temps)
cdws_by_shelf = bt.shelf_sort(shelf_keys,cdwdepths)
gprimes_by_shelf = bt.shelf_sort(shelf_keys,gprimes)
hubs_by_shelf = bt.shelf_sort(shelf_keys,hubs)
adusumilli_shelf_massloss,shelf_areas,sigmas_by_shelf =  cdw.extract_adusumilli("data/Adusumilli.csv")


################################################
# move data from shelf based dictionaries to vectorized arrays 
######################################
thermals=[]
cdws = []
hubshelf=[]
entrance_thickness=[]
front_thicks=[]
front_depths=[]
entrance_spread=[]
h_min=[]
h_max=[]
polynas = []
polynas_weighted = []
gprimes=[]
gldepths = []
bars = []
areas = []
mys = []
slopes = []
salts = []
raw_temps = []
fs = []
sigmas = []
labels = []
gllen = []
avg_drafts = []

for k in slopes_by_shelf.keys():
    if (k in adusumilli_shelf_massloss and ~np.isnan(adusumilli_shelf_massloss[k]) and ~np.isnan(slopes_by_shelf[k]))or k =="Amery" :
        labels.append(k)
        slopes.append(list([slopes_by_shelf[k]])*np.shape(hubheats_by_shelf[k])[1])
        avg_drafts.append(list([drafts_by_shelf[k]])*np.shape(hubheats_by_shelf[k])[1])
        fs.append(list([np.nanmean(fs_by_shelf[k])])*np.shape(hubheats_by_shelf[k])[1])
        thermals.append(np.nanmean(hubheats_by_shelf[k],axis=0))
        cdws.append(np.nanmean(cdws_by_shelf[k],axis=0))
        gprimes.append(np.nanmean(gprimes_by_shelf[k],axis=0))
        hubshelf.append(np.nanmean(hubs_by_shelf[k]))
        gldepths.append(np.nanmean(depths_by_shelf[k]))
        gllen.append(len(depths_by_shelf[k]))
        entrance_thickness.append(np.nanmean(np.abs(hubs_by_shelf[k]))- np.nanmean(np.abs(np.asarray(front_thick[k]))))
        front_thicks.append(np.nanmean(np.abs(np.asarray(front_thick[k]))))
        front_depth[k] = np.asarray(front_depth[k])
        front_depth[k] = np.nanmean(np.abs(front_depth[k][front_depth[k]>100]))
        front_depths.append(front_depth[k])

        polynas.append(np.nansum(polyna_by_shelf[k]))
        salts.append(np.nanmean(salts_by_shelf[k]))
        raw_temps.append(np.nanmean(raw_temps_by_shelf[k]))
        if k == "Amery":
            sigmas.append(0.7)
            areas.append(list([60228])*np.shape(hubheats_by_shelf[k])[1])
            mys.append(0.8)
        else:
            sigmas.append(sigmas_by_shelf[k])
            areas.append(list([shelf_areas[k]])*np.shape(hubheats_by_shelf[k])[1])
            mys.append(adusumilli_shelf_massloss[k])


# Vectorize arrays
areas = np.asarray(areas)[:,0]
polynas = np.asarray(polynas)
polynas_weighted = np.asarray(polynas_weighted)
slopes = np.asarray(slopes)[:,0]
gprimes = np.asarray(gprimes)[:,0]
hubshelf = np.asarray(hubshelf)
entrance_spread = np.asarray(entrance_spread)
salts = np.asarray(salts)
gldepths = np.asarray(gldepths)
avg_drafts = np.asarray(avg_drafts)[:,0]
raw_temps = np.asarray(raw_temps)
front_depth = np.asarray(front_depth)
cdws = np.asarray(cdws)[:,0]
fs = np.asarray(fs)[:,0]
thermals = np.asarray(thermals)[:,0]
mys = np.asarray(mys)


## calculate Btotal, Bmelt and Bpolynya
sorti = np.argsort(polynas)[::-1]
gigatonconv = 10**(-12)
rhoi=910
scalefactor = rhoi*gigatonconv*10**6
meltflux = mys*(1/(60*60*24*365))*(920.0)
Bmelt = 35*(1/(60*60*24*365))*(920.0)*(mys*areas*10**6)/1027*9.8*(gsw.beta(34.5,-1.8,gldepths/2))
Bpolyna = 30*(1/(60*60*24*365))*(920.0)*(polynas)/1027*9.8*(gsw.beta(34.5,-1.9,0))
Btotal = -(Bmelt-Bpolyna)

           
shelf_classnumber,shelf_color = read_shelf_class(labels)

Bmelts = 34.5*(1/(60*60*24*365))*(920.0)*(mys*areas*10**6)/1027*9.8*(gsw.beta(34.5,-1.8,avg_drafts))

Bpolyna = -Bpolyna

shelf_stats = {
    #gl sorted
    "cdws":cdws,
    "salts":salts,
    "raw_temps":raw_temps,
    "Tcdw":thermals,
    "gprimes":gprimes,
    "fs-1":fs,
    "gldepths":gldepths,
    "entrance_thickness":entrance_thickness,
    "front_thick":front_thicks,
    "front_depth":front_depths,

    #area sorted
    "slopes":slopes,
    "mys":mys,
    "sigmas":sigmas,
    "avg_drafts":avg_drafts,

    # simple sum
    "areas":areas*(10**6),
    "gllen":gllen,
    "Bmelt":Bmelt,
    "Bpolyna":Bpolyna,
    "Btotal":Btotal,
    "shelf_class":shelf_classnumber,
    "shelf_color":shelf_color,

    #no sum
    "labels":labels,
}


# Merge shelves
shelf_stats = shelf_merge(shelf_stats,"Filchner","Ronne","FRIS")
shelf_stats = shelf_merge(shelf_stats,"Ross_East","Ross_West","Ross")


# Classification vs Btotal figure
pf.shelf_class_fig(shelf_stats,scalefactor)

# Evaluate the scaling theory
shelf_stats = pf.evaluate_theory(shelf_stats,colorthresh=5,textthresh=5,mode="linear")

# Takes a shelf stats already modified by evaluate theory
pf.optimal_classification(shelf_stats)

