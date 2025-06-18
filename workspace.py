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
createVolumes = False

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

#cdw.extract_drafts(bedmach,polygons)
#exit()
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
## using a simple euclidean distance
################################################
if createClosestHydro:
    closest_hydro = cdw.closestHydro(bedmach,grid,physical,closest_points,sal,temp,shelf_keys)
    with open("data/closest_hydro_woanew.pickle","wb") as f:
        pickle.dump(closest_hydro,f)
with open("data/closest_hydro_woanew.pickle","rb") as f:
    closest_hydro = pickle.load(f)

# slopes_by_shelf = cdw.slope_by_shelf(bedmach,polygons)
if createSlopes:
    slopes_by_shelf = cdw.slope_by_shelf(bedmach,polygons)
    with open("data/new_slopes_by_shelf.pickle","wb") as f:
        pickle.dump(slopes_by_shelf,f)
with open("data/new_slopes_by_shelf.pickle","rb") as f:
    slopes_by_shelf = pickle.load(f)



################################################
## From the closest hydrography points we can now calculate the thermal forcing,
## cdw depths (really a delta pyc-HUB) and gprimes
################################################
#salts,raw_temps,hubheats,cdwdepths,gprimes = cdw.parameterization_quantities(bedmach,grid,physical,hubs,closest_hydro,sal,temp,shelf_keys,quant="hubheat",debug=False)
#with open("data/stats_kitkaboodle.pickle","wb") as f:
    #pickle.dump((salts,raw_temps,hubheats,cdwdepths,gprimes),f)
# if createQuants:
#     #for i in range(len(closest_points)):
#         #closest_hydro[i] = 109752
#     _,_,hubheats,cdwdepths,gprimes = cdw.parameterization_quantities(bedmach,grid,physical,hubs,closest_hydro,sal,temp,shelf_keys,quant="hubheat",debug=False)
#     with open("data/stats_woa.pickle","wb") as f:
#         pickle.dump((hubheats,cdwdepths,gprimes),f)
# with open("data/stats_kitkaboodle.pickle","rb") as f:
#     (salts,raw_temps,hubheats,cdwdepths,gprimes) = pickle.load(f)
    
salts,raw_temps,hubheats,cdwdepths,gprimes = cdw.parameterization_quantities(bedmach,grid,physical,hubs,closest_hydro,sal,temp,shelf_keys,quant="hubheat",debug=False)

cdwdepths[cdwdepths<0]=0

# with open("data/stats_woa.pickle","rb") as f:
#     (hubheats,cdwdepths,gprimes) = pickle.load(f)



if createVolumes:
    disconnectmask = np.abs(np.abs(HUB)-np.abs(bedvalues))<10
    volumes_by_shelf = cdw.volumes_by_shelf(bedmach,polygons)
    with open("data/volumes_by_shelf.pickle","wb") as f:
        pickle.dump(volumes_by_shelf,f)
with open("data/volumes_by_shelf.pickle","rb") as f:
    volumes_by_shelf = pickle.load(f)



#calculate 1/f
print("hello")
projection = pyproj.Proj("epsg:3031")
fs = []
for x,y in physical:
        lon,lat = projection(x,y,inverse=True)
        fs.append(1/np.abs(gsw.f(lat)))

with open("data/polyna_by_shelf_2024.pickle","rb") as f:
    polyna_by_shelf,polyna_by_shelf_weighted = pickle.load(f)


#Sort points by shelf for averaging
fs_by_shelf = bt.shelf_sort(shelf_keys,fs)
hubheats_by_shelf = bt.shelf_sort(shelf_keys,hubheats)
depths_by_shelf = bt.shelf_sort(shelf_keys,depths)
salts_by_shelf = bt.shelf_sort(shelf_keys,salts)
raw_temps_by_shelf = bt.shelf_sort(shelf_keys,raw_temps)
cdws_by_shelf = bt.shelf_sort(shelf_keys,cdwdepths)
gprimes_by_shelf = bt.shelf_sort(shelf_keys,gprimes)
hubs_by_shelf = bt.shelf_sort(shelf_keys,hubs)
rignot_shelf_massloss,shelf_areas,sigmas_by_shelf =  cdw.extract_adusumilli("data/Adusumilli.csv")

#front_thick = bt.front_thickness(bedmach,polygons)
#with open("data/front_thick_by_shelf.pickle","wb") as f:
    #pickle.dump(front_thick,f)
with open("data/front_thick_by_shelf.pickle","rb") as f:
    front_thick = pickle.load(f)


#dump_volume_by_shelf = bt.dump_volume(bedmach,polygons)
#with open("data/dump_volume_by_shelf.pickle","wb") as f:
    #pickle.dump(dump_volume_by_shelf,f)
with open("data/dump_volume_by_shelf.pickle","rb") as f:
    dump_volume_by_shelf = pickle.load(f)


# move data from shelf based dictionaries to vectorized arrays 
thermals=[]
cdws = []
hubshelf=[]
entrance_thickness=[]
polynas = []
polynas_weighted = []
gprimes=[]
gldepths = []
bars = []
areas = []
mys = []
slopes = []
volumes = []
salts = []
raw_temps = []
dump_volumes = []
fs = []
sigmas = []
labels = []

for k in slopes_by_shelf.keys():
    if (k in rignot_shelf_massloss and ~np.isnan(rignot_shelf_massloss[k]) and ~np.isnan(slopes_by_shelf[k]))or k =="Amery" :
        x,y = (polygons[k][0].centroid.x,polygons[k][0].centroid.y)
        slopes.append(list([slopes_by_shelf[k]])*np.shape(hubheats_by_shelf[k])[1])
        volumes.append(list([volumes_by_shelf[k]])*np.shape(hubheats_by_shelf[k])[1])
        dump_volumes.append(list([np.sum(dump_volume_by_shelf[k])])*np.shape(hubheats_by_shelf[k])[1])
        labels.append(k)
        fs.append(list([np.nanmean(fs_by_shelf[k])])*np.shape(hubheats_by_shelf[k])[1])
        thermals.append(np.nanmean(hubheats_by_shelf[k],axis=0))
        cdws.append(np.nanmean(cdws_by_shelf[k],axis=0))
        gprimes.append(np.nanmean(gprimes_by_shelf[k],axis=0))
        hubshelf.append(np.nanmean(hubs_by_shelf[k]))
        gldepths.append(np.nanmean(depths_by_shelf[k]))
        entrance_thickness.append(np.nanmean(np.abs(hubs_by_shelf[k]))- np.nanmean(np.asarray(front_thick[k])))
        #hubshelf.append(np.nanmean(np.abs(hubs_by_shelf[k])))
        polynas.append(np.nansum(polyna_by_shelf[k]))
        polynas_weighted.append(np.nanmean(polyna_by_shelf_weighted[k]))
        salts.append(np.nanmean(salts_by_shelf[k]))
        raw_temps.append(np.nanmean(raw_temps_by_shelf[k]))
        if k == "Amery":
            sigmas.append(0.7)
            areas.append(list([60228])*np.shape(hubheats_by_shelf[k])[1])
            mys.append(0.8)
        else:
            sigmas.append(sigmas_by_shelf[k])
            areas.append(list([shelf_areas[k]])*np.shape(hubheats_by_shelf[k])[1])
            mys.append(rignot_shelf_massloss[k])




# pass to plotting functions
areas = np.asarray(areas)[:,0]
polynas = np.asarray(polynas)
polynas_weighted = np.asarray(polynas_weighted)
slopes = np.asarray(slopes)[:,0]
volumes = np.asarray(volumes)[:,0]
dump_volumes = np.asarray(dump_volumes)[:,0]
gprimes = np.asarray(gprimes)[:,0]
hubshelf = np.asarray(hubshelf)
salts = np.asarray(salts)
gldepths = np.asarray(gldepths)
raw_temps = np.asarray(raw_temps)
cdws = np.asarray(cdws)[:,0]
print("-"*10)
print("NEGATIVES",(cdws<0).sum())
for i in range(len(cdws)):
    if cdws[i]<0:
        print(labels[i])
ipdb.set_trace()
print("-"*10)
# cdws[cdws<0]=100
fs = np.asarray(fs)[:,0]
thermals = np.asarray(thermals)[:,0]
mys = np.asarray(mys)


sorti = np.argsort(polynas)[::-1]
gigatonconv = 10**(-12)
rhoi=910
scalefactor = rhoi*gigatonconv*10**6
B0 = (mys*areas*scalefactor-polynas)/1027*9.8*(7.8*10**(-4))

def read_shelf_class(labels):
    shelf_class = pd.read_csv("shelf_classification.csv",sep=',')
    shelf_color = []
    shelf_classnumber = []
    for i in labels:
        classification = shelf_class.loc[shelf_class['Shelf Name']==i].values[0][1]
        explanation = shelf_class.loc[shelf_class['Shelf Name']==i].values[0][3]
        if classification and type(classification) == str:
            if 'both' in classification:
                shelf_color.append("gray")
                shelf_classnumber.append(0)
            if 'disconnected' in classification:
                if type(explanation) == str:
                    shelf_color.append("plum")
                    shelf_classnumber.append(-0.5)
                else:
                    shelf_color.append("fuchsia")
                    shelf_classnumber.append(-1)
            elif 'connected' in classification:
                if type(explanation) == str:
                    shelf_color.append("bisque")
                    shelf_classnumber.append(0.5)
                else:
                    shelf_color.append("orange")
                    shelf_classnumber.append(1)
            elif 'unknown' in classification:
                shelf_color.append("gray")
                shelf_classnumber.append(0)
            else:
                1+1
                # print(i)
                # print(classification)
        else:
            shelf_color.append('white')
            shelf_classnumber.append(np.nan)
    return shelf_classnumber 
            
#meanfris = np.mean(ratio[Ronnei] + ratio[Filchneri])
#ratio[Ronnei]=0
#ratio[Filchneri]=0

shelf_classnumber = read_shelf_class(labels)

pf.clean(cdws,salts,raw_temps,thermals,gprimes,slopes,dump_volumes,fs,areas,gldepths,entrance_thickness,mys,sigmas,labels,polynas,polynas_weighted,shelf_classnumber,colorthresh=5,textthresh=5)
pf.cleanlog(cdws,salts,raw_temps,thermals,gprimes,slopes,dump_volumes,fs,areas,gldepths,entrance_thickness,mys,sigmas,labels,polynas,polynas_weighted,shelf_classnumber,colorthresh=5,textthresh=5)
#pf.param_vs_melt_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels,xlim=30,ylim=30,textthresh=0,colorthresh=5,colorfield=list(np.log10(ratio)))
thermals =np.asarray(thermals)
#pf.singleparam_vs_melt_fig((thermals*thermals)*slopes,mys,sigmas,labels,r'$\theta_{\mathrm{CDW}}-\theta_{\mathrm{surf}}$')
#pf.singleparam_vs_melt_fig(slopes,mys,sigmas,labels,r'$s_{\mathrm{ice}}$')
#pf.single(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels,xlim=5,ylim=5,textthresh=0,colorthresh=5)


