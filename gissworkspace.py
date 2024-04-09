import bathtub as bt
import pickle
import HUB
import gsw,xarray, pyproj
import matplotlib.pyplot as plt
import numpy as np
import woa
import matplotlib.colors as colors
import paperfigures as pf
from tqdm import tqdm
import cdw
import cProfile

#These flags just make it easy to turn off and steps of the analysis 
writeBedMach = False
writeShelfNumbers = False
writeHUB = False
writePolygons = False
writeGL =False
createWOA = False
createGISS = False
createClosestShelfPoints = False
createClosestHydro = False
createQuants = False
createSlopes = False

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
if createGISS:
    sal,temp = woa.create_GISS(bedmach)
    with open("data/giss.pickle","wb") as f:
        pickle.dump([sal,temp],f)

with open("data/giss.pickle","rb") as f:
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
#sal,temp = woa.create_WOA(bedmach)
if createClosestHydro:
    sal,temp = woa.create_WOA(bedmach)
    closest_hydro = cdw.closestHydro(bedmach,grid,physical,closest_points,sal,temp,shelf_keys)
    with open("data/closest_hydro_giss.pickle","wb") as f:
        pickle.dump(closest_hydro,f)
with open("data/closest_hydro_giss.pickle","rb") as f:
    closest_hydro = pickle.load(f)

from collections import defaultdict

res = defaultdict(list)

for i, j in zip(zip(closest_hydro,hubs), range(len(closest_hydro))):
    res[i].append(j)


################################################
## From the closest hydrography points we can now calculate the thermal forcing,
## cdw depths (really a delta pyc-HUB) and gprimes
################################################
#pf.overview_figure_reduced(bedmach)
#sal,temp = woa.create_WOA(bedmach)
#avg_s, avg_t, d = cdw.averageForShelf("Getz",bedmach,grid,physical,hubs,closest_hydro,sal,temp,shelf_keys)
##plt.plot(avg_t,-np.asarray(d),color="red")
##plt.show()
#with open("data/closest_hydro_giss.pickle","rb") as f:
    #closest_hydro = pickle.load(f)
#for i in range(1,11):
    #sal,temp = woa.create_GISS(bedmach,ens_memb=i)
    #avg_s, avg_t, d = cdw.averageForShelf("Getz",bedmach,grid,physical,hubs,closest_hydro,sal,temp,shelf_keys)
    #plt.plot(avg_t,-np.asarray(d),color="green")
#plt.xlim(-2,2.1)
#plt.show()
 
if createQuants:
    ensemble = {}
    for i in tqdm(range(1,5)):
        sal,temp = woa.create_GISS(bedmach,ens_memb=i)
        hubheats,cdwdepths,gprimes = cdw.parameterization_quantities(bedmach,grid,physical,hubs,res,sal,temp,shelf_keys,quant="hubheat",debug=False)
        ensemble[i] = (hubheats,cdwdepths,gprimes)
    with open("data/ens_f4_giss.pickle","wb") as f:
        pickle.dump(ensemble,f)

with open("data/ens_f4_giss.pickle","rb") as f:
    ensemble = pickle.load(f)

if createSlopes:
    slopes_by_shelf = cdw.slope_by_shelf(bedmach,polygons)
    with open("data/slopes_by_shelf.pickle","wb") as f:
        pickle.dump(slopes_by_shelf,f)
with open("data/slopes_by_shelf.pickle","rb") as f:
    slopes_by_shelf = pickle.load(f)


years = np.linspace(1990,2010,240)
plt.plot((np.min(years),np.max(years)),(1072,1072),c="black",linewidth=5)
totals = []
for i in tqdm(range(1,5)):
    hubheats,cdwdepths,gprimes = ensemble[i]
    #calculate 1/f
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
                #areas.append(60228)
                mys.append(0.8)
            else:
                sigmas.append(sigmas_by_shelf[k])
                areas.append(list([shelf_areas[k]])*np.shape(hubheats_by_shelf[k])[1])
                #areas.append(shelf_areas[k])
                mys.append(rignot_shelf_massloss[k])


    # pass to plotting functions
    areas = np.asarray(areas)
    slopes = np.asarray(slopes)
    hubshelf = np.asarray(hubshelf)
    cdws = np.asarray(cdws)
    thermals=np.asarray(thermals)
    #pf.hub_schematic_figure()
    #exit()
    #for i in range(174,177):
        #plt.scatter(thermals[:,i],mys)
        #plt.show()
    #pf.closestMethodologyFig(bedmach,grid,physical,hubs,closest_points,sal,temp,shelves)
    #pf.hydro_vs_slope_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels)
    #pf.hydro_vs_slope_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels,xlim=1500,ylim=0.005,nozone=(-1000,-1000))
    out = pf.masslossparam(cdws,thermals,gprimes,slopes,fs,areas)
    icedens = 917
    gigatonconv = 10**(-12)
    scalefactor = icedens*gigatonconv*10**6
    badyear = 175
    massloss = np.sum(out,axis=0)*scalefactor
    totals.append(massloss)
    plt.plot(years,massloss,c="grey")
    #plt.plot(cdw.moving_average(years,36),cdw.moving_average(np.sum(out,axis=0)*scalefactor,36))
    plt.xlabel("Year",fontsize=18)
    plt.ylabel("Predicted basal mass loss (Gt/yr)",fontsize=18)
    if np.nanmin(np.sum(out,axis=0)*scalefactor)<480:
        print(i)
    #pf.param_vs_melt_fig(np.nanmean(cdws,axis=1),np.nanmean(thermals,axis=1),np.nanmean(gprimes,axis=1),np.nanmean(slopes,axis=1),np.nanmean(fs,axis=1),mys,sigmas,labels,title=i)
    #pf.param_vs_melt_fig(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels,xlim=5,ylim=5,textthresh=0,colorthresh=5)
    #thermals =np.asarray(thermals)
    #pf.singleparam_vs_melt_fig((thermals*thermals)*slopes,mys,sigmas,labels,r'$\theta_{\mathrm{CDW}}-\theta_{\mathrm{surf}}$')
    #pf.singleparam_vs_melt_fig(slopes,mys,sigmas,labels,r'$s_{\mathrm{ice}}$')
    #pf.single(cdws,thermals,gprimes,slopes,fs,mys,sigmas,labels,xlim=5,ylim=5,textthresh=0,colorthresh=5)
totals = np.asarray(totals)
plt.plot(years,np.mean(totals,axis=0),c="red",linewidth=5)
print(np.mean(np.mean(totals)))
plt.show()
