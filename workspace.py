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

# Create HUB

writeBedMach = False
writeShelfNumbers = False
writeHUB = False
writePolygons = False
writeGL =False
createWOA = True
createGISS = False
createClosestWOA = False

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

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
if writeHUB:

    print("Calculating Hubs")
    HUB = HUB.generateHUBs(bedvalues,icemask)

    with open("data/bedmachHUB.pickle","wb") as f:
        pickle.dump(HUB,f)


with open("data/bedmachHUB.pickle","rb") as f:
    HUB = pickle.load(f)

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
    baths.append(HUB[grid[l][0]][grid[l][1]])

################################################

################################################
if createGISS:
    gisssal,gisstemp = woa.create_GISS(bedmach)
    with open("data/giss_hydro_r2.pickle","wb") as f:
        pickle.dump([gisssal,gisstemp],f)

with open("data/giss_hydro_r2.pickle","rb") as f:
    gisssal,gisstemp = pickle.load(f)


if createWOA:
    sal,temp = woa.create_WOA(bedmach)
    with open("data/woa.pickle","wb") as f:
        pickle.dump([sal,temp],f)

with open("data/woa.pickle","rb") as f:
    sal,temp = pickle.load(f)


print(np.shape(sal.s_an.values))
if createClosestWOA:
    closest_points = cdw.closest_WOA_points(grid,zerobaths,bedmach,method="bfs")
    with open("data/closest_points.pickle","wb") as f:
        pickle.dump(closest_points,f)
with open("data/closest_points.pickle","rb") as f:
    closest_points = pickle.load(f)

depths = []
hubs = []
for l in range(len(baths)):
    depths.append(bedvalues[grid[l][0],grid[l][1]])
    hubs.append(baths[l])
    if np.isnan(baths[l]):
        baths[l]=bedvalues[grid[l][0],grid[l][1]]
        
#closest_hydro = cdw.closestHydro(bedmach,grid,physical,closest_points,sal,temp,shelf_keys)

#with open("data/closest_hydro_woathree.pickle","wb") as f:
    #pickle.dump(closest_hydro,f)
with open("data/closest_hydro_woathree.pickle","rb") as f:
    closest_hydro = pickle.load(f)

#avg_s,avg_t,depths = cdw.averageForShelf("Thwaites",bedmach,grid,physical,hubs,closest_hydro,sal,temp,shelf_keys,quant="hubheat",debug=False)
#with open("data/ThwaitesAverages.pickle","wb") as f:
    #pickle.dump((avg_t,avg_s,depths),f)


#shelf_areas = bt.shelf_areas()
#hubheats,cdwdepths,gprimes = cdw.revampedClosest(bedmach,grid,physical,hubs,closest_hydro,sal,temp,shelf_keys,quant="hubheat",debug=False)
physical = np.asarray(physical)
grid = np.asarray(grid)

#with open("data/stats_woa.pickle","wb") as f:
    #pickle.dump((hubheats,cdwdepths,gprimes),f)
with open("data/stats_woa.pickle","rb") as f:
    (hubheats,cdwdepths,gprimes) = pickle.load(f)

#with open("data/hubheats_gissr2.pickle","rb") as f:
    #hubheats = pickle.load(f)
#with open("data/cdwdepths_gissr2.pickle","rb") as f:
    #cdwdepths = pickle.load(f)
#with open("data/gprimes_gissr2.pickle","rb") as f:
    #gprimes = pickle.load(f)

# slopes_by_shelf = cdw.slope_by_shelf(bedmach,polygons)
# with open("data/slopes_by_shelf.pickle","wb") as f:
#     pickle.dump(slopes_by_shelf,f)
with open("data/slopes_by_shelf.pickle","rb") as f:
    slopes_by_shelf = pickle.load(f)

print("hello")
projection = pyproj.Proj("epsg:3031")
fs = []
for x,y in physical:
        lon,lat = projection(x,y,inverse=True)
        fs.append(1/np.abs(gsw.f(lat)))

#polynas=np.asarray(polynas)
fs_by_shelf = bt.shelf_sort(shelf_keys,fs)
hubheats_by_shelf = bt.shelf_sort(shelf_keys,hubheats)
cdws_by_shelf = bt.shelf_sort(shelf_keys,cdwdepths)
gprimes_by_shelf = bt.shelf_sort(shelf_keys,gprimes)
gldepths_by_shelf = bt.shelf_sort(shelf_keys,depths)
hubs_by_shelf = bt.shelf_sort(shelf_keys,hubs)
rignot_shelf_massloss,shelf_areas,sigmas_by_shelf =  cdw.extract_adusumilli("data/Adusumilli.csv")


thermals=[]
cdws = []
gldepths = []
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
        gldepths.append(np.nanmean(gldepths_by_shelf[k]))
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


