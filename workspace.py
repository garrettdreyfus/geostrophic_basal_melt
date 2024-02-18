import bathtub as bt
import shapely
import pickle
import GLIB
import time,gsw,xarray, pyproj
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn import preprocessing
from sklearn import tree
import numpy as np
from sklearn.metrics import r2_score
import woa
import matplotlib.colors as colors
import cdw
from tqdm import tqdm
import xarray as xr

# Create GLIB

writeBedMach = False
writeShelfNumbers = False
writeGLIB = False
writePolygons = False
writeGL =False
createWOA = True
createGISS = True
createClosestWOA = False

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

#rignot_shelf_massloss =  bt.shelf_mass_loss("",polygons)

#with open("data/onlypositivemelt.pickle","wb") as f:
    #pickle.dump(rignot_shelf_massloss,f)

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

#GLIBregions = GLIB.generateGLIBsLabel(bedvalues,icemask)
##with open("data/bedmachGLIBregions.pickle","wb") as f:
    #pickle.dump(GLIBregions,f)
with open("data/bedmachGLIBregions.pickle","rb") as f:
    GLIBregions = pickle.load(f)
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
if createGISS:
    gisssal,gisstemp = woa.create_GISS(bedmach)
    with open("data/summergiss.pickle","wb") as f:
        pickle.dump([gisssal,gisstemp],f)

with open("data/summergiss.pickle","rb") as f:
    gisssal,gisstemp = pickle.load(f)


if createWOA:
    sal,temp = woa.create_WOA(bedmach)
    with open("data/summerwoa.pickle","wb") as f:
        pickle.dump([sal,temp],f)

with open("data/summerwoa.pickle","rb") as f:
    sal,temp = pickle.load(f)

if createClosestWOA:
    zerobaths = np.full_like(baths,0)
    closest_points = cdw.closest_WOA_points(grid,zerobaths,bedmach,method="bfs")
    with open("data/closest_points_ZERO.pickle","wb") as f:
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
#physical_ice, _, _, _, shelf_keys_edge = bt.get_line_points(bedmach,polygons,mode="edge")

#with open("data/physical_ice.pickle","wb") as f:
    #pickle.dump((physical_ice,shelf_keys_edge),f)
with open("data/physical_ice.pickle","rb") as f:
    physical_ice,shelf_keys_edge = pickle.load(f)

#slopes = cdw.slopeFromClosestPoint(bedmach,physical_ice,grid,physical,depths,closest_points,shelf_keys,shelf_keys_edge)
glibheats = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="glibheat",debug=False)
#layertemp = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="newtemp")
#hubsalts = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="glibheat")
gprimes = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="gprime")
#dsalts = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="dsalt")
cdwdepths = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="cdwdepth",debug=True)
#distances = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="distance")
#isopycnaldepths = cdw.tempFromClosestPoint(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="thermocline",shelfkeys=shelf_keys)
#physical = np.asarray(physical)
#grid = np.asarray(grid)
#plt.scatter(grid.T[0][40000:80000],grid.T[1][40000:80000],c=range(grid.shape[0])[40000:80000])
#plt.colorbar()
#plt.scatter(grid.T[0][40000:80000],grid.T[1][40000:80000],c="red")
#for l in range(40000,80000,100):
    #plt.annotate(l,(grid.T[0,l],grid.T[1,l]))
#plt.show()
#exit()
##with open("data/distances.pickle","wb") as f:
    #pickle.dump(distances,f)
with open("data/distances.pickle","rb") as f:
    distances = pickle.load(f)

#with open("data/layertemp.pickle","wb") as f:
    #pickle.dump(layertemp,f)
with open("data/layertemp.pickle","rb") as f:
    layertemp = pickle.load(f)


#with open("data/hub_shelf_thermals.pickle","wb") as f:
    #pickle.dump(glibheats,f)
#with open("data/hub_shelf_thermals.pickle","rb") as f:
    #glibheats = pickle.load(f)
#glibheats = layertemp
#with open("data/hub_shelf_salts.pickle","wb") as f:
    #pickle.dump(hubsalts,f)
with open("data/hub_shelf_salts.pickle","rb") as f:
    hubsalts = pickle.load(f)

#with open("data/cdwdepths.pickle","wb") as f:
    #pickle.dump(cdwdepths,f)
#with open("data/cdwdepths.pickle","rb") as f:
    #cdwdepths = pickle.load(f)

#with open("data/gprimes.pickle","wb") as f:
    #pickle.dump(gprimes,f)
#with open("data/gprimes.pickle","rb") as f:
    #gprimes = pickle.load(f)

#with open("data/isopycnaldepths.pickle","wb") as f:
    #pickle.dump(isopycnaldepths,f)
with open("data/isopycnaldepths.pickle","rb") as f:
    isopycnaldepths = pickle.load(f)

#slopes_by_shelf = cdw.slope_by_shelf(bedmach,polygons)

#with open("data/slopes_by_shelf.pickle","wb") as f:
    #pickle.dump(slopes_by_shelf,f)
with open("data/slopes_by_shelf.pickle","rb") as f:
    slopes_by_shelf = pickle.load(f)

#volumes_by_shelf = cdw.volumes_by_shelf(bedmach,polygons)

#with open("data/volumes_by_shelf.pickle","wb") as f:
    #pickle.dump(volumes_by_shelf,f)
with open("data/volumes_by_shelf.pickle","rb") as f:
    volumes_by_shelf = pickle.load(f)

#polynas = cdw.polyna_bathtub(bedmach,grid,GLIBregions)
#with open("data/polynasbt.pickle","wb") as f:
    #pickle.dump(polynas,f)
with open("data/polynas.pickle","rb") as f:
    polynas = pickle.load(f)

#polynas = cdw.polyna_bathtub(bedmach,grid,GLIBregions)
#with open("data/polynasbt.pickle","wb") as f:
    #pickle.dump(polynas,f)
#with open("data/polynas.pickle","rb") as f:
    #polynas = pickle.load(f)

#glibheats = cdw.closestMethodologyFig(bedmach,grid,physical,glibs,closest_points,sal,temp,shelf_keys,quant="glibheat",debug=False)

#glibheats = np.asarray(glibheats)*np.asarray(isopycnaldepths)
print("hello")
projection = pyproj.Proj("epsg:3031")
fs = []
for x,y in physical:
        lon,lat = projection(x,y,inverse=True)
        fs.append(1/np.abs(gsw.f(lat)))

polynas=np.asarray(polynas)

fs_by_shelf = bt.shelf_sort(shelf_keys,fs)
hubheats_by_shelf = bt.shelf_sort(shelf_keys,np.asarray(glibheats))
salts_by_shelf = bt.shelf_sort(shelf_keys,hubsalts)
polyna_by_shelf = bt.shelf_sort(shelf_keys,polynas)
#slopes_by_shelf = bt.shelf_sort(shelf_keys,slopes)
cdws_by_shelf = bt.shelf_sort(shelf_keys,cdwdepths)
gprimes_by_shelf = bt.shelf_sort(shelf_keys,gprimes)
isopycnaldepth_by_shelf = bt.shelf_sort(shelf_keys,isopycnaldepths)
gldepths_by_shelf = bt.shelf_sort(shelf_keys,depths)
glibs_by_shelf = bt.shelf_sort(shelf_keys,glibs)
distances_by_shelf = bt.shelf_sort(shelf_keys,glibs)
hubdeltas_by_shelf = bt.shelf_sort(shelf_keys,np.asarray(depths)-np.asarray(baths))
#rignot_shelf_massloss,rignot_shelf_areas,sigma = cdw.extract_rignot_massloss2019("data/rignot2019.xlsx")
#rignot_shelf_massloss,sigmas_by_shelf = cdw.extract_rignot_massloss2013("data/rignot2013.xlsx")
#rignot_shelf_massloss = bt.shelf_mass_loss('data/amundsilli.h5',polygons)
#with open("data/new_massloss.pickle","wb") as f:
    #pickle.dump(rignot_shelf_massloss,f)
#with open("data/new_massloss.pickle","rb") as f:
    #rignot_shelf_massloss = pickle.load(f)
rignot_shelf_massloss,shelf_areas,sigmas_by_shelf =  cdw.extract_adusumilli("data/Adusumilli.csv")
#rignot_shelf_massloss =  bt.shelf_mass_loss("",polygons)

#with open("data/onlypositivemelt.pickle","wb") as f:
    #pickle.dump(rignot_shelf_massloss,f)
#with open("data/onlypositivemelt.pickle","rb") as f:
    #rignot_shelf_massloss = pickle.load(f)



#print(rignot_shelf_massloss.keys())


#with open("data/hub_shelf_massloss.pickle","wb") as f:
    #pickle.dump(slopes_by_shelf,f)
#with open("data/hub_shelf_massloss.pickle","rb") as f:
    #rignot_shelf_massloss = pickle.load(f)
#
#
##with open("data/cdwdiff_hub_shelf_massloss.pickle","wb") as f:
    ##pickle.dump(slopes_by_shelf,f)
#with open("data/cdwdiff_hub_shelf_massloss.pickle","rb") as f:
    #slopes_by_shelf = pickle.load(f)

thermals=[]
cdws = []
isopycnals = []
ys = []
gldepths = []
glibshelf=[]
gprimes=[]
distances = []
hubdeltas = []
polynas = []
gllengths=[]
salts = []


znews= [] 
bars = []
areas = []
mys = []
slopes = []
volumes = []
fs = []
sigmas = []
#
##
##with open("data/MSDS.pickle","rb") as f:
    ##MSDS = pickle.load(f)
#
labels = []
for k in slopes_by_shelf.keys():
    if (k in rignot_shelf_massloss and ~np.isnan(rignot_shelf_massloss[k]) and k!="George_VI")or k =="Amery" :
        print(k)
        x,y = (polygons[k][0].centroid.x,polygons[k][0].centroid.y)
        slopes.append(slopes_by_shelf[k])
        volumes.append(volumes_by_shelf[k])
        labels.append(k)
        fs.append(np.nanmean(fs_by_shelf[k]))
        thermals.append(np.nanmean(hubheats_by_shelf[k]))
        cdws.append(np.nanmean(cdws_by_shelf[k]))
        isopycnals.append(np.nanmean(isopycnaldepth_by_shelf[k]))
        gldepths.append(np.nanmean(gldepths_by_shelf[k]))
        gprimes.append(np.nanmean(gprimes_by_shelf[k]))
        glibshelf.append(np.nanmean(glibs_by_shelf[k]))
        distances.append(np.nanmean(distances_by_shelf[k]))
        hubdeltas.append(np.nanmean(hubdeltas_by_shelf[k]))
        polynas.append(np.nanmean(polyna_by_shelf[k]))
        salts.append(np.nanmean(salts_by_shelf[k]))
        if k == "Amery":
            sigmas.append(0.7)
            areas.append(0)
            mys.append(0.8)
        else:
            sigmas.append(sigmas_by_shelf[k])
            areas.append(shelf_areas[k])
            mys.append(rignot_shelf_massloss[k])

#plt.scatter(np.asarray(thermals)*np.asarray(cdws)*slopes,mys,c=gldepths)
#plt.errorbar(np.asarray(thermals)*np.asarray(cdws)*slopes,mys,yerr=sigmas,ls='none')
#plt.show()

# plt.scatter(np.asarray(thermals),mys)
# for k in range(len(labels)):
#     plt.annotate(labels[k],(np.asarray(thermals[k]),mys[k]))
#
# plt.show()
#
# plt.scatter(np.asarray(thermals)*np.asarray(cdws)*np.asarray(gprimes)*fs,mys)
# for k in range(len(labels)):
#     plt.annotate(labels[k],(np.asarray(thermals)[k]*np.asarray(cdws)[k]*np.asarray(gprimes)[k]*fs[k],mys[k]))
# plt.show()
#
#
# plt.scatter(np.asarray(thermals)*np.asarray(cdws)*np.asarray(gprimes)*fs*np.asarray(slopes),mys)
# for k in range(len(labels)):
#     plt.annotate(labels[k],(slopes[k]*np.asarray(thermals)[k]*np.asarray(cdws)[k]*np.asarray(gprimes)[k]*fs[k],mys[k]))
# plt.show()
#




if False:
    plt.scatter(np.asarray(thermals),mys)
    plt.errorbar(np.asarray(thermals),mys,yerr=sigmas,ls='none')
    for k in range(len(labels)):
        plt.annotate(labels[k],(np.asarray(thermals[k]),mys[k]))
    plt.xlabel("t")
    plt.show()

    plt.scatter(np.asarray(thermals)*np.asarray(isopycnals),mys)
    plt.errorbar(np.asarray(thermals)*np.asarray(isopycnals),mys,yerr=sigmas,ls='none')
    for k in range(len(labels)):
        plt.annotate(labels[k],(np.asarray(thermals[k])*isopycnals[k],mys[k]))
    plt.xlabel("t * delta H")
    plt.show()

    plt.scatter(np.asarray(thermals)*np.asarray(isopycnals)*slopes,mys)
    plt.errorbar(np.asarray(thermals)*np.asarray(isopycnals)*slopes,mys,yerr=sigmas,ls='none')
    plt.xlabel("t * delta H * slope")
    for k in range(len(labels)):
        plt.annotate(labels[k],(np.asarray(thermals[k])*isopycnals[k]*slopes[k],mys[k]))
    plt.show()

    plt.scatter(np.asarray(thermals)*fs*np.asarray(isopycnals)*slopes,mys)
    plt.errorbar(np.asarray(thermals)*fs*np.asarray(isopycnals)*slopes,mys,yerr=sigmas,ls='none')
    plt.xlabel("t * delta H * slope /f")
    for k in range(len(labels)):
        plt.annotate(labels[k],(np.asarray(thermals[k])*fs[k]*isopycnals[k]*slopes[k],mys[k]))
    plt.show()



#
# fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
# ax1.scatter(np.asarray(thermals),mys,c=salts)
# ax1.set_xlabel("Avg temp at HUB")
# ax1.set_ylabel("Melt (m/yr)")
# ax2.scatter(np.asarray(cdws),mys,c=thermals)
# ax2.set_xlabel("Avg depth of thermocline above HUB * delta t")
# ax3.scatter(np.asarray(slopes),mys,c=thermals)
# ax3.set_xlabel("Avg depth of thermocline above HUB * delta t")
#plt.plot(range(30),range(30))
#plt.show()
print("amde it here")
areas = np.asarray(areas)
slopes = np.asarray(slopes)
glibshelf = np.asarray(glibshelf)
cdws = np.asarray(cdws)
#plt.hist(gprimes)
#plt.show()
melts = cdws*np.asarray(thermals)*np.asarray(fs)*(np.asarray(gprimes))*slopes
print("cdws",np.nanmean(cdws),np.nanstd(cdws))
print("thermals",np.nanmean(thermals),np.nanstd(thermals))
print("fs",np.nanmean(fs),np.nanstd(fs))
print("gprimes",np.nanmean(gprimes),np.nanstd(gprimes))
print("slopes",np.nanmean(slopes),np.nanstd(slopes))
melts=melts
mys=np.asarray(mys)
print("now here")
#melts = np.asarray(thermals)*slopes

#plt.scatter(melts*(rho0*spy*Cp*C)/(melten*kgtom),mys)
plt.rc('axes', titlesize=24)     # fontsize of the axes title
xs = np.asarray(([melts])).reshape((-1, 1))
model = LinearRegression().fit(xs, mys)
print("regressed")
r2 = model.score(xs,mys)
print(model.coef_)
rho0 = 1025
rhoi = 910
Cp = 4186
If = 334000
print(model.coef_)
C = model.coef_
print("C: ",C)
#W0 = (rho0*Cp)/(rhoi*If*C)
W0 =  100000#(rho0*Cp)/(rhoi*If*C)
alpha =  C/((rho0*Cp)/(rhoi*If*W0))
print("alpha: ", alpha)
melts = model.predict(xs)
ax = plt.gca()
mys = np.asarray(mys)
sigmas = np.asarray(sigmas)
thermals = np.asarray(thermals)
icedens = 917
gigatonconv = 10**(-12)
scale = icedens*gigatonconv*10**6
gldepths = np.asarray(gldepths)
#print(np.nansum(melts*areas)*scale,np.nansum(mys*areas)*scale,np.nansum(sigmas*areas)*scale)
#ax.scatter(melts[melts>4]*areas[melts>4]*scale,mys[melts>4]*areas[melts>4]*scale,c="red")
#ax.scatter(melts[melts<4]*areas[melts<4]*scale,mys[melts<4]*areas[melts<4]*scale)
#ax.scatter(slopes[melts>4]*areas[melts<4],mys[melts>4]*areas[melts<4],c="red")

thresh=4
coldxs = np.asarray(([slopes[melts<4]*areas[melts<4]])).reshape((-1, 1))
coldmodel = LinearRegression().fit(coldxs, mys[melts<4]*areas[melts<4])

coldmelts = coldmodel.predict(coldxs)

ax.scatter(melts[melts>4]*areas[melts>4]*scale,mys[melts>4]*areas[melts>4]*scale,c="red")
ax.scatter(coldmelts*scale,mys[melts<4]*areas[melts<4]*scale,c="blue")

#ax.scatter(slopes[melts<4]*areas[melts<4],mys[melts<4]*areas[melts<4])
#ax.scatter(slopes[melts<thresh]*thermals[melts<thresh],mys[melts<thresh],c=thermals[melts<thresh])
fs = np.asarray(fs)
gprimes = np.asarray(gprimes)
#c=ax.scatter(slopes[melts<thresh],mys[melts<thresh])#,c=glibshelf[melts<thresh])
#plt.colorbar(c)

markers, caps, bars = ax.errorbar(melts[melts>4]*areas[melts>4]*scale,mys[melts>4]*areas[melts>4]*scale,yerr=sigmas[melts>thresh]*areas[melts>thresh]*scale,ls='none')
[bar.set_alpha(0.25) for bar in bars]
markers, caps, bars = ax.errorbar(coldmelts*scale,mys[melts<4]*areas[melts<4]*scale,c="blue",yerr=sigmas[melts<thresh]*areas[melts<thresh]*scale,ls='none')
[bar.set_alpha(0.25) for bar in bars]

finalxs = np.concatenate((melts[melts>4]*areas[melts>4]*scale,coldmelts*scale))
finalys = np.concatenate((mys[melts>4]*areas[melts>4]*scale,mys[melts<4]*areas[melts<4]*scale))
finalxs = finalxs.reshape((-1, 1))
finalmodel = LinearRegression().fit(finalxs,finalys)

finalr2 = finalmodel.score(finalxs,finalys)
print("MASSLOSS SUMS")
print(np.nansum(finalxs),np.nansum(finalys))
print(len(gldepths))
print(len(mys))
print(len(mys))
#ax.scatter(gldepths[melts<thresh],mys[melts<thresh])
#ax.errorbar(slopes[melts<thresh]*areas[melts<thresh],mys[melts<thresh]*areas[melts<thresh],yerr=sigmas[melts<thresh]*areas[melts<thresh],ls='none')
#ax.errorbar(slopes[melts<thresh],mys[melts<thresh],yerr=sigmas[melts<thresh],ls='none')
#ax.errorbar(areas[melts<thresh],mys[melts<thresh],yerr=sigmas[melts<thresh],ls='none')
#ax.set_xlim(0,5)
#ax.set_ylim(0,5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
#ax.plot(range(30),range(30))
#ax.plot(np.asarray(range(29))*10**10,np.asarray(range(30))*10**10)
ax.text(.05, .95, '$r^2=$'+str(round(finalr2,2)), ha='left', va='top', transform=plt.gca().transAxes,fontsize=12)
#ax.set_xlabel(r"$\dot{m}_{\mathrm{pred}} (m/yr)$",fontsize=18)
#ax.set_ylabel(r'$\dot{m}_{\mathrm{obs}} (m/yr)$',fontsize=18)
ax.set_ylabel(r"Observed basal mass loss (Gt/yr)",fontsize=16)
ax.set_xlabel(r'Predicted combined warm+cold shelf theory (Gt/yr)',fontsize=16)
ax.plot((0,175),(0,175))
labels = np.asarray(labels)
coldlabels = labels[melts<thresh]
warmlabels = labels[melts>thresh]
for k in range(len(coldlabels)):
    if coldmelts[k]*scale>39:
        text= ax.annotate(coldlabels[k],(coldmelts[k]*scale,mys[melts<thresh][k]*areas[melts<thresh][k]*scale))
        #text.set_alpha(.4)

for k in range(len(warmlabels)):
    if melts[melts>thresh][k]*areas[melts>thresh][k]*scale > 39:
        text = ax.annotate(warmlabels[k],(melts[melts>thresh][k]*areas[melts>thresh][k]*scale,mys[melts>thresh][k]*areas[melts>thresh][k]*scale))
        #text.set_alpha(.4)
     #if melts[k]<thresh:
        #ax.annotate(labels[k],(coldmelts[k]*scale,mys[k]*areas[k]*scale))
     #else:
        #ax.annotate(labels[k],(melts[k]*areas[k]*scale,mys[k]*areas[k]*scale))
        #ax.annotate(labels[k],(gldepths[k],mys[k]))

plt.show()
plt.scatter(slopes[melts<4]*areas[melts<4],mys[melts<4]*areas[melts<4]*scale)
markers, caps, bars = plt.errorbar(slopes[melts<thresh]*areas[melts<4],mys[melts<thresh]*areas[melts<4]*scale,yerr=sigmas[melts<thresh]*areas[melts<4]*scale,ls='none')
[bar.set_alpha(0.5) for bar in bars]
for k in range(len(coldlabels)):
    if slopes[melts<thresh][k]*areas[melts<thresh][k]>77:
        plt.annotate(coldlabels[k],(slopes[melts<thresh][k]*areas[melts<thresh][k],mys[melts<thresh][k]*areas[melts<thresh][k]*scale))

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(r"Observed basal mass loss (Gt/yr)",fontsize=16)
plt.xlabel(r'Median ice shelf slope * area (m)',fontsize=16)
plt.show()

if False:
    oldxs = xs
    xs = model.predict(xs)
    tempterms = cdws*np.asarray(thermals)*np.asarray(fs)*np.asarray(gprimes)
    x = np.linspace(np.min(tempterms)*0.95,np.max(tempterms)*1.05,20)
    y = np.linspace(np.min(slopes)*0.95,np.max(slopes)*1.05,20)
    X,Y = np.meshgrid(x,y)
    Z = np.multiply(X,Y)*model.coef_[0]+model.intercept_
    im = plt.pcolormesh(X,Y,Z,cmap="gnuplot",vmin=np.min(Z),vmax=np.max(Z))
    cbar = plt.colorbar(im)
    CS = plt.contour(X,Y,Z,levels=[1,2.5,5,10,15,20],colors="white")
    plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    #plt.xlabel(r"$(T_{\mathrm{CDW}}-T_{\mathrm{f}})*(H_{\mathrm{PYC}}-H_{\mathrm{HUB}})*\frac{g'}{f}$",fontsize=18)
    plt.xlabel(r"Hydrographic terms $(C m^{2} s^{-1})$",fontsize=18)
    #plt.ylabel(r'$S_{\mathrm{ice}}$',fontsize=20)
    plt.ylabel(r'Ice shelf slope $(m^{-1})$',fontsize=20)
    plt.scatter(tempterms,slopes,c="white")
    for k in range(len(labels)):
        plt.annotate(labels[k],(tempterms[k],slopes[k]),c="white")
    plt.show()
#xs=model.predict(xs)
#plt.errorbar(xs,mys,yerr=bars,fmt="o")
#plt.scatter(xs,mys,c=polynas)
#plt.errorbar(xs,mys,yerr=sigmas,ls="none")
#for k in range(len(labels)):
    #plt.annotate(labels[k],(xs[k],mys[k]))
#plt.plot(range(30),range(30))
###plt.colorbar()
#plt.show()

xs = np.asarray(xs)
polynas = np.asarray(polynas)
mys = np.asarray(mys)
labels = np.asarray(labels)
distances = np.asarray(distances)
gldepths = np.asarray(gldepths)
glibs = np.asarray(glibshelf)
sigmas = np.asarray(sigmas)
slopes = np.asarray(slopes)
areas = np.asarray(areas)
volumes = np.asarray(volumes)
thermals = np.asarray(thermals)
hubdeltas = np.asarray(hubdeltas)
thresh = 3

lxs,Ms = LazeromsM()

glmax = 400#np.nanmax(np.abs(gldepths))
ms = []

for i in gldepths:
    print(i,glmax,np.abs(i/glmax))
    ms.append(np.nanmean(Ms[lxs<(np.abs(i/glmax))]))

plt.scatter(np.asarray(ms)[xs<thresh]*glibshelf[xs<thresh],np.asarray(mys)[xs<thresh],c=gldepths[xs<thresh])
plt.colorbar()
plt.show()

# plt.scatter(polynas[xs<2.5]/areas[xs<2.5],mys[xs<2.5],c=polynas[xs<2.5])
# plt.errorbar(xs[xs<2.5]/areas[xs<2.5],mys[xs<2.5],yerr=sigmas[xs<2.5],ls="none")
# for k in range(len(labels[xs<2.5])):
#     plt.annotate(labels[xs<2.5][k],(polynas[xs<2.5][k]/areas[xs<2.5][k],mys[xs<2.5][k]))
freezingtemps = gsw.CT_freezing(34.7,np.abs(gldepths),0)
distances = distances/np.nanmax(distances)
#newxs = ((e^(-distances)))[xs<thresh]/distances[xs<thresh]
ms=np.linspace(0,1,1000)
ms = 1 / (2*np.sqrt(2)) * (3*(1 - ms)**(4/3) - 1) * np.sqrt(1 - (1 - ms)**(4/3))
M = np.cumsum(ms)[::-1]
dnormal = n-p.abs(distances)/(np.nanmax(np.abs(distances)))*0.6
newxs = []
for i in dnormal:
    newxs.append(M[int(i*1000)+400-1]/i)
newxs = np.asarray(gldepths)
newxs=newxs[xs<thresh]
#newxs = -((1-(np.exp(-distances/(np.nanmax(distances))))))[xs<thresh]/(-distances[xs<thresh]/(np.nanmax(distances[xs<thresh])))
plt.scatter(newxs,mys[xs<thresh],c=xs[xs<thresh],cmap="jet")
plt.errorbar(newxs,mys[xs<thresh],yerr=sigmas[xs<thresh],ls="none")
for k in range(len(labels[xs<thresh])):
    plt.annotate(labels[xs<thresh][k],(newxs[k],mys[xs<thresh][k]))
# ##plt.colorbar()
plt.show()
#
#
