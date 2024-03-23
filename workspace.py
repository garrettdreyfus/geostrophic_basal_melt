import bathtub as bt
import pickle
import GLIB
import gsw,xarray, pyproj
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import woa
import matplotlib.colors as colors
import cdw

# Create GLIB

writeBedMach = False
writeShelfNumbers = False
writeGLIB = False
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
glibs = []
for l in range(len(baths)):
    depths.append(bedvalues[grid[l][0],grid[l][1]])
    glibs.append(baths[l])
    if np.isnan(baths[l]):
        baths[l]=bedvalues[grid[l][0],grid[l][1]]
        
#closest_hydro = cdw.closestHydro(bedmach,grid,physical,closest_points,sal,temp,shelf_keys)

#with open("data/closest_hydro_woathree.pickle","wb") as f:
    #pickle.dump(closest_hydro,f)
with open("data/closest_hydro_woathree.pickle","rb") as f:
    closest_hydro = pickle.load(f)

#avg_s,avg_t,depths = cdw.averageForShelf("Thwaites",bedmach,grid,physical,glibs,closest_hydro,sal,temp,shelf_keys,quant="glibheat",debug=False)
#with open("data/ThwaitesAverages.pickle","wb") as f:
    #pickle.dump((avg_t,avg_s,depths),f)


shelf_areas = bt.shelf_areas()
glibheats,cdwdepths,gprimes = cdw.revampedClosest(bedmach,grid,physical,glibs,closest_hydro,sal,temp,shelf_keys,quant="glibheat",debug=False)
physical = np.asarray(physical)
grid = np.asarray(grid)

with open("data/stats_woa.pickle","wb") as f:
    pickle.dump((glibheats,cdwdepths,gprimes),f)
with open("data/stats_woa.pickle","rb") as f:
    (glibheats,cdwdepths,gprimes) = pickle.load(f)

#with open("data/glibheats_gissr2.pickle","rb") as f:
    #glibheats = pickle.load(f)
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
hubheats_by_shelf = bt.shelf_sort(shelf_keys,glibheats)
cdws_by_shelf = bt.shelf_sort(shelf_keys,cdwdepths)
gprimes_by_shelf = bt.shelf_sort(shelf_keys,gprimes)
gldepths_by_shelf = bt.shelf_sort(shelf_keys,depths)
glibs_by_shelf = bt.shelf_sort(shelf_keys,glibs)
rignot_shelf_massloss,shelf_areas,sigmas_by_shelf =  cdw.extract_adusumilli("data/Adusumilli.csv")


thermals=[]
cdws = []
gldepths = []
glibshelf=[]
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
        glibshelf.append(np.nanmean(glibs_by_shelf[k]))
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
glibshelf = np.asarray(glibshelf)
cdws = np.asarray(cdws)
print(np.shape(cdws))
melts = cdws*np.asarray(thermals)*np.asarray(gprimes)*np.asarray(slopes)*np.asarray(fs)


avmelts=np.mean(melts,axis=1)
mys=np.asarray(mys)

#plt.scatter(melts*(rho0*spy*Cp*C)/(melten*kgtom),mys)
plt.rc('axes', titlesize=24)     # fontsize of the axes title
xs = np.asarray(([avmelts])).reshape((-1, 1))
model = LinearRegression().fit(xs, mys)
print("regressed")
r2 = model.score(xs,mys)
print("coef: ",r2)
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
plt.scatter(melts,mys)
for k in range(len(labels)):
    text= plt.annotate(labels[k],(melts[k],mys[k]))

plt.show()
warmC = model.coef_
thresh=3
finalproduct = np.empty(np.shape(melts))
finalproduct[:]=melts

icedens = 917
gigatonconv = 10**(-12)
scale = icedens*gigatonconv*10**6
avmelts = avmelts*warmC
flatslopes = np.mean(slopes,axis=1)
flatareas = np.mean(areas,axis=1)
coldxs = np.asarray(([flatslopes[avmelts<thresh]*flatareas[avmelts<thresh]])).reshape((-1, 1))
coldmodel = LinearRegression().fit(coldxs, mys[avmelts<thresh]*flatareas[avmelts<thresh])
coldC = coldmodel.coef_
#plt.scatter(flatslopes[avmelts<thresh]*flatareas[avmelts<thresh]*coldC*scale,mys[avmelts<thresh]*flatareas[avmelts<thresh]*scale)
plt.scatter(avmelts,mys,c=np.mean(thermals,axis=1))
#plt.plot(range(150),range(150))
plt.show()
plt.scatter(flatslopes[avmelts<thresh]*flatareas[avmelts<thresh]*coldC*scale,mys[avmelts<thresh]*flatareas[avmelts<thresh]*scale,c="blue")
plt.scatter(avmelts[avmelts>thresh]*flatareas[avmelts>thresh]*scale,mys[avmelts>thresh]*flatareas[avmelts>thresh]*scale,c="red")

xs = []
ys = []

sigmas=np.asarray(sigmas)
print(len(sigmas))
print(len(flatareas))
print(sigmas[avmelts>thresh]*flatareas[avmelts>thresh]*scale)
markers, caps, bars = plt.errorbar(flatslopes[avmelts<thresh]*flatareas[avmelts<thresh]*coldC*scale,mys[avmelts<thresh]*flatareas[avmelts<thresh]*scale,yerr=list(sigmas[avmelts<thresh]*flatareas[avmelts<thresh]*scale),ls='none')
[bar.set_alpha(0.25) for bar in bars]
markers, caps, bars = plt.errorbar(avmelts[avmelts>thresh]*flatareas[avmelts>thresh]*scale,mys[avmelts>thresh]*flatareas[avmelts>thresh]*scale,c="blue",yerr=list(sigmas[avmelts>thresh]*flatareas[avmelts>thresh]*scale),ls='none')
[bar.set_alpha(0.25) for bar in bars]


for k in range(len(labels)):
    if avmelts[k]>thresh:
        text= plt.annotate(labels[k],(avmelts[k]*flatareas[k]*scale,mys[k]*flatareas[k]*scale))
        xs.append(avmelts[k]*flatareas[k]*scale)
        ys.append(mys[k]*flatareas[k]*scale)
    else:
        text= plt.annotate(labels[k],(flatslopes[k]*flatareas[k]*coldC*scale,mys[k]*flatareas[k]*scale))
        xs.append(flatslopes[k]*flatareas[k]*coldC*scale)
        ys.append(mys[k]*flatareas[k]*scale)

xs,ys = np.asarray(xs),np.asarray(ys)
xs = np.asarray((xs).reshape((-1, 1)))
coldmodel = LinearRegression().fit(xs, ys)
print(coldmodel.score(xs,ys))
plt.show()
finalproduct[melts*warmC<=thresh] = slopes[melts*warmC<=thresh] * areas[melts*warmC<=thresh]* coldC
finalproduct[melts*warmC>thresh] = melts[melts*warmC>thresh]*areas[melts*warmC>thresh]*warmC
finalproduct = -finalproduct*scale

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
for i in range(len(labels)):
    ax1.plot(cdw.moving_average(1990+(np.asarray(range(240))/12),24),cdw.moving_average(finalproduct[i],24),label=labels[i])
    ax2.plot(cdw.moving_average(1990+(np.asarray(range(240))/12),24),cdw.moving_average(thermals[i],24),label=labels[i])
    ax3.plot(cdw.moving_average(1990+(np.asarray(range(240))/12),24),cdw.moving_average(cdws[i],24),label=labels[i])
    ax4.plot(cdw.moving_average(1990+(np.asarray(range(240))/12),24),cdw.moving_average(gprimes[i],24),label=labels[i])
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
plt.show()



fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
ax1.plot(1990+(np.asarray(range(240))/12),-np.sum(finalproduct,axis=0))
ax1.plot(cdw.moving_average(1990+(np.asarray(range(240))/12),24),cdw.moving_average(-np.sum(finalproduct,axis=0),24))
#ax1.plot(range(3),-np.sum(finalproduct,axis=0))
#ax1.set_title("basal mass loss (gt/yr)")
ax2.plot(1990+(np.asarray(range(240))/12),np.mean(thermals,axis=0))
#ax2.plot(range(3),np.mean(thermals,axis=0))
ax2.plot(cdw.moving_average(1990+(np.asarray(range(240))/12),24),cdw.moving_average(np.mean(thermals,axis=0),24))
#ax2.set_title("offshore temp above freezing(C)")
ax3.plot(1990+(np.asarray(range(240))/12),np.mean(cdws,axis=0))
#ax3.plot(range(3),np.mean(cdws,axis=0))
ax3.plot(cdw.moving_average(1990+(np.asarray(range(240))/12),24),cdw.moving_average(np.mean(cdws,axis=0),24))
#ax3.set_title("offshore Hcdw (m)")
ax4.plot(1990+(np.asarray(range(240))/12),np.mean(gprimes,axis=0))
#ax4.plot(range(3),np.mean(gprimes,axis=0))
ax4.plot(cdw.moving_average(1990+(np.asarray(range(240))/12),24),cdw.moving_average(np.mean(gprimes,axis=0),24))
#ax4.set_title("offshore gprime ")
#plt.xlim(1994,2010)
plt.show()
exit()

#rho0 = 1025
#rhoi = 910
#Cp = 4186
#If = 334000
#print(model.coef_)
#C = model.coef_
#print("C: ",C)
##W0 = (rho0*Cp)/(rhoi*If*C)
#W0 =  100000#(rho0*Cp)/(rhoi*If*C)
#alpha =  C/((rho0*Cp)/(rhoi*If*W0))
#print("alpha: ", alpha)
#melts = model.predict(xs)
#ax = plt.gca()
#mys = np.asarray(mys)
#sigmas = np.asarray(sigmas)
#thermals = np.asarray(thermals)
#icedens = 917
#gigatonconv = 10**(-12)
#scale = icedens*gigatonconv*10**6
#gldepths = np.asarray(gldepths)
#print(np.nansum(melts*areas)*scale,np.nansum(mys*areas)*scale,np.nansum(sigmas*areas)*scale)
#ax.scatter(melts[melts>thresh]*areas[melts>thresh]*scale,mys[melts>thresh]*areas[melts>thresh]*scale,c="red")
#ax.scatter(melts[melts<thresh]*areas[melts<thresh]*scale,mys[melts<thresh]*areas[melts<thresh]*scale)
#ax.scatter(slopes[melts>thresh]*areas[melts<thresh],mys[melts>thresh]*areas[melts<thresh],c="red")

thresh=0
coldxs = np.asarray(([slopes[melts<thresh]*areas[melts<thresh]])).reshape((-1, 1))
coldmodel = LinearRegression().fit(coldxs, mys[melts<thresh]*areas[melts<thresh])

coldmelts = coldmodel.predict(coldxs)

ax.scatter(melts[melts>thresh]*areas[melts>thresh]*scale,mys[melts>thresh]*areas[melts>thresh]*scale,c="red")
ax.scatter(coldmelts*scale,mys[melts<thresh]*areas[melts<thresh]*scale,c="blue")

#ax.scatter(slopes[melts<thresh]*areas[melts<thresh],mys[melts<thresh]*areas[melts<thresh])
#ax.scatter(slopes[melts<thresh]*thermals[melts<thresh],mys[melts<thresh],c=thermals[melts<thresh])
fs = np.asarray(fs)
gprimes = np.asarray(gprimes)
#c=ax.scatter(slopes[melts<thresh],mys[melts<thresh])#,c=glibshelf[melts<thresh])
#plt.colorbar(c)

markers, caps, bars = ax.errorbar(melts[melts>thresh]*areas[melts>thresh]*scale,mys[melts>thresh]*areas[melts>thresh]*scale,yerr=sigmas[melts>thresh]*areas[melts>thresh]*scale,ls='none')
[bar.set_alpha(0.25) for bar in bars]
markers, caps, bars = ax.errorbar(coldmelts*scale,mys[melts<thresh]*areas[melts<thresh]*scale,c="blue",yerr=sigmas[melts<thresh]*areas[melts<thresh]*scale,ls='none')
[bar.set_alpha(0.25) for bar in bars]

finalxs = np.concatenate((melts[melts>thresh]*areas[melts>thresh]*scale,coldmelts*scale))
finalys = np.concatenate((mys[melts>thresh]*areas[melts>thresh]*scale,mys[melts<thresh]*areas[melts<thresh]*scale))
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
plt.scatter(slopes[melts<thresh]*areas[melts<thresh],mys[melts<thresh]*areas[melts<thresh]*scale)
markers, caps, bars = plt.errorbar(slopes[melts<thresh]*areas[melts<thresh],mys[melts<thresh]*areas[melts<thresh]*scale,yerr=sigmas[melts<thresh]*areas[melts<thresh]*scale,ls='none')
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
#polynas = np.asarray(polynas)
mys = np.asarray(mys)
labels = np.asarray(labels)
#distances = np.asarray(distances)
gldepths = np.asarray(gldepths)
glibs = np.asarray(glibshelf)
sigmas = np.asarray(sigmas)
slopes = np.asarray(slopes)
areas = np.asarray(areas)
#volumes = np.asarray(volumes)
thermals = np.asarray(thermals)
#hubdeltas = np.asarray(hubdeltas)
thresh = 5

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
#distances = distances/np.nanmax(distances)
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
