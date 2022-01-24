import shapefile
import rockhound as rh
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import itertools
from tqdm import tqdm
from shapely.geometry import Polygon, Point
from stats import stats
import time


myshp = open("regions/IceBoundaries_Antarctica_v02.shp", "rb")
mydbf = open("regions/IceBoundaries_Antarctica_v02.dbf", "rb")
myshx = open("regions/IceBoundaries_Antarctica_v02.shx", "rb")
r = shapefile.Reader(shp=myshp, dbf=mydbf, shx=myshx)
s = r.shapes()
#r.shapes()[0].points
records = r.shapeRecords()

#bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
#plt.close()
##plt.figure(figsize=(16, 14))
#ax = plt.subplot(111)
#pc = (bedmap.icemask_grounded_and_shelves==0).plot.pcolormesh(
   #ax=ax, cmap=cmocean.cm.gray_r, cbar_kwargs=dict(pad=0.01, aspect=30),
#)
polygons =  {}
for i in range(len(s)):
    l = s[i]
    name = records[i].record[0]
    kind = records[i].record[3]
    if l.shapeTypeName == 'POLYGON' and kind== "FL":
        xs, ys = zip(*l.points)
        if stats[name][1] > 0:
            plt.fill(xs,ys,alpha=stats[name][0]/stats[name][1],c="red")
            polygons[name] = Polygon(l.points)
            plt.annotate(name,(np.mean(xs),np.mean(ys)))
plt.savefig("search.png")

def closest_shelf(coord,polygons):
    min_dist = np.inf
    closestname = None
    closestpolygon = None
    for i, (k, v) in enumerate(polygons.items()):
        dist = v.distance(Point(coord))
        if dist < min_dist:
            min_dist = dist
            closestname = k
            closestpolygon = v
    return closestname, closestpolygon, min_dist

def ocean_search(shelf,start,polygons,stepsize=10):
    depth = np.asarray(shelf.bed.values)
    ice = np.asarray(shelf.icemask_grounded_and_shelves.values)

    moves = list(itertools.product(*[[-stepsize,0,stepsize],[-stepsize,0,stepsize]]))

    #the whole ordering thing confused me so I'm reversing
    start = start[::-1]

    flag = False
    locations = [start]
    alllocations = []
    bath = depth[start[0],start[1]]+15
    baths = []
    for m in moves:
        nextmove = [start[0]+m[0],start[1]+m[1]]
        if 0 < nextmove[1] < len(shelf.x) and 0 < nextmove[0] < len(shelf.y):
            baths.append(depth[nextmove[0],nextmove[1]])
    bath=np.nanmedian(baths)
    p = [shelf.x[start[1]],shelf.y[start[0]]]
    cn, cp, distance = closest_shelf(p,polygons)
    while len(locations)>0 and not flag:
        currentlocation = locations.pop(0)
        for m in moves:
            nextmove = [currentlocation[0]+m[0],currentlocation[1]+m[1]]
            if 0 < nextmove[1] < len(shelf.x) and 0 < nextmove[0] < len(shelf.y)\
                    and depth[nextmove[0],nextmove[1]]<=bath and nextmove not in alllocations:
                if ice[nextmove[0],nextmove[1]]!=0:
                    locations.append(nextmove)
                    alllocations.append(nextmove)
                    p = [shelf.x[nextmove[1]],shelf.y[nextmove[0]]]
                    if np.isnan(ice[nextmove[0],nextmove[1]]) and cp.distance(Point(p))>10**6 and depth[nextmove[0],nextmove[1]] <bath:
                        return True, [[shelf.x[start[1]],shelf.y[start[0]]],[shelf.x[nextmove[1]],shelf.y[nextmove[0]]]]
    return False, []



def highlight_margin(shelf,polygons):
    margin_coords = []
    margin_x = []
    margin_y = []
    icemask = np.asarray(shelf.icemask_grounded_and_shelves.values)
    shelf_names = polygons.keys()
    margin_frac ={}
    lines = []
    count=0
    for l in shelf_names:
        margin_frac[l] = [0,0]
    for i in tqdm(range(1,icemask.shape[0]-1)):
        for j in  range(1,icemask.shape[1]-1):
            if icemask[i][j] == 1:
                a = icemask[i+1][j]
                b = icemask[i-1][j]
                c = icemask[i][j+1]
                d = icemask[i][j-1]
                if np.isnan([a,b,c,d]).any():
                    closestname, closestpolygon, _  = closest_shelf([shelf.x[j],shelf.y[i]],polygons)
                    start = time.time()
                    connected, trip = ocean_search(shelf,[j,i],polygons)
                    end = time.time()
                    if not connected:
                        margin_frac[closestname] = [margin_frac[closestname][0]+1,margin_frac[closestname][1]+1]
                        margin_coords.append(tuple([j,i]))
                        margin_x.append(shelf.x[j])
                        margin_y.append(shelf.y[i])
                    else:
                        lines.append(trip)
                        margin_frac[closestname] = [margin_frac[closestname][0],margin_frac[closestname][1]+1]
    print(margin_frac)
    return margin_coords, margin_x, margin_y, lines

def trimDataset(bm,xbounds,ybounds):
    shelf=bm
    shelf = shelf.where(shelf.x<xbounds[1],drop=True)
    shelf = shelf.where(shelf.y<ybounds[1],drop=True)
    shelf = shelf.where(shelf.x>xbounds[0],drop=True)
    shelf = shelf.where(shelf.y>ybounds[0],drop=True)
    return shelf

bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
#FRIS
xbounds=  [ -1.5*(10**6),-0.75*(10**6)]
ybounds= [-0.5*(10**6), 2.5*(10**6)]
#xbounds=  [ -1.75*(10**6),-0*(10**6)]
#ybounds= [-1.5*(10**6), 3.5*(10**6)]
#PIG
#xbounds=[ -2.4*(10**6),-1.4*(10**6)]
#ybounds=[-1.4*(10**6), -0.2*(10**6)]

shelf = trimDataset(bedmap,xbounds,ybounds)
#shelf = bedmap

plt.figure(figsize=(16, 14))
ax = plt.subplot(111)
print(np.nanmin(shelf.bed.values))
mc,mx,my,lines = highlight_margin(shelf,polygons)
pc = shelf.bed.plot.pcolormesh(
  ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30),vmin=-1000,vmax=0
)

plt.scatter(mx,my,c='red')
for l in lines:
    l = np.asarray(l).T
    plt.plot(l[0],l[1],c="orange",alpha=0.5)

plt.show()
