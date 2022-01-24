import shapefile
import rockhound as rh
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import itertools
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import time


myshp = open("regions/IceBoundaries_Antarctica_v02.shp", "rb")
mydbf = open("regions/IceBoundaries_Antarctica_v02.dbf", "rb")
myshx = open("regions/IceBoundaries_Antarctica_v02.shx", "rb")
r = shapefile.Reader(shp=myshp, dbf=mydbf, shx=myshx)
s = r.shapes()
records = r.shapeRecords()
polygons =  {}

for i in range(len(s)):
    l = s[i]
    name = records[i].record[0]
    kind = records[i].record[3]
    if l.shapeTypeName == 'POLYGON' and kind== "FL":
        xs, ys = zip(*l.points)
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

def GLB_search(shelf,start,polygons,stepsize=10):
    depth = np.asarray(shelf.bed.values)
    ice = np.asarray(shelf.icemask_grounded_and_shelves.values)

    moves = list(itertools.product(*[[-stepsize,0,stepsize],[-stepsize,0,stepsize]]))

    #the whole ordering thing confused me so I'm reversing
    start = start[::-1]

    previouslocations=[]
    locations = [start]
    alllocations = []
    icestatus = []
    bath = depth[start[0],start[1]]

    p = [shelf.x[start[1]],shelf.y[start[0]]]
    cn, cp, distance = closest_shelf(p,polygons)
    while True:
        while len(locations)>0:
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
                            return True,bath-20, previouslocations#[[shelf.x[start[1]],shelf.y[start[0]]],[shelf.x[nextmove[1]],shelf.y[nextmove[0]]]]
        print(".")
        if bath>0:
            return False, 0, []
        locations = alllocations
        previouslocations = alllocations
        alllocations = []
        bath = bath+20





def highlight_margin(shelf,polygons):
    margin_coords = []
    mx = []
    my = []
    icemask = np.asarray(shelf.icemask_grounded_and_shelves.values)
    shelf_names = polygons.keys()
    margin_frac ={}
    lines = []
    count=0
    for l in shelf_names:
        margin_frac[l] = [0,0]
    for i in tqdm(range(1,icemask.shape[0]-1)):
        for j in  range(1,icemask.shape[1]-1):
            if icemask[i][j] == 0:
                a = icemask[i+1][j]
                b = icemask[i-1][j]
                c = icemask[i][j+1]
                d = icemask[i][j-1]
                if (np.asarray([a,b,c,d])==1).any():
                    mx.append(shelf.x[j])
                    my.append(shelf.y[i])
                    #closestname, closestpolygon, _  = closest_shelf([shelf.x[j],shelf.y[i]],polygons)
                    print("Beginning Search at {},{}".format(i,j))
                    start = time.time()
                    connected, boundingbath, locations = GLB_search(shelf,[j,i],polygons)
                    end = time.time()
                    print("It took: {}".format(end-start))
                    print("*"*10)
                    if len(locations)>0:
                        plt.figure(figsize=(16, 14))
                        ax = plt.subplot(111)
                        pc = shelf.bed.plot.pcolormesh(
                            ax=ax, cmap=cmocean.cm.topo, cbar_kwargs=dict(pad=0.01, aspect=30)
                        )
                        plt.scatter(shelf.x[j],shelf.y[i],c="red")
                        locations = np.asarray(locations)
                        #plt.scatter(locations.T[0],locations.T[1])
                        pd = shelf.bed.plot.contour(
                            ax=ax,levels=[boundingbath],c="red")
                        plt.show()

    return mx,my

def trimDataset(bm,xbounds,ybounds):
    shelf=bm
    shelf = shelf.where(shelf.x<xbounds[1],drop=True)
    shelf = shelf.where(shelf.y<ybounds[1],drop=True)
    shelf = shelf.where(shelf.x>xbounds[0],drop=True)
    shelf = shelf.where(shelf.y>ybounds[0],drop=True)
    return shelf

bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
#FRIS
##Default
#xbounds=  [ -1.6*(10**6),-0.75*(10**6)]
#ybounds= [-0.5*(10**6), 2.5*(10**6)]
##Closer in
#xbounds=  [ -1.0*(10**6),-0.75*(10**6)]
#ybounds= [-0.5*(10**6), 2.5*(10**6)]
#xbounds=  [ -1.75*(10**6),-0*(10**6)]
#ybounds= [-1.5*(10**6), 3.5*(10**6)]
#PIG
xbounds=[ -2.4*(10**6),-1.4*(10**6)]
ybounds=[-1.4*(10**6), -0.2*(10**6)]

shelf = trimDataset(bedmap,xbounds,ybounds)
#shelf = bedmap

# plt.figure(figsize=(16, 14))
# ax = plt.subplot(111)
# pc = shelf.bed.plot.pcolormesh(
#     ax=ax, cmap=cmocean.cm.topo, cbar_kwargs=dict(pad=0.01, aspect=30)
# )
# #plt.scatter(shelf.x[j],shelf.y[i],c="red")
# pd = shelf.bed.plot.contour(
#     ax=ax,levels=[-494],c="red")
# plt.show()


print(np.nanmin(shelf.bed.values))
#mc,mx,my,lines = highlight_margin(shelf,polygons)
mx,my = highlight_margin(shelf,polygons)
pc = shelf.icemask_grounded_and_shelves.plot.pcolormesh(
  ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30)
)
plt.scatter(mx,my,c='red')
# for l in lines:
#     l = np.asarray(l).T
#     plt.plot(l[0],l[1],c="orange",alpha=0.5)

plt.show()
