import shapefile
import rockhound as rh
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import itertools
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import time
from scipy.ndimage import label 


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

def GLB_search(shelf,start,polygons,distance_mask):
    depth = np.asarray(shelf.bed.values)
    ice = np.asarray(shelf.icemask_grounded_and_shelves.values)
    #the whole ordering thing confused me so I'm reversing
    previouslocations=[]
    bath = depth[start[1],start[0]]
    startbath=bath
    #indices = np.indices(depth)
    while True:
        below = depth<=bath
        regions, _ = label(below)
        if bath>0:
            return False, bath, []
        if (regions[start[1],start[0]] == regions[~distance_mask]).any():
            if bath==startbath:
                return False, bath, []
            else:
                return True, bath-20,previouslocations
        previouslocations = (regions == regions[start[1],start[0]])
        bath+=20

def get_grounding_line_points(shelf,polygons):
    margin_coords = []
    mx = []
    my = []
    icemask = np.asarray(shelf.icemask_grounded_and_shelves.values)
    beddepth = np.asarray(shelf.bed.values)
    shelf_names = polygons.keys()
    margin_frac ={}
    lines = []
    count=0
    for l in shelf_names:
        margin_frac[l] = [0,0]

    physical_cords = []
    grid_indexes = []
    depths=[]
    for i in range(1,icemask.shape[0]-1):
        for j in  range(1,icemask.shape[1]-1):
            if icemask[i][j] == 1:
                a = icemask[i+1][j]
                b = icemask[i-1][j]
                c = icemask[i][j+1]
                d = icemask[i][j-1]
                if (np.asarray([a,b,c,d])==0).any():
                    physical_cords.append([shelf.x[j],shelf.y[i]])
                    grid_indexes.append([j,i])
                    depths.append(beddepth[i,j])
    return physical_cords, grid_indexes, depths

def shelf_baths(shelf,polygons):
    physical, grid,depths = get_grounding_line_points(shelf,polygons)
    s = np.argsort(depths)
    s=s[::-1]
    physical,grid,depths = np.asarray(physical)[s],np.asarray(grid)[s],np.asarray(depths)[s]
    distance_mask = shelf_distance_mask(shelf,polygons)
    baths = np.zeros(len(physical),dtype=float)
    baths[:] =np.nan
    bathtub_depths = []
    bathtubs = []
    for i in tqdm(range(len(baths))):
        #print(np.sum(np.isnan(baths)))
        if np.isnan(baths[i]):
            foundGLB, boundingbath, region_mask = GLB_search(shelf,grid[i],polygons,distance_mask)
            if foundGLB:
                bathtubs.append(region_mask)
                bathtub_depths.append(boundingbath)
                baths[i]=boundingbath
                for l_i in range(len(grid)):
                    l = grid[l_i]
                    if region_mask[l[1],l[0]]:
                        baths[l_i]=boundingbath
            else:
                baths[i]=1
                
    return physical,grid,baths,bathtubs, bathtub_depths

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

def shelf_distance_mask(shelf,polygons):
    mask = np.full_like(shelf.icemask_grounded_and_shelves.values,1,dtype=bool)
    cn, cp, distance = closest_shelf([shelf.x[21],shelf.y[569]],polygons)
    for x in range(len(shelf.x))[::10]:
        for y in range(len(shelf.y))[::10]:
            p = [shelf.x[x],shelf.y[y]]
            if cp.exterior.distance(Point(p))>10**5:
                mask[y,x]=0
            else:
                mask[y,x]=1
    return mask


def shelf_distance_test(shelf,polygons):
    plt.figure(figsize=(16, 14))
    ax = plt.subplot(111)
    pc = shelf.icemask_grounded_and_shelves.plot.pcolormesh(
        ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30)
    )
    xs= []
    ys= []
    cn, cp, distance = closest_shelf([shelf.x[21],shelf.y[569]],polygons)
    for x in shelf.x[::10]:
        for y in shelf.y[::10]:
            p = [x,y]
            if cp.exterior.distance(Point(p))>10**5:
                xs.append(x)
                ys.append(y)
    plt.scatter(xs,ys,c="red")
    plt.show()



#shelf_distance_test(shelf,polygons)
#mc,mx,my,lines = highlight_margin(shelf,polygons)

physical,grid,baths, bathtubs,bathtub_depths = shelf_baths(shelf,polygons)
print(bathtub_depths)
overallmap = np.full_like(bathtubs[0],0,dtype=int)
for i in range(len(bathtubs)):
    overallmap[bathtubs[i]]=bathtub_depths[i]
plt.imshow(overallmap)
plt.colorbar()
plt.show()
# physical = np.asarray(physical)
# mx,my = physical.T[0],physical.T[1]
# plt.figure(figsize=(16, 14))
# ax = plt.subplot(111)
# pc = shelf.icemask_grounded_and_shelves.plot.pcolormesh(
#   ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30)
# )
# plt.scatter(mx,my,c=bs)
# for l in lines:
#     l = np.asarray(l).T
#     plt.plot(l[0],l[1],c="orange",alpha=0.5)

plt.show()
