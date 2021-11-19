import shapefile
import rockhound as rh
import matplotlib.pyplot as plt
import cmocean
import numpy as np
from shapely.geometry import Polygon, Point


myshp = open("regions/IceBoundaries_Antarctica_v02.shp", "rb")
mydbf = open("regions/IceBoundaries_Antarctica_v02.dbf", "rb")
myshx = open("regions/IceBoundaries_Antarctica_v02.shx", "rb")
r = shapefile.Reader(shp=myshp, dbf=mydbf, shx=myshx)
s = r.shapes()
#r.shapes()[0].points
records = r.shapeRecords()

bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
plt.close()
plt.figure(figsize=(16, 14))
ax = plt.subplot(111)
pc = bedmap.icemask_grounded_and_shelves.plot.pcolormesh(
   ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30),
)
polygons =  {}
for i in range(len(s)):
    l = s[i]
    name = records[i].record[0]
    kind = records[i].record[3]
    if l.shapeTypeName == 'POLYGON' and kind== "FL":
        print(list(l.bbox))
        xs, ys = zip(*l.points)
        plt.plot(xs,ys)
        print(records[i].record)
        polygons[name] = Polygon(l.points)
        plt.annotate(name,(np.mean(xs),np.mean(ys)))
plt.savefig("search.png")
print(polygons)

def closest_shelf(coord,polygons):
    min_dist = np.inf
    closest = None
    for i, (k, v) in enumerate(polygons.items()):
        dist = v.distance(Point(coord) )
        if dist < min_dist:
            min_dist = dist
            closest = k
    return closest


def highlight_margin(shelf,polygons):
    margin_coords = []
    margin_x = []
    margin_y = []
    icemask = shelf.icemask_grounded_and_shelves.values
    for i in range(1,icemask.shape[0]-1):
        for j in  range(1,icemask.shape[1]-1):
            if icemask[i][j] == 1:
                a = icemask[i+1][j]
                b = icemask[i-1][j]
                c = icemask[i][j+1]
                d = icemask[i][j-1]
                if np.isnan([a,b,c,d]).any():
                    margin_coords.append(tuple([j,i]))
                    margin_x.append(shelf.x[j])
                    margin_y.append(shelf.y[i])
    return margin_coords, margin_x, margin_y

def trimDataset(bm,xbounds,ybounds):
    shelf=bm
    shelf = shelf.where(shelf.x<xbounds[1],drop=True)
    shelf = shelf.where(shelf.y<ybounds[1],drop=True)
    shelf = shelf.where(shelf.x>xbounds[0],drop=True)
    shelf = shelf.where(shelf.y>ybounds[0],drop=True)
    return shelf

bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
xbounds=  [ -1.5*(10**6),-0.75*(10**6)]
ybounds= [-0.5*(10**6), 2.5*(10**6)]
shelf = trimDataset(bedmap,xbounds,ybounds)
highlight_margin(shelf,polygons)
