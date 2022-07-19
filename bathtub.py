import shapefile
import sys
import rockhound as rh
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import itertools
import pickle
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import time
from scipy.ndimage import label 
import xarray as xr


def save_polygons():
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
        output = []
        if l.shapeTypeName == 'POLYGON' and kind== "FL":
            print(l.parts)
            xs, ys = zip(*l.points)
            polygons[name] = [Polygon(l.points),l.parts]
            plt.annotate(name,(np.mean(xs),np.mean(ys)))
    plt.savefig("search.png")
    with open("data/shelfpolygons.pickle","wb") as f:
        pickle.dump(polygons,f)

def closest_shelf(coord,polygons):
    min_dist = np.inf
    closestname = None
    closestpolygon = None
    for i, (k, v) in enumerate(polygons.items()):
        dist = v[0].distance(Point(coord))
        if dist < min_dist:
            min_dist = dist
            closestname = k
            closestpolygon = v[0]
    return closestname, closestpolygon, min_dist


def get_line_points(shelf,polygons,mode):
    margin_coords = []
    mx = []
    my = []
    icemask = np.asarray(shelf.icemask_grounded_and_shelves.values)
    beddepth = np.asarray(shelf.bed.values)
    shelf_names = polygons.keys()
    lines = []
    count=0
    physical_cords = []
    grid_indexes = []
    depths=[]
    shelves = {}
    for k in polygons.keys():
        shelves[k] = []
    print("Grabbing grounding line points")
    for i in tqdm(range(1,icemask.shape[0]-1)):
        for j in  range(1,icemask.shape[1]-1):
            if icemask[i][j] == 1:
                if (icemask[i-1:i+2,j]==0).any() or (icemask[i,j-1:j+2]==0).any():
                    physical_cords.append([shelf.x.values[j],shelf.y.values[i]])
                    cn, _, _ = closest_shelf([shelf.x.values[j],shelf.y.values[i]],polygons)
                    shelves[cn].append([shelf.x.values[j],shelf.y.values[i]])
                    grid_indexes.append([j,i])
                    depths.append(beddepth[i,j])
    pc = np.asarray(physical_cords)
    plt.scatter(pc.T[0],pc.T[1])
    for cn in shelves.keys():
        xy = np.asarray(shelves[cn]).T
        plt.scatter(xy[0],xy[1])
    plt.show()
    return physical_cords, grid_indexes, depths,shelves


def trimDataset(bm,xbounds,ybounds):
    shelf=bm
    shelf = shelf.where(shelf.x<xbounds[1],drop=True)
    shelf = shelf.where(shelf.y<ybounds[1],drop=True)
    shelf = shelf.where(shelf.x>xbounds[0],drop=True)
    shelf = shelf.where(shelf.y>ybounds[0],drop=True)
    return shelf

def shelf_distance_mask(ds,shelf,polygons):
    ice = ds.icemask_grounded_and_shelves.values
    bed = ds.bed.values
    #mask = np.full_like(ice,1,dtype=bool)
    mask = ~np.logical_and(np.isnan(ice),bed<-3000)
    return mask

def convert_bedmachine(fname,coarsenfact=2):
    bedmach = xr.open_dataset(fname)
    mask = bedmach.mask.values
    newmask = np.full_like(mask,np.nan,dtype=np.float64)
    newmask[mask==0] = np.nan
    newmask[np.logical_or(mask==1,mask==2,mask==4)] = 0
    newmask[mask==3] = 1
    bedmach["icemask_grounded_and_shelves"] = bedmach.mask
    bedmach.icemask_grounded_and_shelves.values=newmask
    bedmach = bedmach.coarsen(x=coarsenfact,y=coarsenfact, boundary="trim").mean()
    bedmach.icemask_grounded_and_shelves.values[np.logical_and(bedmach.icemask_grounded_and_shelves.values<1,bedmach.icemask_grounded_and_shelves.values>0)] = 1
    return bedmach
 

