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
        if l.shapeTypeName == 'POLYGON' and kind== "FL":
            xs, ys = zip(*l.points)
            polygons[name] = Polygon(l.points)
            plt.annotate(name,(np.mean(xs),np.mean(ys)))
    plt.savefig("search.png")
    with open("data/shelfpolygons.pickle","wb") as f:
        pickle.dump(polygons,f)

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

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
        below = np.logical_and((depth<=bath),ice!=0)
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

def GLB_search_bulk(shelf,gl_points,polygons,distance_mask):
    depth = np.asarray(shelf.bed.values)
    ice = np.asarray(shelf.icemask_grounded_and_shelves.values)
    #the whole ordering thing confused me so I'm reversing
    baths = np.empty(gl_points.shape[0])
    baths[:] = np.nan
    bath= np.nanmin(depth[np.flip(gl_points,1)])
    bathtubs = [[]] *len(baths)
    #indices = np.indices(depth)
    previous_label = []
    old_count = 0
    while np.sum(np.isnan(baths))>0:
        print(bath,np.sum(np.isnan(baths)),end="\r")
        sys.stdout.flush()
        below = np.logical_and((depth<=bath),ice!=0)
        regions, _ = label(below)
        if bath>0:
            baths[np.isnan(baths)]=1
        for i in range(len(gl_points)):
            if np.isnan(baths[i]):
                coord = gl_points[i]
                if bath>depth[coord[1],coord[0]]:
                    if (regions[coord[1],coord[0]] == regions[~distance_mask]).any():
                        baths[i] = bath
                        if len(previous_label)>0:
                            bathtubs[i] = (previous_label == previous_label[coord[1],coord[0]])
                        else:
                            bathtubs[i] = []
                
        previous_label = regions
        bath+=20
    return baths, bathtubs

def GLB_search_bulk(shelf,gl_points,polygons,distance_mask):
    depth = np.asarray(shelf.bed.values)
    ice = np.asarray(shelf.icemask_grounded_and_shelves.values)
    #the whole ordering thing confused me so I'm reversing
    baths = np.empty(gl_points.shape[0])
    baths[:] = np.nan
    bath= np.nanmin(depth[np.flip(gl_points,1)])
    bathtubs = [[]] *len(baths)
    #indices = np.indices(depth)
    previous_label = []
    old_count = 0
    while np.sum(np.isnan(baths))>0:
        print(bath,np.sum(np.isnan(baths)),end="\r")
        sys.stdout.flush()
        below = np.logical_and((depth<=bath),ice!=0)
        regions, _ = label(below)
        if bath>0:
            baths[np.isnan(baths)]=1
        for i in range(len(gl_points)):
            if np.isnan(baths[i]):
                coord = gl_points[i]
                if bath>depth[coord[1],coord[0]]:
                    if (regions[coord[1],coord[0]] == regions[~distance_mask]).any():
                        baths[i] = bath
                        if len(previous_label)>0:
                            bathtubs[i] = (previous_label == previous_label[coord[1],coord[0]])
                        else:
                            bathtubs[i] = []
                
        previous_label = regions
        bath+=20
    return baths, bathtubs



def get_grounding_line_length(gl_physical,polygons):
    shelf_count = {}
    for k in polygons.keys():
        shelf_count[k] = 0
    for l in tqdm(gl_physical):
        cn, cp, distance = closest_shelf([l[0],l[1]],polygons)
        shelf_count[cn] = shelf_count[cn]+1
    return shelf_count



def get_line_points(shelf,polygons,mode):
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
    shelves = {}
    for k in polygons.keys():
        shelves[k] = []
    print("Grabbing grounding line points")
    for i in tqdm(range(1,icemask.shape[0]-1)):
        for j in  range(1,icemask.shape[1]-1):
            if icemask[i][j] == 1:
                a = icemask[i+1][j]
                b = icemask[i-1][j]
                c = icemask[i][j+1]
                d = icemask[i][j-1]
                if (mode=="gl" and (np.asarray([a,b,c,d])==0).any())) or (mode=="is" and (np.isnan(np.asarray([a,b,c,d])).any())):
                    physical_cords.append([shelf.x[j],shelf.y[i]])
                    cn, _, _ = closest_shelf([shelf.x[j],shelf.y[i]],polygons)
                    shelves[cn].append([shelf.x[j],shelf.y[i]])
                    grid_indexes.append([j,i])
                    depths.append(beddepth[i,j])
    # plt.imshow(beddepth,vmin=-2000,vmax=2000)
    # grid_indexes = np.asarray(grid_indexes).T
    # plt.scatter(grid_indexes[0],grid_indexes[1])
    # plt.show()
    return physical_cords, grid_indexes, depths,shelves


def shelf_baths(shelf,polygons):
    physical, grid,depths,_ = get_line_points(shelf,polygons,"gl")
    s = np.argsort(depths)
    s=s[::-1]
    physical,grid,depths = np.asarray(physical)[s],np.asarray(grid)[s],np.asarray(depths)[s]
    distance_mask = shelf_distance_mask(shelf,"Moscow_University",polygons)
    baths = np.zeros(len(physical),dtype=float)
    baths[:] =np.nan
    bathtub_depths = []
    bathtubs = []
    #for i in tqdm(range(len(baths))):
        #print(np.sum(np.isnan(baths)))
        #cn, cp, distance = closest_shelf([physical[i][0],physical[i][1]],polygons)
        #if np.isnan(baths[i]):# and cn == "Pine_Island":
            #foundGLB, boundingbath, region_mask = GLB_search(shelf,grid[i],polygons,distance_mask)
            # if foundGLB:
            #     bathtubs.append(np.where(region_mask==1))
            #     bathtub_depths.append(boundingbath)
            #     baths[i]=boundingbath
            #     for l_i in range(len(grid)):
            #         l = grid[l_i]
            #         if region_mask[l[1],l[0]]:
            #             baths[l_i]=boundingbath
            #else:
                #baths[i]=1
    baths, bathtubs = GLB_search_bulk(shelf,grid,polygons,distance_mask)
    return physical,grid,baths,bathtubs, baths

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
 

def find_and_save_bathtubs(dataset,outfname):
    if dataset == "bedmach":
        shelf = convert_bedmachine("data/BedMachine.nc")
    elif dataset == "bedmap":
        shelf = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
    physical,grid,baths, bathtubs,bathtub_depths = shelf_baths(shelf,polygons)
    with open(outfname,"wb") as f:
        pickle.dump([physical,grid,baths,bathtubs,bathtub_depths],f)
    return physical,grid,baths,bathtubs,bathtub_depths

##Default
#xbounds=  [ -1.6*(10**6),-0.25*(10**6)]
#ybounds= [-1*(10**6), 1.85*(10**6)]
##Closer in
#xbounds=  [ -1.0*(10**6),-0.75*(10**6)]
#ybounds= [-0.5*(10**6), 2.5*(10**6)]
#xbounds=  [ -1.75*(10**6),-0*(10**6)]
#ybounds= [-1.5*(10**6), 3.5*(10**6)]
#PIG
#jxbounds=[ -2.4*(10**6),-1.4*(10**6)]
#ybounds=[-1.4*(10**6), -0.2*(10**6)]
#ROSS
#xbounds = [ -0.575*(10**6),0.375*(10**6)]
#ybounds = [-2*(10**6), 0]

#shelf_distance_test(shelf,polygons)
#mc,mx,my,lines = highlight_margin(shelf,polygons)

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

