import shapefile
import sys
import copy
import numpy as np
import pickle
import pyproj 
from tqdm import tqdm
from shapely.geometry import Polygon, Point
import shapely
from scipy.ndimage import binary_dilation as bd
import xarray as xr
import rioxarray as riox
import rasterio

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def shelf_areas():
    myshp = open("regions/IceBoundaries_Antarctica_v02.shp", "rb")
    mydbf = open("regions/IceBoundaries_Antarctica_v02.dbf", "rb")
    myshx = open("regions/IceBoundaries_Antarctica_v02.shx", "rb")
    r = shapefile.Reader(shp=myshp, dbf=mydbf, shx=myshx)
    s = r.shapes()
    records = r.shapeRecords()
    print(records[0])
    polygons =  {}
    for i in range(len(s)):
        l = s[i]
        name = records[i].record[0]
        kind = records[i].record[3]
        if l.shapeTypeName == 'POLYGON' and kind== "FL":
            xs, ys = zip(*l.points)
            area = PolyArea(xs,ys)
            polygons[name] = area
    return polygons
 

def save_polygons():
    myshp = open("regions/IceBoundaries_Antarctica_v02.shp", "rb")
    mydbf = open("regions/IceBoundaries_Antarctica_v02.dbf", "rb")
    myshx = open("regions/IceBoundaries_Antarctica_v02.shx", "rb")
    r = shapefile.Reader(shp=myshp, dbf=mydbf, shx=myshx)
    s = r.shapes()
    records = r.shapeRecords()
    print(records)
    polygons =  {}
    for i in range(len(s)):
        l = s[i]
        name = records[i].record[0]
        kind = records[i].record[3]
        if l.shapeTypeName == 'POLYGON' and kind== "FL":
            xs, ys = zip(*l.points)
            polygons[name] = [Polygon(l.points),l.parts]
            plt.annotate(name,(np.mean(xs),np.mean(ys)))
    plt.savefig("search.png")
    with open("data/shelfpolygons.pickle","wb") as f:
        pickle.dump(polygons,f)

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def shelf_mass_loss(dsfname,polygons,firstrun=False):
    if firstrun:
        melts = xr.open_dataset(dsfname)
        melts["phony_dim_1"] = melts.x.T[0]
        melts["phony_dim_0"] = melts.y.T[0]
        melts = melts.drop_vars(["x","y"])
        melts = melts.rename({"phony_dim_1":"x","phony_dim_0":"y"})
        melts.rio.to_raster("data/amundsilli.tif")
        print(melts)
    melt_rates = {}
    for k in tqdm(polygons.keys()):
        raster = riox.open_rasterio('data/amundsilli.tif')
        raster = raster.rio.write_crs("epsg:3031")
        gons = []
        parts = polygons[k][1]
        polygon = polygons[k][0]
        if len(parts)>1:
            parts.append(-1)
            for l in range(0,len(parts)-1):
                poly_path=shapely.geometry.Polygon(np.asarray(polygon.exterior.coords.xy)[:,parts[l]:parts[l+1]].T)
                gons.append(poly_path)
        else:
            gons = [polygon]
        clipped = raster.rio.clip(gons,from_disk=True,all_touched=True)
        print(k," clipped")

        if k == "Shackleton":
            plt.imshow(clipped[0])
            plt.colorbar()
            plt.show()
        if ~np.isnan(clipped).all():
            melt = np.asarray(clipped[0])
            melt_rates[k] = np.nanmean(melt[melt>0])
            del melt
        del clipped

    return melt_rates


    
def shelf_areas():
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
            polygons[name] = PolyArea(xs,ys)
    return polygons

def closest_shelf(coord,polygons):
    min_dist = 1000
    closestname = None
    closestpolygon = None
    for i, (k, v) in enumerate(polygons.items()):
        dist = v[0].distance(Point(coord))
        if dist < min_dist:
            min_dist = dist
            closestname = k
            closestpolygon = v[0]
    return closestname, closestpolygon, min_dist


def get_line_points(shelf,polygons,debug=False,mode="grounding"):
    icemask = np.asarray(shelf.icemask_grounded_and_shelves.values)
    beddepth = np.asarray(shelf.bed.values)
    physical_cords = []
    grid_indexes = []
    depths=[]
    shelves = {}
    for k in polygons.keys():
        shelves[k] = []
    print("Grabbing grounding line points")
    if mode == "grounding":
        iceexpand = bd(icemask==0)
        glline = np.logical_and(iceexpand==1,icemask!=0)
    if mode == "edge":
        iceexpand = bd(icemask==1)
        glline = np.logical_and(iceexpand==1,np.isnan(icemask))
    shelf_keys = []

    for i in tqdm(range(1,icemask.shape[0]-1)):
        for j in  range(1,icemask.shape[1]-1):
            if glline[i][j]:
                physical_cords.append([shelf.x.values[j],shelf.y.values[i]])
                cn, _, _ = closest_shelf([shelf.x.values[j],shelf.y.values[i]],polygons)
                if cn:
                    shelves[cn].append([shelf.x.values[j],shelf.y.values[i]])
                shelf_keys.append(cn)
                grid_indexes.append([i,j])
                depths.append(beddepth[i,j])
    pc = np.asarray(grid_indexes)
    if debug:
        plt.imshow(icemask)
        plt.scatter(pc.T[1],pc.T[0])
        for cn in shelves.keys():
            xy = np.asarray(shelves[cn]).T
            if len(xy)>0:
                plt.scatter(xy[0],xy[1])
        plt.show()
    return physical_cords, grid_indexes, depths,shelves,shelf_keys

def convert_bedmachine(fname,coarsenfact=1):
    bedmach = xr.open_dataset(fname)
    plt.imshow(bedmach.bed.values,vmin=-600,vmax=-200)

    mask = bedmach.mask.values
    newmask = np.full_like(mask,np.nan,dtype=np.float64)
    newmask[mask==0] = np.nan
    newmask[np.logical_or(mask==1,mask==2,mask==4)] = 0
    newmask[mask==3] = 1

    plt.imshow(newmask)
    plt.show()
    bedmach["icemask_grounded_and_shelves"] = bedmach.mask
    bedmach.icemask_grounded_and_shelves.values=newmask
    bedmach = bedmach.coarsen(x=coarsenfact,y=coarsenfact, boundary="trim").mean()
    bedmach.icemask_grounded_and_shelves.values[np.logical_and(bedmach.icemask_grounded_and_shelves.values<1,bedmach.icemask_grounded_and_shelves.values>0)] = 1
    return bedmach

def shelf_sort(shelf_keys,quant):
    shelf_dict = {}
    quant = np.asarray(quant)
    for l in range(len(shelf_keys)):
        k = shelf_keys[l]
        if k:
            if k not in shelf_dict:
                shelf_dict[k]=[]
            if len(np.shape(quant))>1:
                shelf_dict[k].append(quant[:,l])
            else:
                shelf_dict[k].append(quant[l])
    return shelf_dict

 
def shelf_numbering(polygons,bed):
    shelf_count=1
    shelf_number_labels = {}

    mach = bed.icemask_grounded_and_shelves.copy(deep=True)
    shelf_numbers = bed.icemask_grounded_and_shelves.copy(deep=True).values
    mach = mach.rio.write_crs("epsg:3031")

    projection = pyproj.Proj("epsg:3031")
    del mach.attrs['grid_mapping']
    mach.rio.to_raster("data/shelfnumberraster.tif")

    for k in tqdm(list(polygons.keys())[:10]):
        gons = []
        parts = polygons[k][1]
        polygon = polygons[k][0]
        if len(parts)>1:
            parts.append(-1)
            for l in range(0,len(parts)-1):
                poly_path=shapely.geometry.Polygon(np.asarray(polygon.exterior.coords.xy)[:,parts[l]:parts[l+1]].T).buffer(10**4)
                gons.append(poly_path)
        else:
            gons = [polygon]

        raster = riox.open_rasterio('data/shelfnumberraster.tif')
        x = raster.rio.clip(gons,drop=False)
        shelf_numbers[np.where(x[0]==1)]=shelf_count
        shelf_number_labels[k]=shelf_count
        shelf_count+=1

    X,Y = np.meshgrid(bed.x.values,bed.y.values)
    lon,lat = projection(X,Y,inverse=True)
    print(lon,lat)

    return shelf_number_labels, shelf_numbers

