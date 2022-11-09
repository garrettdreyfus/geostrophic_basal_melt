import scipy.io
import pyproj
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm



def AMPS_wind(polygons,fname,icemask):
    mat = scipy.io.loadmat(fname)
    projection = pyproj.Proj("epsg:3031")
    lons,lats = np.meshgrid(mat["ERA_lon"],mat["ERA_lat"])
    x,y = projection.transform(lons,lats)
    zonal_winds = mat["zonal_winds_AMPS"]
    zonal_winds = np.nanmean(zonal_winds[:,:,:,:2],axis=3)
    #zonal_winds = zonal_winds[:,:,[10,11,0,1],1]#np.nanmean(zonal_winds,axis=3)
    #zonal_winds = zonal_winds[:,:,[10,11,0,1],1]#np.nanmean(zonal_winds,axis=3)
    zonal_winds = np.nanmean(zonal_winds,axis=2)
    merid_winds = mat["merid_winds_AMPS"]
    merid_winds = merid_winds[:,:,[6,7,8,9],2]#np.nanmean(merid_winds,axis=3)
    merid_winds = np.nanmean(merid_winds,axis=2)
    winds_by_shelf = {}
    for k in tqdm(polygons.keys()):
        centroid = list(polygons[k][0].centroid.coords)[0]
        dist = np.sqrt((x- centroid[0])**2 + (y- centroid[1])**2)
        radius=250*10**3
        z = np.nanmean(zonal_winds[dist<radius])
        m = np.nanmean(merid_winds[dist<radius])
        a = np.arctan2(m,z)
        winds_by_shelf[k] = z#np.degrees(a)#np.nanmean(np.degrees(a))
    return winds_by_shelf
