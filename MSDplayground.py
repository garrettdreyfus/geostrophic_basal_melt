import xarray
import rockhound as rh
import xarray as xr
from scipy.ndimage import label 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import pickle
from bathtub import get_line_points, shelf_sort
import copy
from matplotlib import colors
from scipy.ndimage import binary_dilation as bd
from cdw import extract_rignot_massloss
from GLIB import *
from MSD import *


# Retrieve  the bathymetry and ice data
# shelf = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
# # Get matrix of bathymetry values
# depth = np.asarray(shelf.bed.values)
# # Get matrix of ice values (grounded floating noice)
# ice = np.asarray(shelf.icemask_grounded_and_shelves.values)

# # depth = depth[2400:2900,4600:6200]
# # ice = ice[2400:2900,4600:6200]

# with open("data/shelfpolygons.pickle","rb") as f:
#     polygons = pickle.load(f)

# # physical, glpoints, zoop , shelves,shelf_keys = get_line_points(shelf,polygons)
# # glpoints = np.asarray(glpoints)

# # with open("data/glpoints.pickle","wb") as f:
# #    pickle.dump([physical, glpoints,zoop, shelves,shelf_keys],f)
# with open("data/glpoints.pickle","rb") as f:
#     [physical, glpoints,zoop, shelves,shelf_keys] = pickle.load(f)
# #GLIB = generateBedmapGLIBs(depth,ice,resolution=5)
# # MSDS = []

# # for k in tqdm(range(-800,0,20)):
# #     MSD= calcMSD(depth,ice,k,glpoints,resolution=5,debug=True)
# #     MSDS.append(MSD)
# # MSDS = np.asarray(MSDS)
# # with open("data/MSDS.pickle","wb") as f:
# #    pickle.dump(MSDS,f)
# with open("data/GLIBnew.pickle","rb") as f:
#     GLIB = pickle.load(f)
# with open("data/MSDS.pickle","rb") as f:
#     MSDS = pickle.load(f)
# plt.imshow(GLIB)
# plt.show()
# #overlay = ice
# #depth[overlay==0]=np.nan
# #plt.imshow(depth,vmin=-750,vmax=-200,cmap="jet")
# #plt.colorbar()
# ##plt.imshow(overlay)
# #plt.show()

# depths = np.asarray(range(-800,0,20))
# # for l in range(10000):#MSDS.shape[1]):
#     # plt.plot(MSDS[:,l],depths-GLIB[glpoints[l][0],glpoints[l][1]])
# slopes = []
# glibs = []
# numbernans=[]
# for l in tqdm(range(MSDS.shape[1])):
#     deltas = depths-GLIB[glpoints[l][0],glpoints[l][1]]
#     glibs.append(GLIB[glpoints[l][0],glpoints[l][1]])
#     count=0
#     d1=0
#     for step in range(40):
#         if np.argmin(np.abs(deltas))+step<MSDS.shape[0]:
#             m = MSDS[:,l][np.argmin(np.abs(deltas))+step]
#             if ~np.isnan(m):
#                 count+=1
#                 d1 += m
#     numbernans.append((40-np.argmin(np.abs(deltas)))-count)
#     if count != 0:
#         slopes.append(d1/count)
#     else:
#         slopes.append(np.nan)
# plt.hist(numbernans)
# plt.show()
# slopes_by_shelf = shelf_sort(shelf_keys,slopes)

# glibs_by_shelf = shelf_sort(shelf_keys,glibs)
# rignot_shelf_massloss,rignot_shelf_areas,sigma = extract_rignot_massloss("data/rignot2019.xlsx")
# xs,ys,melts=[],[],[]
# with open("data/shc_GLIB.pickle","rb") as f:
#    shelf_heat_content_byshelf_GLIB = pickle.load(f)

# shelf_names = []
# lengths = []
# #fig = plt.figure(figsize = (10, 7))
# #ax = plt.axes(projection ="3d")
# for k in slopes_by_shelf.keys():
#     if k in rignot_shelf_massloss:
#         x,y = np.nanmean(shelf_heat_content_byshelf_GLIB[k]),np.nanmean(slopes_by_shelf[k])
#         c = rignot_shelf_massloss[k]/len(slopes_by_shelf[k])#/rignot_shelf_areas[k]
#         lengths.append(len(slopes_by_shelf[k]))
#         xs.append(float(x))
#         ys.append(y)
#         melts.append(float(c))
#         shelf_names.append(k)
#         plt.annotate(k,(x,c))
#         #ax.text(x,y,c,k)
# print(len(xs))
# print(len(melts))
# plt.scatter(xs,melts)
# plt.show()
# #plt.scatter(xs,ys)
# # Creating plot
# cax= ax.scatter3D(xs,ys,cs,vmax=1000,cmap="jet")
# ax.set_xlabel("heat content above GLIB")
# ax.set_ylabel("max width")
# plt.show()

# cmap1 = plt.cm.rainbow
# norm = colors.BoundaryNorm(range(0,60,5), cmap1.N)

# plt.scatter(glpoints.T[1],glpoints.T[0],c=slopes,cmap=cmap1, norm=norm)
# plt.colorbar()
# plt.show()

