from cdw import *
from bathtub import *
#from contourtree import ocean_regions
import winds
import pdb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from matplotlib.path import Path
import shapely
from matplotlib.colors import ListedColormap
with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)
with open("data/GLIBnew.pickle","rb") as f:
    GLIB = pickle.load(f)
bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
shelfmaskkey={}
x, y = np.meshgrid(bedmap.x,bedmap.y)
fullmask = np.full_like(bedmap.bed.values,fill_value=0,dtype=float)
count=1
# for k in tqdm(polygons.keys()):
#     shelfmaskkey[k]=count
#     polygon,parts = polygons[k]
#     minx,miny,maxx,maxy = polygon.bounds
#     firstmask = np.logical_and(np.logical_and(np.logical_and(x>minx,y>miny),x<maxx),y <maxy)
#     coors=np.hstack((x[firstmask].reshape(-1, 1), y[firstmask].reshape(-1,1))) # coors.shape is (4000000,2)
#     parts.append(-1)
#     if len(parts)>1:
#         centroid=np.mean(polygon.exterior.coords.xy,axis=1)
#         plt.scatter(centroid[0],centroid[1])
#         for l in range(0,len(parts)-1):
#             poly_path=shapely.geometry.Polygon(np.asarray(polygon.exterior.coords.xy)[:,parts[l]:parts[l+1]].T)
#             poly_path = poly_path.buffer(5000)
#             poly_path = Path(np.asarray(poly_path.exterior.coords.xy).T)
#             mask = poly_path.contains_points(coors)
#             mask = mask*count
#             fullmask[firstmask]=np.logical_or(mask,fullmask[firstmask])*count
#     count=count+1
# with open("data/groundinglinepoints.pickle","rb") as f:
#     physical,grid,depths,shelves = pickle.load(f)
# physical = np.asarray(physical).T
# fullmask[fullmask==0] = np.nan
# plt.imshow(bedmap.icemask_grounded_and_shelves.values)
# plt.imshow(fullmask)
# #plt.scatter(grid[1],grid[0],c="red")
# plt.show()

with open("data/shelfmask.pickle","wb") as f:
    pickle.dump([fullmask,shelfmaskkey],f)
with open("data/shelfmask.pickle","rb") as f:
    fullmask,shelfmaskkey = pickle.load(f)
randomcmap = ListedColormap(np.random.rand ( 256,3))
# deltaglib = GLIB-bedmap.bed.values
# shelfvolume={}
# fullmask[bedmap.icemask_grounded_and_shelves==0]=0
# print("assembling deltaglib")
# areas={}
# for k in tqdm(polygons.keys()):
#     if np.nansum(fullmask==shelfmaskkey[k])>0:
#         mask = np.logical_and(fullmask==shelfmaskkey[k],bedmap.icemask_grounded_and_shelves!=0)
#         areas[k]=np.nansum(mask)
#         shelfvolume[k]=np.nansum(deltaglib[mask])
# rignot_shelf_massloss,rignot_shelf_areas,sigma = extract_rignot_massloss("data/rignot2019.xlsx")
# print("plotting")
# for k in tqdm(rignot_shelf_massloss.keys()):
#     if k in shelfvolume.keys():
#         plt.scatter(rignot_shelf_massloss[k]/areas[k],shelfvolume[k]/areas[k])
#         plt.text(rignot_shelf_massloss[k]/areas[k],shelfvolume[k]/areas[k],k)
#     else:
#         print(k)
# plt.xlabel("Surface Mass Balance / Ice Shelf Area")
# plt.ylabel("Total Area Under Every GLIB on Shelf / Ice Shelf Area")
# plt.xlim(-1,0.1)
#plt.show()
physical, grid, depths, shelves, shelf_keys = get_line_points(bedmap,polygons,"gl")
# # grid = np.asarray(grid).T
# # #plt.imshow(bedmap.icemask_grounded_and_shelves.values)

with open("data/groundinglinepoints.pickle","wb") as f:
    pickle.dump([physical,grid,depths,shelves,shelf_keys],f)
with open("data/groundinglinepoints.pickle","rb") as f:
    physical,grid,depths,shelves,shelf_keys = pickle.load(f)
grid = np.asarray(grid)
print(grid.T[0])
grid = np.asarray([grid.T[1],grid.T[0]]).T
#plt.imshow(fullmask,cmap=randomcmap,interpolation=None)
#plt.scatter(grid[0],grid[1],c="r")
# for cn in shelves.keys():
#     xy = np.asarray(shelves[cn]).T
#     plt.scatter(xy[0],xy[1])
# plt.show()
# print(shelves)


bedvalues = bedmap.bed.values

baths = []
for l in range(len(grid)):
    baths.append(GLIB[grid[l][1]][grid[l][0]])

#new_closest_WOA(physical,grid,baths,bedmap)

with open("data/woawithbed.pickle","rb") as f:
    sal,temp = pickle.load(f)
with open("data/cdw_closest.pickle","rb") as f:
    closest_points = pickle.load(f)


glibheats =  tempFromClosestPoint(bedmap,grid,physical,baths,closest_points,sal,temp)
bedvalues = bedmap.bed.values

baths = []
for l in range(len(grid)):
    baths.append(bedvalues[grid[l][1]][grid[l][0]])
noglibheats =  tempFromClosestPoint(bedmap,grid,physical,baths,closest_points,sal,temp)

with open("data/newheats.pickle","wb") as f:
    pickle.dump([noglibheats,glibheats],f)
with open("data/newheats.pickle","rb") as f:
    noglibheats,glibheats = pickle.load(f)

fig, (ax1,ax2) = plt.subplots(1,2)
physical = np.asarray(physical).T
print(physical.shape,len(noglibheats))
ax1.scatter(physical[0],physical[1],c=noglibheats,cmap="jet",vmin=0,vmax=200)
c= ax2.scatter(physical[0],physical[1],c=glibheats,cmap="jet",vmin=0,vmax=200)
plt.colorbar(c)
plt.show()
plt.scatter(physical[0],physical[1],c=np.asarray(glibheats)-np.asarray(noglibheats),vmin=-50,vmax=0,cmap="jet")
plt.colorbar()
plt.show()
# # # # #bedmach = convert_bedmachine("data/BedMachine.nc")
# # bedvalues = bedmap.bed.values

# # bedvalues[bedvalues<-2000] = -2000
# # bedvalues[bedmap.icemask_grounded_and_shelves==0]=0

# # # plt.imshow(bedvalues)
# # # print(np.max(bedvalues))
# # # plt.show()
#r_p,r_z, r_d, r_parents, r_m = build_contour_tree(bedvalues,step=5,start=-2000,stop=0)                
# # #plt.imshow(r_m)
# # #plt.show()


# # with open("data/contourtree.pickle","wb") as f:
# #    pickle.dump([r_p,r_z,r_d,r_parents,r_m],f)

# with open("data/contourtree.pickle","rb") as f:
#    r_p,r_z,r_d,r_parents,r_m = pickle.load(f)

# # bedvalues = bedmap.bed.values
# # icemask = bedmap.icemask_grounded_and_shelves.values
# # baths = np.full_like(baths,np.nan)
# # o_r = ocean_regions(r_z,r_m,icemask)
# # found_glibs = {}
# # r_ids = np.full_like(baths,np.nan)
# # for l in tqdm(range(len(grid))):
# #     coord = grid[l]
# #     if bedvalues[coord[1],coord[0]]<-1:
# #         r_id = int(r_m[coord[1],coord[0]])
# #         r_ids[l] = r_id
# #         depth = np.nan
# #         if r_id not in o_r and ~np.in1d(r_d[r_id],o_r).any():
# #             if not (r_id in found_glibs.keys()):
# #                 eyed,depth = find_GLIB_of_region(r_id,r_d,r_parents,r_z,o_r)
# #                 found_glibs[r_id] = (eyed,depth)
# #             else:
# #                 eyed,depth = found_glibs[r_id]
# #         baths[l]=depth

# # baths[np.isnan(baths)]=1
# # with open("data/contourtreeglibs.pickle","wb") as f:
# #     pickle.dump([baths,found_glibs],f)

# with open("data/contourtreeglibs.pickle","rb") as f:
#    baths,found_glibs = pickle.load(f)


# # print("done with baths")
# # print("points: ", r_p)
# # print("r_z: ", r_z)
# # print("r_d: ", r_d)
# # print("r_parents: ", r_parents)


#jplt.imshow(np.logical_and(bedvalues<-1300,np.isnan(bedmap.icemask_grounded_and_shelves.values)))
#plt.show()
# #
physical, grid, depths, is_points,shelf_keys = get_line_points(bedmap,polygons,"is")
with open("data/ispoints.pickle","wb") as f:
   pickle.dump([physical,grid,depths,is_points,shelf_keys],f)
with open("data/ispoints.pickle","rb") as f:
   [p,g,d,is_points,shelf_keys] = pickle.load(f)

shelf_profiles, shelf_profile_heat_functions = generate_shelf_profiles("data/woawithbed.nc",is_points,polygons)

with open("data/shelfprof.pickle","wb") as f:
   pickle.dump([shelf_profiles,shelf_profile_heat_functions],f)

with open("data/shelfprof.pickle","rb") as f:
   [shelf_profiles, shelf_profile_heat_functions] = pickle.load(f)

# interface_depths = cdw_interface_depth(shelf_profiles)
# froudes = []
# froudes2 = []
# for l in range(len(baths)):
#     cn,_,_ = closest_shelf(physical[l],polygons)
#     if cn in interface_depths.keys() and baths[l]<0:
#         i_d,gprime,nsquared = interface_depths[cn]
#         if abs(i_d) < abs(baths[l]):
#             fr1 = 0.075/np.sqrt(i_d*gprime)
#             fr2 = 0.075/(np.sqrt(gprime*abs(abs(i_d)-abs(baths[l]))))
#             G = np.sqrt(fr1**2+fr2**2)
#             froudes.append(abs(np.sqrt(nsquared))*abs(baths[l]))
#             froudes2.append(G)
#         else:
#             froudes.append(np.nan)
#             froudes2.append(np.nan)
#     else:
#         froudes.append(np.nan)
#         froudes2.append(np.nan)

# froude_by_shelf = {}
# froude2_by_shelf = {}
# for l in range(len(froudes)):
#     cn,_,_ = closest_shelf(physical[l],polygons)
#     if ~np.isnan(froudes[l]) and "Wordie" not in cn:
#         if cn not in froude_by_shelf:
#             froude_by_shelf[cn] = []
#             froude2_by_shelf[cn] = []
#         froude_by_shelf[cn].append(froudes[l])
#         froude2_by_shelf[cn].append(froudes2[l])

# for k in froude_by_shelf.keys():
#     froude_by_shelf[k] = np.nanmedian(froude_by_shelf[k])
#     froude2_by_shelf[k] = np.nanmean(froude2_by_shelf[k])

# vals = []
# names = []
# for k in froude_by_shelf.keys():    
#     names.append(k)
#     vals.append(froude_by_shelf[k])
#     #plt.scatter(froude_by_shelf[k],froude_by_shelf[k])
#     #plt.text(froude_by_shelf[k],froude_by_shelf[k],k)
# s = np.argsort(vals)[::-1]
# names,vals = np.asarray(names)[s],np.asarray(vals)[s]
# fig, (ax2,ax1) = plt.subplots(1,2)
# N1 = int(len(vals)/2)
# N2 = len(vals)-N1
# ax1.barh(range(N1),vals[:N1])
# ax1.set_yticks(range(N1),labels=names[:N1])
# ax1.set_xlim(0,2)
# ax2.barh(range(N2),vals[N1:])
# ax2.set_yticks(range(N2),labels=names[N1:])
# ax2.set_xlim(0,2)
# plt.xlabel("$U_{crit}$")
# plt.show()

# x,y = physical.T[0],physical.T[1]
# plt.scatter(x,y,c=froudes)
# plt.colorbar()
# plt.show()

shelf_heat_content, shelf_heat_content_byshelf = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,True,intsize=100)

with open("data/shc_GLIB.pickle","wb") as f:
    pickle.dump(shelf_heat_content_byshelf,f)

shelf_heat_content, shelf_heat_content_byshelf = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,False,intsize=100)

with open("data/shc_noGLIB.pickle","wb") as f:
    pickle.dump(shelf_heat_content_byshelf,f)

with open("data/shc_noGLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_noGLIB = pickle.load(f)

with open("data/shc_GLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_GLIB = pickle.load(f)


# print(len(physical),len(baths))
# shcno, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,copy(baths),bedvalues,grid,physical,False)
# shcyes, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,True)
# x,y = physical.T[0],physical.T[1]

# fig,(ax1,ax2) = plt.subplots(2,1)
# ax1.scatter(x,y,c = shcno,cmap="jet")
# c = ax2.scatter(x,y,c = shcyes,cmap="jet")
# plt.colorbar(c)

# plt.show()

rignot_shelf_massloss,rignot_shelf_areas,sigma = extract_rignot_massloss("data/rignot2019.xlsx")

with open("data/polynainterp.pickle","rb") as f:
   polyna = pickle.load(f)


# bathtub_volume = {}
# points_by_shelf = {}
# polyna_in_shelf = {}
# for p in tqdm(polygons.keys()):
#     bathtub_volume[p] = 0
#     polyna_in_shelf[p] = []
#     points_by_shelf[p] = [0]

# icemask = bedmap.icemask_grounded_and_shelves.values
# o_r = ocean_regions(r_z,r_m,icemask)
# xvals,yvals = np.meshgrid(bedmap.x,bedmap.y)
# print("starting polyna stuff")
# print(np.sum(polyna>0))
# answers = {}
# for i in tqdm(range(bedmap.bed.shape[0])):
#     for j in range(bedmap.bed.shape[1]):
#         if icemask[i,j] !=0 and polyna[i,j] != 0 and ~np.isnan(r_m[i,j]) and bedvalues[i,j] > -1500:
#             cn,_,_ = closest_shelf((bedmap.x[i],bedmap.y[j]),polygons)
#             r_id  = r_m[i,j]
#             if r_id not in answers:
#                 answers[r_id] = ~(np.in1d(r_d[r_id],o_r).any())
#             if answers[r_id]:
#                 polyna_in_shelf[cn].append(polyna[i,j])

# with open("data/polynabathtub.pickle","wb") as f:
#     pickle.dump(polyna_in_shelf,f)
with open("data/polynabathtub.pickle","rb") as f:
    polyna_in_shelf = pickle.load(f)

# # #gl_length = get_grounding_line_length(physical,polygons)
# # # with open("data/gl_length.pickle","wb") as f:
# # #     pickle.dump(gl_length,f)
with open("data/gl_length.pickle","rb") as f:
    gl_length = pickle.load(f)

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)
icemask = bedmap.icemask_grounded_and_shelves.values
winds_by_shelf = winds.AMPS_wind(polygons,"data/AMPS_winds.mat",icemask)
fig,(ax1,ax2) = plt.subplots(2,1)
shcsum= []
shcmean= []
smb = []
c = []
prate = []
sigmas = []
areas=[]
lengths = []
for k in rignot_shelf_massloss.keys():
    if k in shelf_heat_content_byshelf_noGLIB.keys():
        d = shelf_heat_content_byshelf_noGLIB[k]
        m,s = np.nanmean(d),2*np.nanstd(d)
        d = np.asarray(d)
        print(k,len(d))
        #d = d[np.logical_and(d>m-s,d<m+s)]
        if len(d)==0:
            print(shelf_heat_content_byshelf_noGLIB[k])
        print(len(d))
        print("*"*5)
        if len(d)>0:
            shcsum.append(np.nansum(d))
            shcmean.append(np.nanmean(d))
            #lengths.append(len(d))
            lengths.append(gl_length[k])
            areas.append((rignot_shelf_areas[k]))
            smb.append(rignot_shelf_massloss[k])
            x = np.nanmedian(polyna_in_shelf[k])
            #if len(polyna_in_shelf[k])>0:
            prate.append(np.nanmean(polyna_in_shelf[k]))
            #sigmas.append(max(sigma[k]/(abs(rignot_shelf_massloss[k])+0.01),1))
            sigmas.append(sigma[k])
            #ax1.text(shcmean[-1],smb[-1]/lengths[-1],k)
prate,smb,areas= np.asarray(prate),  np.asarray(smb), np.asarray(areas)
shcsum,shcmean,lengths = np.asarray(shcsum), np.asarray(shcmean), np.asarray(lengths)

ax1.scatter(smb/lengths,shcmean,c=areas,cmap="jet_r")
ax1.set_ylabel("Fraction of Shelf Average temperature points 1C above freezing")
ax1.set_xlabel("Cumulative Mass Balance / groudingline length")

shcsum= []
shcmean= []
smb = []
c = []
prate = []
sigmas = []
areas=[]
lengths = []
for k in rignot_shelf_massloss.keys():
    if k in shelf_heat_content_byshelf_noGLIB.keys():
        d = shelf_heat_content_byshelf_GLIB[k]
        m,s = np.nanmean(d),2*np.nanstd(d)
        d = np.asarray(d)
        #d = d[np.logical_and(d>m-s,d<m+s)]
        if len(d)>0:
            shcsum.append(np.nansum(d))
            shcmean.append(np.nanmean(d))
            lengths.append(gl_length[k])
            areas.append((rignot_shelf_areas[k]))
            smb.append(rignot_shelf_massloss[k])
            x = np.nanmedian(polyna_in_shelf[k])
            #if len(polyna_in_shelf[k])>0:
            prate.append(np.sum(polyna_in_shelf[k]))
            #sigmas.append(max(sigma[k]/(abs(rignot_shelf_massloss[k])+0.01),1))
            sigmas.append(sigma[k])
            print(k)
            #ax2.text(np.shcmean[-1],smb[-1]/lengths[-1],k)
        # if len(polyna_in_shelf[k])>0:
        #     prate.append(np.nanmedian(polyna_in_shelf[k]))
        # else:
        #     prate.append(0)
prate,smb,areas= np.asarray(prate), np.asarray(smb), np.asarray(areas)
shcsum,shcmean,lengths = np.asarray(shcsum), np.asarray(shcmean), np.asarray(lengths)

ax2.set_ylabel("Fraction of Shelf Average temperature points 1C above freezing")
ax2.set_xlabel("Cumulative Mass Balance / groudingline length (GLIB VERSION)")
c = ax2.scatter(smb/lengths,shcmean,c=prate,cmap="jet_r")
plt.show()

pc = bedmap.icemask_grounded_and_shelves.plot.pcolormesh(
  ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30)
)

plt.show()
physical = np.asarray(physical).T
plt.scatter(physical[0],physical[1],c=shelf_heat_content,vmin=4.4*10**7,vmax=4.5*10**7,cmap="jet")
plt.colorbar()

plt.show()


# ##MAP VIEW GLB COMPARISON
# print(len(physical),len(baths))
# shcno, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,copy(baths),bedvalues,grid,physical,False)
# shcyes, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,True)
# x,y = physical.T[0],physical.T[1]

# fig,(ax1,ax2) = plt.subplots(2,1)
# ax1.scatter(x,y,c = shcno,cmap="jet")
# c = ax2.scatter(x,y,c = shcyes,cmap="jet")
# plt.colorbar(c)

# plt.show()
