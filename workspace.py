from cdw import *
from bathtub import *
import winds
import pdb
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

# with open("data/GLBsearchresults.pickle","rb") as f:
#     physical,grid,baths,bathtubs,bathtub_depths = pickle.load(f)


# # # # #bedmach = convert_bedmachine("data/BedMachine.nc")
# # bedvalues = bedmap.bed.values

# # bedvalues[bedvalues<-2000] = -2000
# # bedvalues[bedmap.icemask_grounded_and_shelves==0]=0

# # # plt.imshow(bedvalues)
# # # print(np.max(bedvalues))
# # # plt.show()
# r_p,r_z, r_d, r_parents, r_m = build_contour_tree(bedvalues,step=5,start=-2000,stop=0)                
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


# bedvalues = bedmap.bed.values

#jplt.imshow(np.logical_and(bedvalues<-1300,np.isnan(bedmap.icemask_grounded_and_shelves.values)))
#plt.show()
# #
# physical, grid, depths, is_points = get_line_points(bedmap,polygons,"is")

# with open("data/ispoints.pickle","wb") as f:
#    pickle.dump([physical,grid,depths,is_points],f)
with open("data/ispoints.pickle","rb") as f:
   [p,g,d,is_points] = pickle.load(f)
shelf_profiles, shelf_profile_heat_functions = generate_shelf_profiles("data/woawithbed.nc",is_points,polygons)

with open("data/shelfprof.pickle","wb") as f:
   pickle.dump([shelf_profiles,shelf_profile_heat_functions],f)

with open("data/shelfprof.pickle","rb") as f:
   [shelf_profiles, shelf_profile_heat_functions] = pickle.load(f)


shelf_heat_content, shelf_heat_content_byshelf = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,True,intsize=20)

with open("data/shc_GLIB.pickle","wb") as f:
    pickle.dump(shelf_heat_content_byshelf,f)


shelf_heat_content, shelf_heat_content_byshelf = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,False,intsize=20)

with open("data/shc_noGLIB.pickle","wb") as f:
    pickle.dump(shelf_heat_content_byshelf,f)

with open("data/shc_noGLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_noGLIB = pickle.load(f)

with open("data/shc_GLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_GLIB = pickle.load(f)


rignot_shelf_massloss,rignot_shelf_areas,sigma = extract_rignot_massloss("data/rignot2019.xlsx")

# # # polyna_in_shelf_dist =  {}
# # # for p in tqdm(polygons.keys()):
# # #     shelf = polygons[p]
# # #     centroid = list(shelf.centroid.coords)[0]
# # #     mask = np.full_like(polyna,np.nan,dtype=bool)
# # #     dist = np.sqrt((bedmap.coords["x"]- centroid[0])**2 + (bedmap.coords["y"] - centroid[1])**2)
# # #np.sqrt((bedmap.coords["x"]- centroid[0])**2 + (bedmap.coords["y"] - centroid[1])**2)
# # #     radius=1000*10**3
# # #     mask[dist<radius] = True
# # #     mask[dist>radius] = False
# # #     polyna_in_shelf_dist[p] = np.nansum(polyna[mask])
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

# #gl_length = get_grounding_line_length(physical,polygons)
# # with open("data/gl_length.pickle","wb") as f:
# #     pickle.dump(gl_length,f)
# with open("data/gl_length.pickle","rb") as f:
#     gl_length = pickle.load(f)

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)
icemask = bedmap.icemask_grounded_and_shelves.values
winds_by_shelf = winds.AMPS_wind(polygons,"data/AMPS_winds.mat",icemask)
fig,(ax1,ax2) = plt.subplots(1,2)
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
        m,s = np.median(d),2*np.nanstd(d)
        d = np.asarray(d)
        d = d[np.logical_and(d>m-s,d<m+s)]
        if len(d)>0:
            shcsum.append(np.nansum(d))
            shcmean.append(np.nanmean(d))
            lengths.append(len(d))
            areas.append((rignot_shelf_areas[k]))
            smb.append(rignot_shelf_massloss[k])
            x = np.nanmedian(polyna_in_shelf[k])
            #if len(polyna_in_shelf[k])>0:
            prate.append(polyna_in_shelf[k])
            #sigmas.append(max(sigma[k]/(abs(rignot_shelf_massloss[k])+0.01),1))
            sigmas.append(sigma[k])
prate,smb,areas= np.asarray(prate),  np.asarray(smb), np.asarray(areas)
shcsum,shcmean,lengths = np.asarray(shcsum), np.asarray(shcmean), np.asarray(lengths)
X = np.asarray([shcsum,shcmean,lengths,areas,(1/lengths),(1/areas),np.sqrt(areas),np.sqrt(lengths)]).T
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# y = smb
# rf = RandomForestRegressor(max_depth=2, random_state=1)
# rf.fit(X,y)
# print(rf.feature_importances_)

ax1.set_xlabel("Mean heat content +- 50 dbar of max(groundingline,GLIB)")
#ax1.scatter(shc[prate<=0],smb[prate<=0],marker="x")
#ax1.scatter(shc[prate>0],smb[prate>0],c=np.log10(prate[prate>0]),cmap="jet_r")
ax1.scatter(shcmean,smb,cmap="jet_r")
#ax1.scatter(wrate,prate,c=prate,cmap="jet_r")
#ax1.scatter(prate,shc,c=smb,cmap="magma",vmin=-3,vmax=0)
#ax1.scatter(shc,smb)
ax1.set_ylabel("Mean heat content +- 50 dbar of groundingline")
ax1.set_xlabel("Winter Zonal Surface wind average")

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
        m,s = np.median(d),2*np.nanstd(d)
        d = np.asarray(d)
        d = d[np.logical_and(d>m-s,d<m+s)]
        if len(d)>0:
            shcsum.append(np.nansum(d))
            shcmean.append(np.nanmean(d))
            lengths.append(len(d))
            areas.append((rignot_shelf_areas[k]))
            smb.append(rignot_shelf_massloss[k])
            x = np.nanmedian(polyna_in_shelf[k])
            #if len(polyna_in_shelf[k])>0:
            prate.append(polyna_in_shelf[k])
            #sigmas.append(max(sigma[k]/(abs(rignot_shelf_massloss[k])+0.01),1))
            sigmas.append(sigma[k])
        # if len(polyna_in_shelf[k])>0:
        #     prate.append(np.nanmedian(polyna_in_shelf[k]))
        # else:
        #     prate.append(0)
        #ax2.text(prate[-1],smb[-1],k)
prate,smb,areas= np.asarray(prate), np.asarray(smb), np.asarray(areas)
shcsum,shcmean,lengths = np.asarray(shcsum), np.asarray(shcmean), np.asarray(lengths)
# X = np.asarray([shcsum,areas]).T
# scaler = StandardScaler()
# scaler.fit(X)
# X = scaler.transform(X)
# y = smb
# rf = RandomForestRegressor(max_depth=5, random_state=1)
# lr = LinearRegression().fit(X,y)
# print(lr.coef_)

# rf.fit(X,y)
# print(rf.feature_importances_)
#ax2.set_ylabel("Mean heat content +- 50 dbar of max(groundingline,GLIB)")
ax2.set_xlabel("Winter Zonal Surface wind average")
#ax2.scatter(shc[prate<=0],smb[prate<=0],marker="x")
c = ax2.scatter(shcmean,smb,cmap="jet_r")

#ax2.errorbar(shc[~np.isnan(shc)],smb[~np.isnan(shc)],yerr=sigmas[~np.isnan(shc)],fmt="o")
#c = ax2.scatter(prate,shc,c=smb,cmap="magma")
# c= ax2.scatter(shc[prate>0],smb[prate>0],c=np.log10(prate[prate>0]),cmap="jet_r")
#ax2.scatter(shc,smb)
plt.colorbar(c)
plt.show()

pc = bedmap.icemask_grounded_and_shelves.plot.pcolormesh(
  ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30)
)

plt.show()
physical = np.asarray(physical).T
plt.scatter(physical[0],physical[1],c=shelf_heat_content,vmin=4.4*10**7,vmax=4.5*10**7,cmap="jet")
plt.colorbar()

plt.show()


##MAP VIEW GLB COMPARISON
print(len(physical),len(baths))
shcno, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,copy(baths),bedvalues,grid,physical,False)
shcyes, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,True)
x,y = physical.T[0],physical.T[1]

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.scatter(x,y,c = shcno,cmap="jet")
c = ax2.scatter(x,y,c = shcyes,cmap="jet")
plt.colorbar(c)

plt.show()
