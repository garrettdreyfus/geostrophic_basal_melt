from cdw import *
from bathtub import *
import winds
import pdb
from sklearn.tree import DecisionTreeRegressor

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

with open("data/GLBsearchresults.pickle","rb") as f:
    physical,grid,baths,bathtubs,bathtub_depths = pickle.load(f)

def get_descendants(region_d,region_p,previous_slice,mask):
    descendants = list(np.unique(previous_slice[mask]))
    for eyed in np.unique(previous_slice[mask]):
        if eyed != 0:
            descendants += region_d[eyed]
    return list(np.unique(descendants))

def build_contour_tree(bedvalues,step=20,start=-2000,stop=0):
    ## previous slice
    previous_slice = np.full_like(bedvalues,0)
    unique_id = 0
    region_points = {}
    region_depths = {}
    region_descendents = {}
    region_parents = {}
    region_map = np.full_like(bedvalues,np.nan)
    for depth in tqdm(range(-2001,0,step)):
        next_slice =  np.full_like(bedvalues,0)
        labels, c = label(bedvalues<depth)
        for label_number in tqdm(range(1,c+1)):
            label_mask = np.asarray(labels==label_number)
            if len(np.unique(previous_slice[label_mask]))>2 or np.nanmin(np.unique(previous_slice[label_mask]))==0 :
                ## this means that we are going to merge to regions
                new_region_id = unique_id-1
                unique_id -=1
                coords = np.where(np.logical_and(label_mask,previous_slice==0))
                region_points[new_region_id] = list(np.ravel_multi_index(coords,bedvalues.shape))
                region_map[coords]=new_region_id
                region_depths[new_region_id] = [depth]
                region_descendents[new_region_id] = get_descendants(region_descendents,region_parents,previous_slice,label_mask)
                for eyed in np.unique(previous_slice[label_mask]):
                    region_parents[eyed] = new_region_id
                next_slice[label_mask] = new_region_id

            else:
                ## This regions just growing in volume or staying the same
                ## the id will be the previous slices id at that location that isn't 0,
                ## our ids are negative so we can grab that with the minimum
                region_id = np.min(previous_slice[label_mask])

                coords = np.where(np.logical_and(label_mask,previous_slice==0))
                region_map[coords]=region_id
                region_points[new_region_id] += list(np.ravel_multi_index(coords,bedvalues.shape))
                #region_points[region_id].append(np.where(label_mask))
                next_slice[label_mask] = region_id
        # fig,(ax1,ax2) = plt.subplots(1,2)
        # print(labels)
        # ax1.imshow(labels)
        # ax2.imshow(next_slice)
        # plt.show()
        previous_slice=next_slice
    return region_points, region_depths, region_descendents,region_parents,region_map

def find_GLIB_of_region(region_id,region_descendents,region_parents,region_depths,ocean_regions):
    start_id = region_id
    previous_id = region_id
    count = 0
    while region_id in region_parents.keys():
        for ocean_id in ocean_regions:
            if ocean_id in region_descendents[region_id] or ocean_id == region_id:
                return previous_id, region_depths[previous_id][0]
        previous_id=region_id
        region_id = region_parents[region_id]
        count+=1
    print("All the way out: ", count)
    return start_id, np.nan

def ocean_regions(region_depths,region_map,icemask):
    ocean = []
    for k in tqdm(region_depths.keys()):
        if region_depths[k][0]<-1880 and ~(((icemask[region_map==k]) ==1).all()):
            ocean.append(k)
    return ocean



# bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])

# # # #bedmach = convert_bedmachine("data/BedMachine.nc")
# bedvalues = bedmap.bed.values

# bedvalues[bedvalues<-2000] = -2000
# bedvalues[bedmap.icemask_grounded_and_shelves==0]=0

# # plt.imshow(bedvalues)
# # print(np.max(bedvalues))
# # plt.show()
# r_p,r_z, r_d, r_parents, r_m = build_contour_tree(bedvalues,step=5,start=-2000,stop=0)                
# #plt.imshow(r_m)
# #plt.show()



# with open("data/contourtree.pickle","wb") as f:
#    pickle.dump([r_p,r_z,r_d,r_parents,r_m],f)

with open("data/contourtree.pickle","rb") as f:
   r_p,r_z,r_d,r_parents,r_m = pickle.load(f)

# bedvalues = bedmap.bed.values
# icemask = bedmap.icemask_grounded_and_shelves.values
# baths = np.full_like(baths,np.nan)
# o_r = ocean_regions(r_z,r_m,icemask)
# found_glibs = {}
# r_ids = np.full_like(baths,np.nan)
# for l in tqdm(range(len(grid))):
#     coord = grid[l]
#     if bedvalues[coord[1],coord[0]]<-1:
#         r_id = int(r_m[coord[1],coord[0]])
#         r_ids[l] = r_id
#         depth = np.nan
#         if r_id not in o_r and ~np.in1d(r_d[r_id],o_r).any():
#             if not (r_id in found_glibs.keys()):
#                 eyed,depth = find_GLIB_of_region(r_id,r_d,r_parents,r_z,o_r)
#                 found_glibs[r_id] = (eyed,depth)
#             else:
#                 eyed,depth = found_glibs[r_id]
#         baths[l]=depth

# baths[np.isnan(baths)]=1
# with open("data/contourtreeglibs.pickle","wb") as f:
#     pickle.dump([baths,found_glibs],f)

with open("data/contourtreeglibs.pickle","rb") as f:
   baths,found_glibs = pickle.load(f)




# print("done with baths")
# print("points: ", r_p)
# print("r_z: ", r_z)
# print("r_d: ", r_d)
# print("r_parents: ", r_parents)


# shelf_profiles, shelf_profile_heat_functions = generate_shelf_profiles("data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc",polygons,bedmap)
# shelf_heat_content, shelf_heat_content_byshelf = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,False,intsize=20)

# with open("data/shc_noGLIB.pickle","wb") as f:
#    pickle.dump(shelf_heat_content_byshelf,f)

with open("data/shc_noGLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_noGLIB = pickle.load(f)

with open("data/shc_GLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_GLIB = pickle.load(f)


rignot_shelf_massloss,rignot_shelf_areas = extract_rignot_massloss("data/rignot2019.xlsx")

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


# bedvalues = bedmap.bed.values
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

# gl_length = get_grounding_line_length(physical,polygons)
# with open("data/gl_length.pickle","wb") as f:
#     pickle.dump(gl_length,f)
with open("data/gl_length.pickle","rb") as f:
    gl_length = pickle.load(f)

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)
icemask = bedmap.icemask_grounded_and_shelves.values
winds_by_shelf = winds.AMPS_wind(polygons,"data/AMPS_winds.mat",icemask)
polyna_in_shelf = winds_by_shelf
fig,(ax1,ax2) = plt.subplots(1,2)
shc= []
smb = []
prate = []
c = []
for k in rignot_shelf_massloss.keys():
    if k in shelf_heat_content_byshelf_noGLIB.keys():
        d = shelf_heat_content_byshelf_noGLIB[k]
        m,s = np.nanmedian(d),2*np.nanstd(d)
        d = np.asarray(d)
        d = d[np.logical_and(d>m-s,d<m+s)]
        shc.append(np.nanmean(d))
        smb.append(rignot_shelf_massloss[k])#/gl_length[k])
        prate.append(polyna_in_shelf[k])
        # if len(polyna_in_shelf[k])>0:
        #     prate.append(np.nanmedian(polyna_in_shelf[k]))
        # else:
        #     prate.append(0)
        #ax1.text(shc[-1],smb[-1],k)

prate,shc,smb= np.asarray(prate), np.asarray(shc), np.asarray(smb)
ax1.set_xlabel("Mean heat content +- 50 dbar of max(groundingline,GLIB)")
#ax1.scatter(shc[prate<=0],smb[prate<=0],marker="x")
#ax1.scatter(shc[prate>0],smb[prate>0],c=np.log10(prate[prate>0]),cmap="jet_r")
ax1.scatter(shc,smb,c=prate,cmap="jet_r")
#ax1.scatter(prate,shc,c=smb,cmap="magma",vmin=-3,vmax=0)
#ax1.scatter(shc,smb)
ax1.set_ylabel("Mean heat content +- 50 dbar of groundingline")
ax1.set_xlabel("Winter Zonal Surface wind average")

shc= []
smb = []
c = []
prate = []
for k in rignot_shelf_massloss.keys():
    if k in shelf_heat_content_byshelf_noGLIB.keys():
        d = shelf_heat_content_byshelf_GLIB[k]
        m,s = np.median(d),2*np.nanstd(d)
        d = np.asarray(d)
        d = d[np.logical_and(d>m-s,d<m+s)]
        shc.append(np.nanmean(d))
        smb.append(rignot_shelf_massloss[k])#/gl_length[k])
        x = np.nanmedian(polyna_in_shelf[k])
        #if len(polyna_in_shelf[k])>0:
        prate.append(polyna_in_shelf[k])
        # if len(polyna_in_shelf[k])>0:
        #     prate.append(np.nanmedian(polyna_in_shelf[k]))
        # else:
        #     prate.append(0)
        #ax2.text(prate[-1],smb[-1],k)
prate,shc,smb= np.asarray(prate), np.asarray(shc), np.asarray(smb)
#ax2.set_ylabel("Mean heat content +- 50 dbar of max(groundingline,GLIB)")
ax2.set_xlabel("Winter Zonal Surface wind average")
#ax2.scatter(shc[prate<=0],smb[prate<=0],marker="x")
c = ax2.scatter(shc,smb,c=prate,cmap="jet_r")
#c = ax2.scatter(prate,shc,c=smb,cmap="magma")
# c= ax2.scatter(shc[prate>0],smb[prate>0],c=np.log10(prate[prate>0]),cmap="jet_r")
#ax2.scatter(shc,smb)
plt.colorbar(c)
plt.show()

# pc = bedmap.icemask_grounded_and_shelves.plot.pcolormesh(
#   ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30)
# )

# plt.show()
# physical = np.asarray(physical).T
# plt.scatter(physical[0],physical[1],c=shelf_heat_content,vmin=4.4*10**7,vmax=4.5*10**7,cmap="jet")
# plt.colorbar()

# plt.show()


##MAP VIEW GLB COMPARISON
shcno, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,copy(baths),bedvalues,grid,physical,False)
shcyes, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,True)
x,y = physical.T[0],physical.T[1]

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.scatter(x,y,c = shcno,vmin=-100,vmax=500,cmap="jet")
c = ax2.scatter(x,y,c = shcyes,vmin=-100,vmax=500,cmap="jet")
plt.colorbar(c)

plt.show()
