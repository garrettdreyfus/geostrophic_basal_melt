from cdw import *
from bathtub import *

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

with open("data/GLBsearchresults.pickle","rb") as f:
    physical,grid,baths,bathtubs,bathtub_depths = pickle.load(f)
# physical,grid,baths,bathtubs,bathtub_depths = find_and_save_bathtubs("bedmap","data/bulkGLBcalc.pickle")
# for b in bathtubs:
#     plt.imshow(b)
# plt.show()
bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])

# #bedmach = convert_bedmachine("data/BedMachine.nc")
bedvalues = bedmap.bed.values

# shelf_profiles, shelf_profile_heat_functions = generate_shelf_profiles("data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc",polygons,bedmap)
# shelf_heat_content, shelf_heat_content_byshelf = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,True)

# with open("data/shc_GLIB.pickle","wb") as f:
#    pickle.dump(shelf_heat_content_byshelf,f)

with open("data/shc_noGLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_noGLIB = pickle.load(f)

with open("data/shc_GLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_GLIB = pickle.load(f)


rignot_shelf_massloss,rignot_shelf_areas = extract_rignot_massloss("data/rignot2019.xlsx")

with open("data/polynainterp.pickle","rb") as f:
   polyna = pickle.load(f)

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

with open("data/polynabathtub.pickle","rb") as f:
    polyna_in_shelf = pickle.load(f)

# polyna_in_shelf_dist =  {}
# for p in tqdm(polygons.keys()):
#     shelf = polygons[p]
#     centroid = list(shelf.centroid.coords)[0]
#     mask = np.full_like(polyna,np.nan,dtype=bool)
#     dist = t_content_byshelf,f)

with open("data/shc_noGLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_noGLIB = pickle.load(f)

with open("data/shc_GLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_GLIB = pickle.load(f)


rignot_shelf_massloss,rignot_shelf_areas = extract_rignot_massloss("data/rignot2019.xlsx")

with open("data/polynainterp.pickle","rb") as f:
   polyna = pickle.load(f)

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

with open("data/polynabathtub.pickle","rb") as f:
    polyna_in_shelf = pickle.load(f)

# polyna_in_shelf_dist =  {}
# for p in tqdm(polygons.keys()):
#     shelf = polygons[p]
#     centroid = list(shelf.centroid.coords)[0]
#     mask = np.full_like(polyna,np.nan,dtype=bool)
#     dist = np.sqrt((bedmap.coords["x"]- centroid[0])**2 + (bedmap.coords["y"] - centroid[1])**2)
#np.sqrt((bedmap.coords["x"]- centroid[0])**2 + (bedmap.coords["y"] - centroid[1])**2)
#     radius=1000*10**3
#     mask[dist<radius] = True
#     mask[dist>radius] = False
#     polyna_in_shelf_dist[p] = np.nansum(polyna[mask])

bedvalues = bedmap.bed.values
bathtub_volume = {}
points_by_shelf = {}
for p in tqdm(polygons.keys()):
    bathtub_volume[p] = 0
    polyna_in_shelf[p] = 0
    points_by_shelf[p] = [0]
xvals,yvals = np.meshgrid(bedmap.x,bedmap.y)
for i in tqdm(range(len(bathtubs))):
    bd = bathtub_depths[i]
    b = bathtubs[i]
    shelf = polygons[p]
    centroid = tuple([np.nanmean(xvals[b]),np.nanmean(yvals[b])])
    cn,_,_ = closest_shelf(centroid,polygons)
    # if cn == "Thwaites" and len(b[0])>100:
    omap = np.full_like(bedvalues,np.nan)
    omap[b] = polyna[b]
    #     plt.imshow(bedvalues)
    #     plt.imshow(omap)
    #     plt.show()
    for l in grid:
        if omap[l[1],l[0]]:
            points_by_shelf[cn].append(1)
    if np.max(polyna[b])>0:
        if cn in rignot_shelf_areas:
            polyna_in_shelf[cn] += np.nansum(polyna[b])*np.sum((points_by_shelf[cn]))



with open("data/polynabathtub.pickle","wb") as f:
    pickle.dump(polyna_in_shelf,f)
with open("data/polynabathtub.pickle","rb") as f:
    polyna_in_shelf = pickle.load(f)
# rignot_shelf_massloss["Filcher-Ronne"] = rignot_shelf_massloss["Filchner"]+rignot_shelf_massloss["Ronne"]
# shelf_heat_content_byshelf_GLIB["Filcher-Ronne"] = shelf_heat_content_byshelf_GLIB["Filchner"]+shelf_heat_content_byshelf_GLIB["Ronne"]
# shelf_heat_content_byshelf_noGLIB["Filcher-Ronne"] = shelf_heat_content_byshelf_noGLIB["Filchner"]+shelf_heat_content_byshelf_noGLIB["Ronne"]

# rignot_shelf_massloss["Crosson-Dotson"] = rignot_shelf_massloss["Crosson"]+rignot_shelf_massloss["Dotson"]
# shelf_heat_content_byshelf_GLIB["Crosson-Dotson"] = shelf_heat_content_byshelf_GLIB["Crosson"]+shelf_heat_content_byshelf_GLIB["Dotson"]
# shelf_heat_content_byshelf_noGLIB["Crosson-Dotson"] = shelf_heat_content_byshelf_noGLIB["Crosson"]+shelf_heat_content_byshelf_noGLIB["Dotson"]

# del rignot_shelf_massloss["Crosson"]
# del rignot_shelf_massloss["Dotson"]
# del shelf_heat_content_byshelf_noGLIB["Dotson"]
# del shelf_heat_content_byshelf_noGLIB["Crosson"]
# del shelf_heat_content_byshelf_GLIB["Crosson"]
# del shelf_heat_content_byshelf_GLIB["Dotson"]
# del rignot_shelf_massloss["LarsenB"]
# del rignot_shelf_massloss["George_VI"]
# del shelf_heat_content_byshelf_noGLIB["LarsenB"]
# del shelf_heat_content_byshelf_noGLIB["George_VI"]
# del shelf_heat_content_byshelf_GLIB["LarsenB"]
# del shelf_heat_content_byshelf_GLIB["George_VI"]

# del rignot_shelf_massloss["Filchner"]
# del rignot_shelf_massloss["Ronne"]
# del shelf_heat_content_byshelf_noGLIB["Filchner"]
# del shelf_heat_content_byshelf_noGLIB["Ronne"]
# del shelf_heat_content_byshelf_GLIB["Filchner"]
# del shelf_heat_content_byshelf_GLIB["Ronne"]

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
        smb.append(rignot_shelf_massloss[k])
        prate.append(polyna_in_shelf[k]/np.sum(points_by_shelf[k]))

prate,shc,smb= np.asarray(prate), np.asarray(shc), np.asarray(smb)
ax1.set_xlabel("Mean heat content +- 50 dbar of max(groundingline,GLIB)")
ax1.scatter(shc[prate<=0],smb[prate<=0],marker="x")
ax1.scatter(shc[prate>0],smb[prate>0],c=np.log10(prate[prate>0]),cmap="jet_r")
ax1.set_xlabel("Mean heat content +- 50 dbar of groundingline")
ax1.set_ylabel("Mass Loss from Rignot 2019")

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
        smb.append(rignot_shelf_massloss[k])
        x = np.nanmedian(polyna_in_shelf[k])
        #if len(polyna_in_shelf[k])>0:
        prate.append(polyna_in_shelf[k]/np.sum(points_by_shelf[k]))
        #ax2.text(shc[-1],smb[-1],k)
prate,shc,smb= np.asarray(prate), np.asarray(shc), np.asarray(smb)
ax2.set_xlabel("Mean heat content +- 50 dbar of max(groundingline,GLIB)")
ax2.scatter(shc[prate<=0],smb[prate<=0],marker="x")
c= ax2.scatter(shc[prate>0],smb[prate>0],c=np.log10(prate[prate>0]),cmap="jet_r")
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
# shcno, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,copy(baths),bedvalues,grid,physical,False)
# shcyes, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,True)
# x,y = physical.T[0],physical.T[1]

# fig,(ax1,ax2) = plt.subplots(1,2)
# ax1.scatter(x,y,c = shcno,vmin=-100,vmax=500,cmap="jet")
# c = ax2.scatter(x,y,c = shcyes,vmin=-100,vmax=500,cmap="jet")
# plt.colorbar(c)

# plt.show()
