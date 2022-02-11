from cdw import *

with open("data/shelfpolygons.pickle","rb") as f:
    polygons = pickle.load(f)

with open("data/GLBsearchresults.pickle","rb") as f:
    physical,grid,baths,bathtubs,bathtub_depths = pickle.load(f)

bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
# #bedmach = convert_bedmachine("data/BedMachine.nc")
bedvalues = bedmap.bed.values

shelf_profiles, shelf_profile_heat_functions = generate_shelf_profiles("data/woa18_decav81B0_s00_04.nc","data/woa18_decav81B0_t00_04.nc",polygons,bedmap)
#shelf_heat_content, shelf_heat_content_byshelf = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,False)

# with open("data/shc_noGLIB.pickle","wb") as f:
#    pickle.dump(shelf_heat_content_byshelf,f)

with open("data/shc_noGLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_noGLIB = pickle.load(f)

with open("data/shc_GLIB.pickle","rb") as f:
   shelf_heat_content_byshelf_GLIB = pickle.load(f)


rignot_shelf_massloss = extract_rignot_massloss("data/rignot2019.xlsx")


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


# fig,(ax1,ax2) = plt.subplots(1,2)
# shc= []
# smb = []
# c = []
# for k in rignot_shelf_massloss.keys():
#     if k in shelf_heat_content_byshelf_noGLIB.keys():
#         shc.append(np.nanmedian(shelf_heat_content_byshelf_noGLIB[k]))
#         smb.append(rignot_shelf_massloss[k])
#         c.append(len(shelf_heat_content_byshelf_noGLIB[k]))
# ax1.scatter(shc,smb)
# ax1.set_xlabel("Median heat content +- 50 dbar of groundingline")
# ax1.set_ylabel("Mass Loss from Rignot 2019")

# shc= []
# smb = []
# c = []
# for k in rignot_shelf_massloss.keys():
#     if k in shelf_heat_content_byshelf_noGLIB.keys():
#         shc.append(np.nanmedian(shelf_heat_content_byshelf_GLIB[k]))
#         smb.append(rignot_shelf_massloss[k])
#         #ax2.text(shc[-1],smb[-1],k)
# ax2.set_xlabel("Median heat content +- 50 dbar of max(groundingline,GLIB)")
# ax2.scatter(shc,smb)
# plt.show()


# pc = bedmap.icemask_grounded_and_shelves.plot.pcolormesh(
#   ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30)
# )

# plt.show()
# physical = np.asarray(physical).T
# plt.scatter(physical[0],physical[1],c=shelf_heat_content,vmin=4.4*10**7,vmax=4.5*10**7,cmap="jet")
# plt.colorbar()

# plt.show()



shcno, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,copy(baths),bedvalues,grid,physical,False)
shcyes, _ = heat_by_shelf(polygons,shelf_profile_heat_functions,baths,bedvalues,grid,physical,True)
x,y = physical.T[0],physical.T[1]

fig,(ax1,ax2) = plt.subplots(1,2)
ax1.scatter(x,y,c = shcno,vmin=-100,vmax=500,cmap="jet")
c = ax2.scatter(x,y,c = shcyes,vmin=-100,vmax=500,cmap="jet")
plt.colorbar(c)

plt.show()
