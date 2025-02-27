from pysheds.grid import Grid
import rioxarray as riox
import rasterio
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
generatefdir = True
if generatefdir:
    with open("data/bedmach.pickle","rb") as f:
        bedmach = pickle.load(f)

    coarsened = bedmach.coarsen({"x":4,"y":4},boundary="pad").min()
    bedvalues = coarsened.bed
    bedvalues = bedvalues.rio.write_crs("epsg:3031")
    del bedvalues.attrs['grid_mapping']
    bedvalues.rio.to_raster("data/bedmachcoarse.tif")
    grid = Grid.from_raster('data/bedmachcoarse.tif')
    dem = grid.read_raster('data/bedmachcoarse.tif')


    grid = Grid.from_raster('data/bedmachcoarse.tif')

    dem = grid.read_raster('data/bedmachcoarse.tif')

    #dem[dem<-2000]=-2000
    mask = np.logical_or(coarsened.icemask_grounded_and_shelves==0,coarsened.icemask_grounded_and_shelves==3)

    dem[mask]=0

    print("filling bits")
    pit_filled_dem = grid.fill_pits(dem)
    print("resolving flatS")
    depressions = grid.detect_depressions(pit_filled_dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)
    print("flow dir nwo")
    fdir = grid.flowdir(inflated_dem)
    acc = grid.accumulation(fdir, apply_output_mask=False)

    with open("data/fdir.pickle","wb") as f:
            pickle.dump((grid,fdir,acc),f)



#with open("data/fdir.pickle","rb") as f:
    #fdir = pickle.load(f)


#North: 64
#Northeast: 128
#East: 1
#Southeast: 2
#South: 4
#Southwest: 8
#West: 16
#Northwest: 32
def follow_flow(fdir,i_old,j_old,stopmask,progress):
    i_move = {64:0,128:1,1:1,2:1,4:0,8:-1,16:-1,32:-1,-1:0,0:0,-2:0}
    j_move = {64:1,128:1,1:0,2:-1,4:-1,8:-1,16:0,32:1,-1:0,0:0,-2:0}

    new_i = i_move[fdir[i_old,j_old]]+i_old
    new_j = -j_move[fdir[i_old,j_old]]+j_old
    path = []
    while stopmask[new_i,new_j] == 1 :
        path.append((new_i,new_j))
        i_old = new_i
        j_old = new_j
        new_i = i_move[fdir[i_old,j_old]]+i_old
        new_j = -j_move[fdir[i_old,j_old]]+j_old
        if ~np.isnan(progress[new_i,new_j]):
            return progress[new_i,new_j]
        if (new_i== i_old and new_j==j_old):
            return 3
        if (new_i,new_j) in path:
            return 4
    if len(path)>0 and False:
        path = np.asarray(path).T
        print(path)
        plt.imshow(stopmask)
        plt.scatter(path[0],path[1],c="red")
        plt.show()
    return stopmask[new_i,new_j]
    


#with open("data/bedmach.pickle","rb") as f:
    #bedmach = pickle.load(f)
    #
#with open("data/fdir.pickle","rb") as f:
    #grid, fdir, acc = pickle.load(f)

#follow_flow(fdir,4022,4045,bedmach.bed.values<-2000)
#follow_flow(fdir,6026,9666,bedmach.bed.values<-2000)
#follow_flow(fdir,2785,5906,bedmach.bed.values<-2000)
mask = np.logical_or(coarsened.icemask_grounded_and_shelves==0,coarsened.icemask_grounded_and_shelves==3)
eligible = np.logical_and(coarsened.bed.values>-2000,coarsened.bed.values<0)
eligible = eligible.astype(int)
eligible[mask] = 2
coords = np.where(eligible==1)
output = np.full_like(coarsened.bed.values,np.nan)
for i in tqdm(range(len(coords[0]))):
    #if coords[0][i]>120 and coords[0][i]<250:
        #if coords[1][i]>120 and coords[1][i]<250:
    output[coords[0][i],coords[1][i]] = follow_flow(fdir,coords[0][i],coords[1][i],eligible,output)
with open("data/output.pickle","wb") as f:
    pickle.dump((grid,fdir,acc),f)

plt.imshow(output)
plt.show()
#plt.imshow(fdir)
#plt.show()
#follow_flow(fdir,
