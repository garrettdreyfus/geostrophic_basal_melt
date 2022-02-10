from cdw import *
from bathtub import * 

def bath_map(shelf,bathtubs,bathtub_depths):
    overallmap = np.full_like(shelf.bed.values,0,dtype=float)

    for i in range(len(bathtubs))[::-1]:
        overallmap[bathtubs[i]]=bathtub_depths[i]
    overallmap[overallmap==0]=np.nan
    redactedbed = shelf.bed.values
    redactedbed[shelf.icemask_grounded_and_shelves==0]=np.nan
    plt.imshow(redactedbed,cmap=cmocean.cm.topo,vmin=-2000,vmax=2000)
    plt.colorbar()
    plt.imshow(overallmap-redactedbed,cmap="magma")
    plt.colorbar()
    plt.show()
