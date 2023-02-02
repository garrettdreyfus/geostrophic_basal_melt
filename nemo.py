import GLIB as GL
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pickle

for ext in ["006","016","018","021"]:
    bath = xr.open_dataset("data/nemo/processed_bathymetry_OPM"+ext+".nc")

    nemo_masks = xr.open_dataset("data/nemo/nemo_5km_isf_masks_and_info_and_distance_new_oneFRIS_OPM"+ext+".nc")

    print(nemo_masks)


    shelfmask = nemo_masks.ISF_mask.values.astype(int)

    Nisf = nemo_masks.Nisf.values
    names = nemo_masks.isf_name.values

    shelfids = np.unique(shelfmask)
    print(shelfids)

    icemask = nemo_masks.ISF_mask.values 
    icemask[icemask==1]=np.nan
    icemask[icemask>1]=1

    bedvalues = bath.bathymetry.values
    #plt.imshow(bedvalues)
    #plt.show()

    GLIB = GL.generateGLIBs(-bedvalues,icemask)
    bedvalue_per_shelf = []
    glib_per_shelf = []
    for l in Nisf:
        glib_per_shelf.append(np.nanmean(GLIB[shelfmask==l]))
        bedvalue_per_shelf.append(-np.nanmean(bedvalues[shelfmask==l]))
    plt.scatter(glib_per_shelf,bedvalue_per_shelf)
    for i in range(len(Nisf)):
        plt.annotate(names[i],(glib_per_shelf[i],bedvalue_per_shelf[i]))

    plt.xlim(-1100,0)
    plt.ylim(-1100,0)
    plt.plot(range(-1100,0),range(-1100,0))
    plt.show()

    with open("data/nemo/nemoHUB"+ext+".pickle","wb") as f:
        pickle.dump([glib_per_shelf,GLIB],f)
#plt.imshow(GLIB+bedvalues)

#plt.show()
