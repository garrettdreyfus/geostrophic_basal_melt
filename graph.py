from cdw import *
from bathtub import * 
import cmocean
from matplotlib.animation import FFMpegFileWriter
# def bath_map(shelf,bathtubs,bathtub_depths):
#     overallmap = np.full_like(shelf.bed.values,0,dtype=float)

#     for i in range(len(bathtubs))[::-1]:
#         overallmap[bathtubs[i]]=bathtub_depths[i]
#     overallmap[overallmap==0]=np.nan
#     redactedbed = shelf.bed.values
#     redactedbed[shelf.icemask_grounded_and_shelves==0]=np.nan
#     plt.imshow(redactedbed,cmap=cmocean.cm.topo,vmin=-2000,vmax=2000)
#     plt.colorbar()
#     plt.imshow(overallmap-redactedbed,cmap="magma")
#     plt.colorbar()
#     plt.show()


def animation3D(xbounds,ybounds,grid,physical):
    bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
    bedvalues = bedmap.bed.values
    #fig, ax2 = plt.subplots()
    fig = plt.figure(1)
    ax1 = fig.add_subplot(111, projection='3d')
    ax2=fig.add_subplot(222)
    X = bedmap.x.values[xbounds[0]:xbounds[1]]
    Y = bedmap.y.values[ybounds[0]:ybounds[1]]
    ice = np.asarray(bedmap.icemask_grounded_and_shelves.values)
    X, Y = np.meshgrid(Y, X)
    Z = bedvalues[xbounds[0]:xbounds[1],ybounds[0]:ybounds[1]]
    I = np.asarray(bedmap.icemask_grounded_and_shelves.values)[xbounds[0]:xbounds[1],ybounds[0]:ybounds[1]]
    surf = ax1.plot_surface(X, Y, Z, cmap="magma",linewidth=0,vmin=-1600,vmax=100)
    Z2 = np.full_like(Z,np.nan)
    Z2[:] = Z
    ax2.imshow(Z2,cmap="magma",vmin=-1600,vmax=100)
    surf = ax1.plot_surface(X, Y, Z, cmap="magma",linewidth=0,vmin=-1600,vmax=100)
    ice=np.full_like(Z,np.nan)
    ice[I==0]=Z[I==0]
    ax1.plot_surface(X, Y, ice+20, color="lightskyblue",linewidth=0,zorder=10)
    ax2.imshow(ice, cmap="Blues")
    ax1.view_init(elev=40., azim=300)
    fig.set_size_inches(18.5, 10.5)
    moviewriter = FFMpegFileWriter()
    with moviewriter.saving(fig, 'myfile.mp4', dpi=250):
        for i in tqdm(list(range(-2000,-599,50))+[-600,-600,-600,-600,-600]):
            if i!=-2000:
                surf.remove()
                im.remove()
            sweep = np.full_like(Z,np.nan)
            sweep[Z<=i]=i
            if i <-601:
                im = ax2.imshow(sweep,cmap="Greens_r")
                surf = ax1.plot_surface(X, Y, sweep, color="green",linewidth=0,zorder=100,alpha=0.65)
            else:
                im = ax2.imshow(sweep,cmap="Reds_r")
                surf = ax1.plot_surface(X, Y, sweep, color="red",linewidth=0,zorder=0)
            fig.suptitle("z={}".format(i))
            #plt.savefig("pics/z={}.png".format(i))
            moviewriter.grab_frame()
        plt.show()



with open("data/groundinglinepoints.pickle","rb") as f:
    physical,grid,depths,shelves = pickle.load(f)

xbounds=  [1500,3000]
ybounds= [1800, 2800]
animation3D(xbounds,ybounds,grid,physical)
