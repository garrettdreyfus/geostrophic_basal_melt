import rockhound as rh
import matplotlib.pyplot as plt
import cmocean
import numpy as np
import itertools
def highestBoundingBath(shelf,shelfname,seedcoord,searchrange,endcoord):
    depth = shelf.bed.values
    ice = shelf.icemask_grounded_and_shelves.values
    moves = list(itertools.product(*[[-100,0,100],[-100,0,100]]))
    seedcoord = seedcoord[::-1]
    endcoord = endcoord[::-1]
    for bath in searchrange:
        print(bath)
        flag = False
        locations = [seedcoord]
        alllocations = []
        while len(locations)>0 and not flag:
            currentlocation = locations.pop(0)
            for m in moves:
                nextmove = [currentlocation[0]+m[0],currentlocation[1]+m[1]]
                if 0 < nextmove[1] < len(shelf.x) and 0 < nextmove[0] < len(shelf.y)\
                        and depth[nextmove[0],nextmove[1]]<bath and nextmove not in alllocations:
                    if ice[nextmove[0],nextmove[1]]!=0:
                        locations.append(nextmove)
                        alllocations.append(nextmove)
                        if abs(nextmove[0]-endcoord[0]) < 10 and abs(nextmove[1]-endcoord[1]) < 10:
                            flag = True
                            print(bath,seedcoord,nextmove,print(len(shelf.x)))
        if not flag:
            return bath,alllocations
    return None,None

def ocean_search(shelf,start,openocean,lastconnect,lastnotconnect,stepsize=50):
    depth = shelf.bed.values
    ice = shelf.icemask_grounded_and_shelves.values

    moves = list(itertools.product(*[[-stepsize,0,stepsize],[-stepsize,0,stepsize]]))

    #the whole ordering thing confused me so I'm reversing
    start = start[::-1]
    openocean = openocean[::-1]
    lastconnect = lastconnect[::-1]
    lastnotconnect = lastnotconnect[::-1]

    flag = False
    locations = [start]
    alllocations = []
    bath = depth[start[0],start[1]]
    while len(locations)>0 and not flag:
        currentlocation = locations.pop(0)
        for m in moves:
            nextmove = [currentlocation[0]+m[0],currentlocation[1]+m[1]]
            if 0 < nextmove[1] < len(shelf.x) and 0 < nextmove[0] < len(shelf.y)\
                    and depth[nextmove[0],nextmove[1]]<bath and nextmove not in alllocations:
                if ice[nextmove[0],nextmove[1]]!=0:
                    locations.append(nextmove)
                    alllocations.append(nextmove)
                    if (abs(nextmove[0]-lastconnect[0]) < stepsize and abs(nextmove[1]-lastconnect[1]) < stepsize) \
                        or (abs(nextmove[0]-openocean[0]) < stepsize and abs(nextmove[1]-openocean[1]) < stepsize):
                        #plt.scatter(start[0],start[1],c="green")
                        #plt.scatter(openocean[0],openocean[1],c="red")
                        #plt.scatter(lastconnect[0],lastconnect[1],c="red")
                        #plt.scatter(lastnotconnect[0],lastnotconnect[1],c="purple")
                        #alll = np.asarray(alllocations).T
                        #plt.scatter(alll[0],alll[1],c="yellow")
                        #plt.savefig("path.jpg")
                        #plt.close()
                        #print("wait")
                        #input()
                        return True
                    if (abs(nextmove[0]-lastnotconnect[0]) < stepsize and abs(nextmove[1]-lastnotconnect[1]) < stepsize):
                        return False
    return False

def trimDataset(bm,xbounds,ybounds):
    shelf=bm
    shelf = shelf.where(shelf.x<xbounds[1],drop=True)
    shelf = shelf.where(shelf.y<ybounds[1],drop=True)
    shelf = shelf.where(shelf.x>xbounds[0],drop=True)
    shelf = shelf.where(shelf.y>ybounds[0],drop=True)
    return shelf

def FRIS():
    # Load the ice thickness grid
    bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","icemask_grounded_and_shelves"])
    FRISx = [ -3*(10**6),-0.5*(10**6)]
    FRISx = [ -1.5*(10**6),-0.75*(10**6)]
    FRISy = [-0.5*(10**6), 2.5*(10**6)]
    FRIS = trimDataset(bedmap,FRISx,FRISy)
    plt.figure(figsize=(8, 7))
    ax = plt.subplot(111)

    #seed = np.unravel_index(np.argmin(FRIS.values, axis=None), FRIS.values.shape)
    seed=[int(-7.3e+05),int(5e+05)]
    seedcord = selectPoint(FRIS,seed[0],seed[1])
    endcord = selectPoint(FRIS,int(-1.6e+06),int(1.4e+06))
    lastnotconnect = [999999999,9999999999]
    print("for the red dot", ocean_search(FRIS,seedcord,endcord,seedcord,lastnotconnect))

    baths = list(range(-1000,-6,10))[::-1]
    #bath,alllocations = highestBoundingBath(FRIS,"FRIS",seedcord,baths,endcord)
    bath = -600
    print(bath)

    #alllocations = np.asarray(alllocations)
    #pc = bedmap.bed.plot.pcolormesh(
        #ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30),
        #vmax=0
    #)
    pc = FRIS.icemask_grounded_and_shelves.plot.pcolormesh(
        ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30),
    )

    mc, mx, my = highlight_margin(FRIS)
    mcnp = np.asarray(mc)
    mask = []
    print(len(mc))
    lastconnect = endcord
    lastnotconnect = [999999999,9999999999]
    for i in range(len(mc)):
        if i % 10 == 0:
            print(i)
        mask.append(ocean_search(FRIS,mc[i],endcord,lastconnect,lastnotconnect,stepsize=100))
        if mask[-1]:
            lastconnect = mc[i]
        else:
            lastnotconnect = mc[i]

    mx, my, mask = np.asarray(mx),np.asarray(my),np.asarray(mask)
    mask = ~mask
    pe = FRIS.bed.plot.contour(
        ax=ax,levels=range(-1000,0,200),c="red")
    mx = np.asarray(mx[mask])
    my = np.asarray(my[mask])
    print(len(mx)," out of ",len(mc))
    ax.scatter(mx,my,c="red")


    #pd = FRIS.plot.contour(
        #ax=ax,levels=range(-1400,0,200))
    plt.scatter(seed[0],seed[1],s=100,c="red")
    #plt.scatter(FRIS.x[alllocations.T[1,:]],FRIS.y[alllocations.T[0,:]])
    ax.set_title("FRIS Greatest Lower Bath")
    plt.savefig("FRIS.jpg")

def selectPoint(shelf,xval,yval):
    seed=[0,0]
    seed[0] = shelf.sel(x=xval,method="nearest").x
    seed[1] = shelf.sel(y=yval,method="nearest").y
    seedcord = []
    seedcord.append(np.where(shelf.x==seed[0])[0][0])
    seedcord.append(np.where(shelf.y==seed[1])[0][0])
    return seedcord
 
def PIG():
    # Load the ice thickness grid
    bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
    FRISx = [ -1.7*(10**6),-1.4*(10**6)]
    FRISy = [-.4*(10**6), -0.2*(10**6)]
    FRIS = trimDataset(bedmap,FRISx,FRISy)
    plt.figure(figsize=(8, 7))
    ax = plt.subplot(111)
    seed = [int(-1.605e+06),int(-0.275e+06)]
    seedcord = selectPoint(FRIS,seed[0],seed[1])
    endcord = selectPoint(FRIS,int(-1.620e+06),int(-0.380e+06))
    #seed = np.unravel_index(np.argmin(FRIS.values, axis=None), FRIS.values.shape)
    baths = list(range(-620,-200,10))[::-1]
    bath,alllocations = highestBoundingBath(FRIS,"PIG",seedcord,baths,endcord)
    bath=None
    print(bath)
    if not bath:
        bath=-1000
    mc, mx, my = highlight_margin(FRIS)
    mask = []
    print(len(mc))
    lastconnect = endcord
    lastnotconnect = [999999999,9999999999]
    for i in range(len(mc)):
        if i % 10 == 0:
            print(i)
        mask.append(ocean_search(FRIS,mc[i],endcord,lastconnect,lastnotconnect,stepsize=10))
        if mask[-1]:
            lastconnect = mc[i]
        else:
            lastnotconnect = mc[i]

    mx, my, mask = np.asarray(mx),np.asarray(my),np.asarray(mask)
    mx,my = mx[~mask], my[~mask]
    print(len(mx),len(mc))
    pc = FRIS.icemask_grounded_and_shelves.plot.pcolormesh(
        ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30),
    )

    print(bath)
    pe = FRIS.bed.plot.contour(
        ax=ax,levels=[bath],c="red")
    pd = FRIS.bed.plot.contour(
        ax=ax,levels=range(-1000,0,200))
    ax.scatter(mx,my,c="red")
    ax.clabel(
        pd,  # Typically best results when labelling line contours.
        colors=['black'],
        manual=False,  # Automatic placement vs manual placement.
        inline=True,  # Cut the line where the label will be placed.
        fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    )

    #plt.scatter(FRIS.x[alllocations.T[1,:]],FRIS.y[alllocations.T[0,:]])
    ax.set_title("PIG Greatest Lower Bath")
    plt.savefig("PIG.jpg")

def ROSS():
    # Load the ice thickness grid
    bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
    FRISx = [ -0.575*(10**6),0.375*(10**6)]
    FRISy = [-1.5*(10**6), 0]
    FRIS = trimDataset(bedmap,FRISx,FRISy)
    plt.figure(figsize=(8, 7))
    ax = plt.subplot(111)
    seed = [int(-1.605e+06),int(-0.275e+06)]
    seedcord = selectPoint(FRIS,seed[0],seed[1])
    endcord = selectPoint(FRIS,int(-0.45e+06),int(-1.42e+06))
    #seed = np.unravel_index(np.argmin(FRIS.values, axis=None), FRIS.values.shape)
    baths = list(range(-620,-200,10))[::-1]
    #bath,alllocations = highestBoundingBath(FRIS,"PIG",seedcord,baths,endcord)
    bath = -1000
    bath=None
    print(bath)
    if not bath:
        bath=-1000
    pc = FRIS.icemask_grounded_and_shelves.plot.pcolormesh(
        ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30),
    )
    pd = FRIS.bed.plot.contour(
        ax=ax,levels=range(-1000,0,200),cmap=cmocean.cm.gray)
    mc, mx, my = highlight_margin(FRIS)
    mask = []
    lastconnect = endcord
    lastnotconnect = [999999999,9999999999]
    for i in range(len(mc)):
        if i % 10 == 0:
            print(i)
        mask.append(ocean_search(FRIS,mc[i],endcord,lastconnect,lastnotconnect,stepsize=100))
        if mask[-1]:
            lastconnect = mc[i]
        else:
            lastnotconnect = mc[i]
    mx, my, mask = np.asarray(mx),np.asarray(my),np.asarray(mask)
    mask = ~mask
    ax.scatter(mx[mask],my[mask],c="red")
    print(len(mx[mask]),len(mc))
    ax.clabel(
        pd,  # Typically best results when labelling line contours.
        colors=['black'],
        manual=False,  # Automatic placement vs manual placement.
        inline=True,  # Cut the line where the label will be placed.
        fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    )

    #plt.scatter(FRIS.x[endcord[0]],FRIS.y[endcord[1]],s=100,c="red")
    #plt.scatter(FRIS.x[alllocations.T[1,:]],FRIS.y[alllocations.T[0,:]])
    ax.set_title("ROSS Greatest Lower Bath")
    plt.savefig("ROSS.jpg")

def highlight_margin(shelf):
    margin_coords = []
    margin_x = []
    margin_y = []
    icemask = shelf.icemask_grounded_and_shelves.values
    for i in range(1,icemask.shape[0]-1):
        for j in  range(1,icemask.shape[1]-1):
            if icemask[i][j] == 1:
                a = icemask[i+1][j]
                b = icemask[i-1][j]
                c = icemask[i][j+1]
                d = icemask[i][j-1]
                if np.isnan([a,b,c,d]).any():
                    margin_coords.append(tuple([j,i]))
                    margin_x.append(shelf.x[j])
                    margin_y.append(shelf.y[i])
    return margin_coords, margin_x, margin_y

def fullmap():
    plt.figure(figsize=(8, 7))
    ax = plt.subplot(111)
    bedmap = rh.fetch_bedmap2(datasets=["bed"])
    pc = bedmap.bed.plot.pcolormesh(
        ax=ax, cmap=cmocean.cm.topo, cbar_kwargs=dict(pad=0.01, aspect=30),
        vmax=7000
    )
    ax.set_title("Full Map")
    plt.savefig("full.jpg")
#PIG()
#FRIS()
ROSS()
