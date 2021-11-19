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

def ocean_search(shelf,start,openocean,lastconnect,lastnotconnect,stepsize=10):
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
    bath = depth[start[0],start[1]]+15
    baths = []
    for m in moves:
        nextmove = [start[0]+m[0],start[1]+m[1]]
        if 0 < nextmove[1] < len(shelf.x) and 0 < nextmove[0] < len(shelf.y):
            baths.append(depth[nextmove[0],nextmove[1]])
    baths=np.nanmax(baths)+10
                
    while len(locations)>0 and not flag:
        currentlocation = locations.pop(0)
        for m in moves:
            nextmove = [currentlocation[0]+m[0],currentlocation[1]+m[1]]
            if 0 < nextmove[1] < len(shelf.x) and 0 < nextmove[0] < len(shelf.y)\
                    and depth[nextmove[0],nextmove[1]]<bath and nextmove not in alllocations:
                if ice[nextmove[0],nextmove[1]]!=0:
                    locations.append(nextmove)
                    alllocations.append(nextmove)
                    if (abs(nextmove[0]-openocean[0]) < stepsize and abs(nextmove[1]-openocean[1]) < stepsize):
                        return True
    #plt.scatter(start[0],start[1],c="green")
    #plt.scatter(openocean[0],openocean[1],c="red")
    #alll = np.asarray(alllocations).T
    #print(alll)
    #for m in moves:
        #nextmove = [currentlocation[0]+m[0],currentlocation[1]+m[1]]
        #print(m)
        #depth[nextmove[0],nextmove[1]]
    #plt.scatter(alll[0],alll[1],c="yellow")
    ##plt.savefig("path.jpg")
    #plt.close()
    #print("wait")
    #input()
    return False

def trimDataset(bm,xbounds,ybounds):
    shelf=bm
    shelf = shelf.where(shelf.x<xbounds[1],drop=True)
    shelf = shelf.where(shelf.y<ybounds[1],drop=True)
    shelf = shelf.where(shelf.x>xbounds[0],drop=True)
    shelf = shelf.where(shelf.y>ybounds[0],drop=True)
    return shelf

def selectPoint(shelf,xval,yval):
    seed=[0,0]
    seed[0] = shelf.sel(x=xval,method="nearest").x
    seed[1] = shelf.sel(y=yval,method="nearest").y
    seedcord = []
    seedcord.append(np.where(shelf.x==seed[0])[0][0])
    seedcord.append(np.where(shelf.y==seed[1])[0][0])
    return seedcord

def points_not_connected(bedmap,xbounds,ybounds,endcord,name):
    # Load the ice thickness grid
    shelf = trimDataset(bedmap,xbounds,ybounds)
    endcord = selectPoint(shelf,int(endcord[0]),int(endcord[1]))
    mc, mx, my = highlight_margin(shelf)
    mask = []
    lastconnect = endcord
    lastnotconnect = [999999999,9999999999]
    for i in range(len(mc)):
        mask.append(ocean_search(shelf,mc[i],endcord,lastconnect,lastnotconnect,stepsize=10))
        #if mask[-1]:
            #lastconnect = mc[i]
        #else:
            #lastnotconnect = mc[i]
    mx, my, mask = np.asarray(mx),np.asarray(my),np.asarray(mask)
    mask = ~mask

    ## plotting
    plt.figure(figsize=(8, 7))
    ax = plt.subplot(111)
    pc = shelf.bed.plot.pcolormesh(
        ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30),vmin=-500,vmax=0
    )
    ax.scatter(mx[mask],my[mask],c="red")
    pd = shelf.bed.plot.contour(
        ax=ax,levels=range(-1000,0,100),cmap=cmocean.cm.gray)
    ax.clabel(
        pd,  # Typically best results when labelling line contours.
        colors=['black'],
        manual=False,  # Automatic placement vs manual placement.
        inline=True,  # Cut the line where the label will be placed.
        fmt=' {:.0f} '.format,  # Labes as integers, with some extra space.
    )
    plt.savefig("{}.jpg".format(name))
    return (len(mx[mask]),len(mc))



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
def FRIS():
    # Load the ice thickness grid
    bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
    xbounds = [ -1.5*(10**6),-0.75*(10**6)]
    ybounds = [-0.5*(10**6), 2.5*(10**6)]
    endcord = (int(-1.6e+06),int(1.4e+06))
    points_not_connected(bedmap,xbounds,ybounds,endcord,"FRIS")


def PIG():
    # Load the ice thickness grid
    bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
    xbounds = [ -1.7*(10**6),-1.4*(10**6)]
    ybounds = [-.4*(10**6), -0.2*(10**6)]
    endcord = (int(-1.620e+06),int(-0.380e+06))

    points_not_connected(bedmap,xbounds,ybounds,endcord,"PIG")

def ROSS():
    # Load the ice thickness grid
    bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
    xbounds = [ -0.575*(10**6),0.375*(10**6)]
    ybounds = [-1.5*(10**6), 0]
    endcord = (int(-0.45e+06),int(-1.42e+06))

    points_not_connected(bedmap,xbounds,ybounds,endcord,"Ross")

shelves={
        "FRIS": {"xbounds":[ -1.5*(10**6),-0.75*(10**6)],
                "name" : "FRIS",
                "ybounds":[-0.5*(10**6), 2.5*(10**6)],
                "endcord":(int(-1.6e+06),int(1.4e+06)),
                "massloss":43+-77,
                "area":2146+644.7,
                "not_connected_percentage":(336/820)
        },
        "PIG": {"xbounds":[ -1.7*(10**6),-1.4*(10**6)],
                "name" : "PIG",
                "ybounds":[-.4*(10**6), -0.2*(10**6)],
                "endcord":(int(-1.620e+06),int(-0.380e+06)),
                "massloss":-1066,
                "area":181.4,
                "not_connected_percentage":(14/145)
        },
        "ROSS": {
                "name" : "ROSS",
                "xbounds":[ -0.575*(10**6),0.375*(10**6)],
                "ybounds":[-1.5*(10**6), 0],
                "endcord":(int(-0.45e+06),int(-1.42e+06)),
                "massloss":-32+916,
                "area":788+1649,
                "not_connected_percentage":(648/908)
        },
        "Wilkins": {"xbounds":[-2.2e6,-1.925e6],
                "name" : "Wilkins",
                "ybounds":[0.4575e6,0.75e6],
                "endcord":[-2.1e6,0.5e6],
                "massloss":0,
                "area":14.7,
                "not_connected_percentage":(65/ 316),
        },
       "Larsen C": {"xbounds":[-2.4e6,-1.75e6],
                "name" : "Larsen C",
                "ybounds":[0.95e6,1.355e6],
                "endcord":[-1.8e6,1.3e6],
                "massloss":-32,
                "area":18.1,
                "not_connected_percentage":(196/731)
        },
       "Riiser-Larsen": {"xbounds":[-0.8e6,0e6],
                "name" : "Riiser-Larsen",
                "ybounds":[1.3e6,2e6],
                "endcord":[-0.7e6,1.7e6],
                "massloss":25,
                "area":92.8,
                "not_connected_percentage":(37/880)
        },
       "Mertz": {"xbounds":[-0.8e6,0e6],
                "name" : "Mertz",
                "ybounds":[1.3e6,2e6],
                "endcord":[-0.7e6,1.7e6],
                "massloss":25,
                "area":92.8,
                "not_connected_percentage": 0,
        },
       "Cook": {"xbounds":[-0.8e6,0e6],
                "name" : "Cook",
                "ybounds":[1.3e6,2e6],
                "endcord":[-0.7e6,1.7e6],
                "massloss":25,
                "area":92.8,
                "not_connected_percentage":0,
        },
       "Amery": {
                "name" : "Amery",
                "xbounds":[1.675e6,2.3e6],
                "ybounds":[0.5e6,0.9e6],
                "endcord":[2.29e6,0.7e6],
                "massloss":-65,
                "area":1338.2,
                "not_connected_percentage":55/442
        }}

def not_connected_percentage(shelf_details):
    # Load the ice thickness grid
    bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
    print(points_not_connected(bedmap,shelf_details["xbounds"],shelf_details["ybounds"],shelf_details["endcord"],shelf_details["name"]))

#for k in shelves.keys():
    #print(k)
    #not_connected_percentage(shelves[k])
if False:
    scaledmassloss = []
    metric = []
    labels=[]
    for k in shelves.keys():
        labels.append(shelves[k]["name"])
        scaledmassloss.append(shelves[k]["massloss"]/shelves[k]["area"])
        metric.append(shelves[k]["not_connected_percentage"])

    plt.figure(figsize=(8, 7))
    ax = plt.subplot(111)
    for i, txt in enumerate(labels):
        ax.annotate(txt, (metric[i], scaledmassloss[i]))
    ax.scatter(metric,scaledmassloss)
    ax.set_xlabel("Perecentage of not connected ice shelf margin points")
    ax.set_ylabel("Mass Loss / Ice Shelf Area")
    plt.savefig("metricd.png")




bedmap = rh.fetch_bedmap2(datasets=["bed","thickness","surface","icemask_grounded_and_shelves"])
xbounds= [1.5e6,2.5e6]
ybounds= [1e6,2e6]
endcord = [-0.7e6,1.7e6]
#shelf = trimDataset(bedmap,xbounds,ybounds)
plt.figure(figsize=(8, 7))
ax = plt.subplot(111)
ax.scatter(endcord[0],endcord[1],c="red")

def trimDataset(bm,xcenter,ycenter,width):
    shelf=bm
    shelf = shelf.where(shelf.x<xcenter+width,drop=True)
    shelf = shelf.where(shelf.y<ycenter+width,drop=True)
    shelf = shelf.where(shelf.x>xcenter-width,drop=True)
    shelf = shelf.where(shelf.y>ycenter-width,drop=True)
    return shelf

# Fimbul

pc = shelf.icemask_grounded_and_shelves.plot.pcolormesh(
   ax=ax, cmap=cmocean.cm.haline, cbar_kwargs=dict(pad=0.01, aspect=30),
)
plt.savefig("search.png")
