import xarray as xr
import pyproj
import gsw
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import gridspec
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from adjustText import adjust_text
from tqdm import tqdm
import pickle
import rioxarray as riox
import cmocean
from matplotlib.patches import Rectangle
import matplotlib as mpl
import rasterio
from scipy.ndimage import label
from scipy import interpolate
from sklearn.linear_model import LinearRegression
from cdw import pycnocline
from scipy.stats import pearsonr
import ipdb
import random
from sympy import Symbol
from sympy import solve,nsolve,re
from matplotlib.patches import Rectangle

def shelf_class_fig(stats,scalefactor):
    shelf_classnumber = stats["shelf_class"]
    labels = stats["labels"]
    sigmas = stats["sigmas"]
    areas = stats["areas"]
    shelf_color = stats["shelf_color"]
    Btotal = -stats["Btotal"]
    

    fig,ax=plt.subplots(1,1,figsize=(7,9))

    sortlist = np.argsort(np.asarray(shelf_classnumber))
    sorted_nums = list(np.asarray(shelf_classnumber)[sortlist])
    count=0

    
    for c in sorted(np.unique(shelf_classnumber)):
        categorymask = shelf_classnumber==c
        Bcat = Btotal[categorymask]
        sigmacat = np.asarray(sigmas)[categorymask]
        sortmask = np.argsort(Bcat)
        counts = range(count,count+np.sum(shelf_classnumber==c))

        ax.errorbar(Bcat[sortmask],counts,xerr=np.asarray(sigmacat)[sortmask]*(1/(60*60*24*365))*(920.0)*areas[categorymask][sortmask]*34.5/1027*9.8*(7.8*10**(-4)),linestyle='',ecolor="gray",alpha=0.5,capsize=0)
        ax.scatter(Bcat[sortmask],counts,c=np.asarray(shelf_color)[categorymask][sortmask],zorder=2)

        labelstrunc = np.asarray(labels)[categorymask][sortmask]
        for i in range(len(labelstrunc)):
            if np.sign(Bcat[sortmask][i])>0:
                ax.annotate(labelstrunc[i],(Bcat[sortmask][i]+2.5e-5,counts[i]-0.25),horizontalalignment='left')
            if np.sign(Bcat[sortmask][i])<0:
                ax.annotate(labelstrunc[i],(Bcat[sortmask][i]-2.5e-5,counts[i]-0.25),horizontalalignment='right')
            #ax.annotate(labelstrunc[i],(0.001,counts[i]))
        count+=np.sum(shelf_classnumber==c)

    plt.axhspan(0-0.5,sorted_nums.index(-0.5)-0.5,color="fuchsia",alpha=0.1)
    plt.axhspan(sorted_nums.index(-0.5)-0.5,sorted_nums.index(0)-0.5,color="plum",alpha=0.1)

    plt.axhspan(sorted_nums.index(0)-0.5,sorted_nums.index(0.5)-0.5,color="gray",alpha=0.1)

    plt.axhspan(sorted_nums.index(0.5)-0.5,sorted_nums.index(1)-0.5,color="bisque",alpha=0.1)
    plt.axhspan(sorted_nums.index(1)-0.5,len(sorted_nums)-0.5,color="orange",alpha=0.1)

    x1 = ((sorted_nums.index(-0.5)-0.5) + (0-0.5))/2
    x2 = ((sorted_nums.index(-0.5)-0.5) + (sorted_nums.index(0)-0.5))/2
    x3 = ((sorted_nums.index(0)-0.5) + (sorted_nums.index(0.5)-0.5))/2
    x4 = ((sorted_nums.index(0.5)-0.5) + (sorted_nums.index(1)-0.5))/2
    x5 = ((sorted_nums.index(1)-0.5) + len(sorted_nums))/2

    ax.set_yticks([x1,x2,x3,x4,x5])
    ax.set_yticklabels(['disconnected','likely disconnected','unknown or both','likely connected','connected'])

    ax.set_xlim(-1000,1000)
    # ax.set_ylim(-1000,1000)

    ax.set_xticks([-1500,-750,0,750,1500])
    ax.set_xlabel(r"$B_{total}~(m^4 s^{-3})$",fontsize=18)
    ax.axvline(x=0,linestyle='--',color='gray')
    fig.subplots_adjust(left=0.3)
    plt.xticks(fontsize=14,rotation=0)
    plt.yticks(fontsize=14,rotation=0)
    plt.savefig("/home/garrett/Downloads/b0class.svg")




def evaluate_theory(s,colorthresh=5,textthresh=5,mode="linear"):
    rho0 = 1025
    rhoi = 910
    Cp = 4186
    If = 334000
    W0 =  100000
    Bthresh = 50
 
    coldfull = np.full_like(s['slopes'],np.nan)
    for i in tqdm(range(len(s['salts']))):
        ## Let us walk through this because it requires some precisionn
        #x is the area average melt rate in m/yr
        x = Symbol('x',real=True,positive=True)

        # we convert the melt rate to a buoyancy (note the conversion to m/s of melt)
        meltflux = (x/(60*60*24*365))*(920.0)
        # Bpolyna is negative!
        Btotal = ((meltflux*s['areas'][i]*34.5))/rho0*9.8*gsw.beta(34.5,-1.9,np.abs(s['avg_drafts'][i]))+s['Bpolyna'][i]


        ## The equations are in the SI but this solves for g'_dc
        rhomean = (rho0/9.8)*((1/s['fs-1'][i])*(-Btotal))**(1/2)/(np.nanmean(s['front_depth'][i]))
        # stratterm = rhomax-rhomin
        stratterm = rhomean

        rho_s = gsw.beta(34.5,-1.9,0)*rho0
        Spolyna = 34.5 + rhomean/rho_s

        Tf = gsw.CT_freezing(34.5,200,0)
        Tpolyna = gsw.CT_freezing(34.5,0,0)#-1.9
        D = (1-(Cp/If)*(Tf-(Tpolyna))/4)
        gprimes_cold = (9.8/1027)*(Spolyna*(1-1/D)*rho_s + (stratterm/6))

        Tfnew = gsw.CT_freezing(34.5,s['front_thick'][i],0)

        candidate_alpha = 0.02
        C = (candidate_alpha*(rho0*Cp)/(rhoi*If*W0))

        try:
            solveexpr = x - (C*(60*60*24*365))*s['slopes'][i]*s['entrance_thickness'][i]*s['fs-1'][i]*(Tpolyna-Tfnew)*gprimes_cold
            coldfull[i] = float(re(nsolve(solveexpr,x,0)))*s['areas'][i]/candidate_alpha
        except:
            coldfull[i] = np.nan
            
    cmask = np.asarray(-s["Btotal"])<-Bthresh#0.0005
    cold_mys=np.asarray(s['mys']*s['areas'])[cmask]
            
        
    wmask = np.logical_or(np.asarray((-s["Btotal"]))>Bthresh,np.isnan(coldfull))#0.0005
    warm_mys=np.asarray(s['mys']*s['areas'])[wmask]
    warmfull = np.asarray(np.asarray(s['cdws'])*np.asarray(s['Tcdw'])*np.asarray(s['gprimes'])*np.asarray(s['slopes'])*np.asarray(s['fs-1']))*np.asarray(s['areas'])*((rho0*Cp)/(rhoi*If*W0))*(364*24*60*60)

    rhoi = 910
    gigatonconv = 10**(-12)
    scalefactor = rhoi*gigatonconv

    if mode == "log":
        print(coldfull)
        coldfull = np.log10(coldfull*scalefactor)
        warmfull = np.log10(warmfull*scalefactor)
        warm_mys = np.log10(warm_mys*scalefactor)
        cold_mys = np.log10(cold_mys*scalefactor)

    warm = warmfull[wmask]
    warm_xs = np.asarray(([warm])).reshape((-1, 1))
    warm_model = LinearRegression(fit_intercept=False).fit(warm_xs, warm_mys)
    warm_melts = warm_model.predict(warm_xs)

    cold = coldfull[cmask]
    cold_xs = np.asarray(([cold])).reshape((-1, 1))
    cold_model = LinearRegression(fit_intercept=False).fit(cold_xs, cold_mys)
    cold_melts = cold_model.predict(cold_xs)

    alpha_connected =  ((warm_model.coef_))[0]
    alpha_disconnected =  ((cold_model.coef_))[0]

    print('alpha_connected: ',alpha_connected)
    print('alpha_disconnected: ',alpha_disconnected)


    graymask = np.logical_and(np.abs(s["Btotal"])<=Bthresh,~np.isnan(coldfull)) #0.0005
    graycold_xs = np.asarray(([coldfull[graymask]])).reshape((-1,1))
    graywarm_xs = np.asarray(([warmfull[graymask]])).reshape((-1,1))
    if len(graycold_xs)>0:
        gray_melts =(cold_model.predict(graycold_xs) + warm_model.predict(graywarm_xs))/2.0
    else:
        gray_melts = np.asarray([])
    
    if mode == "log":
        gray_mys  = np.log10(np.asarray(s['mys']*s['areas'])[graymask]*scalefactor)
    else:
        gray_mys  = np.asarray(s['mys']*s['areas'])[graymask]

    print(gray_melts)

    r2 = pearsonr(np.concatenate((cold_melts.flatten(),warm_melts.flatten(),gray_melts.flatten())),np.concatenate((cold_mys.flatten(),warm_mys.flatten(),gray_mys.flatten()))).statistic**2
    fig, ax = plt.subplots(1,1)

    if mode == "linear":
        axin1 = ax.inset_axes([5, 80, 50, 55], transform=ax.transData)
        axin1.tick_params(axis='both', labelsize=12)
        ax.indicate_inset_zoom(axin1, edgecolor="black")
        ax.scatter(warm_melts.flatten()*scalefactor,scalefactor*warm_mys,c="orange")
        ax.scatter(cold_melts.flatten()*scalefactor,scalefactor*cold_mys,c="purple")
        ax.scatter(gray_melts.flatten()*scalefactor,scalefactor*gray_mys,c="gray")

        axin1.scatter(warm_melts.flatten()*scalefactor,scalefactor*warm_mys,c="orange")
        axin1.scatter(cold_melts.flatten()*scalefactor,scalefactor*cold_mys,c="purple")
        axin1.scatter(gray_melts.flatten()*scalefactor,scalefactor*gray_mys,c="gray")
    if mode == "log":
        ax.scatter(10**(warm_melts.flatten()),10**(warm_mys),c="orange")
        ax.scatter(10**(cold_melts.flatten()),10**(cold_mys),c="purple")
        ax.scatter(10**(gray_melts.flatten()),10**(gray_mys),c="gray")

    cmaskin = np.cumsum(cmask)-1
    wmaskin = np.cumsum(wmask)-1
    graymaskin = np.cumsum(graymask)-1


    if mode == "linear":
        s['sigmas'] = np.asarray(s['sigmas'])
        for k in range(len(s['labels'])):
            if cmask[k]:
                if cold_melts.flatten()[cmaskin[k]]*scalefactor>20 or (scalefactor*cold_mys)[cmaskin[k]]>20:
                    text=ax.annotate(s['labels'][k],(cold_melts.flatten()[cmaskin[k]]*scalefactor,(scalefactor*cold_mys)[cmaskin[k]]))
                else:
                    text=axin1.annotate(s['labels'][k],(cold_melts.flatten()[cmaskin[k]]*scalefactor,(scalefactor*cold_mys)[cmaskin[k]]))
            elif wmask[k]:
                if warm_melts.flatten()[wmaskin[k]]*scalefactor>20 or (scalefactor*warm_mys)[wmaskin[k]]>20:
                    text=ax.annotate(s['labels'][k],(warm_melts.flatten()[wmaskin[k]]*scalefactor,(scalefactor*warm_mys)[wmaskin[k]]))
                else:
                    text=axin1.annotate(s['labels'][k],(warm_melts.flatten()[wmaskin[k]]*scalefactor,(scalefactor*warm_mys)[wmaskin[k]]))
            elif graymask[k]:
                if gray_melts.flatten()[graymaskin[k]]*scalefactor>20 or (scalefactor*gray_mys)[graymaskin[k]]>20:
                    text=ax.annotate(s['labels'][k],(gray_melts.flatten()[graymaskin[k]]*scalefactor,(scalefactor*gray_mys)[graymaskin[k]]))
                else:
                    text=axin1.annotate(s['labels'][k],(gray_melts.flatten()[graymaskin[k]]*scalefactor,(scalefactor*gray_mys)[graymaskin[k]]))
  
        markers, caps, bars = ax.errorbar(cold_melts.flatten()*scalefactor,scalefactor*cold_mys,yerr=s['sigmas'][cmask]*s['areas'][cmask]*scalefactor,ls='none',ecolor="purple")
        [bar.set_alpha(0.3) for bar in bars]                  
        markers, caps, bars = ax.errorbar(warm_melts.flatten()*scalefactor,scalefactor*warm_mys,yerr=s['sigmas'][wmask]*s['areas'][wmask]*scalefactor,ls='none',ecolor="orange")
        [bar.set_alpha(0.3) for bar in bars]                  
        markers, caps, bars = ax.errorbar(gray_melts.flatten()*scalefactor,scalefactor*gray_mys,yerr=s['sigmas'][graymask]*s['areas'][graymask]*scalefactor,ls='none',ecolor="gray")
        [bar.set_alpha(0.3) for bar in bars]


    if mode == "log":
        s['sigmas'] = np.asarray(s['sigmas'])
        for k in range(len(s['labels'])):
            if cmask[k]:
                text=plt.annotate(s['labels'][k],(10**(cold_melts.flatten())[cmaskin[k]],(10**cold_mys)[cmaskin[k]]))
            elif wmask[k]:
                text=plt.annotate(s['labels'][k],(10**(warm_melts.flatten())[wmaskin[k]],(10**warm_mys)[wmaskin[k]]))
            elif graymask[k]:
                text=plt.annotate(s['labels'][k],(10**(gray_melts.flatten())[graymaskin[k]],(10**gray_mys)[graymaskin[k]]))

        markers, caps, bars = ax.errorbar(10**cold_melts,10**cold_mys,yerr=s['sigmas'][cmask]*s['areas'][cmask]*scalefactor,ls='none',ecolor="purple")
        [bar.set_alpha(0.3) for bar in bars]        
        markers, caps, bars = ax.errorbar(10**warm_melts,10**warm_mys,yerr=s['sigmas'][wmask]*s['areas'][wmask]*scalefactor,ls='none',ecolor="orange")
        [bar.set_alpha(0.3) for bar in bars]         
        markers, caps, bars = ax.errorbar(10**gray_melts,10**gray_mys,yerr=s['sigmas'][graymask]*s['areas'][graymask]*scalefactor,ls='none',ecolor="gray")
        [bar.set_alpha(0.3) for bar in bars]



    ax.text(110, 30, '$r^2=$'+str(round(r2,2)), ha='left', va='top', transform=plt.gca().transData,fontsize=12)
    ax.text(110, 25, r'$\alpha_\mathrm{disconnected}=$'+str(round(alpha_disconnected,5)), ha='left', va='top', transform=plt.gca().transData,fontsize=12)
    ax.text(110, 20, r'$\alpha_\mathrm{connected}=$'+str(round(alpha_connected,5)), ha='left', va='top', transform=plt.gca().transData,fontsize=12)


    ax.tick_params(axis='both', labelsize=18)
    ax.set_xlabel(r"$\dot{M}_{\mathrm{pred}} (Gt/yr)$",fontsize=24)
    ax.set_ylabel(r'$\dot{M}_{\mathrm{obs}} (Gt/yr)$',fontsize=24)

    ax.plot([0,140],[0,140],linestyle='dashed')
    if mode == "linear":
        ax.set_xlim(0,140)
        ax.set_ylim(0,140)
        axin1.set_xlim(0,20)
        axin1.set_ylim(0,20)
        axin1.grid(True)
    if mode == "log":
        ax.set_yscale('log')
        ax.set_xscale('log')
        # ax.set_xlim(-2,2)
        # ax.set_ylim(-2,2)
        # ax.plot([0,2],[0,2],linestyle='dashed')

    ax.grid(True)
    plt.show()
    s['disconnected_estimate'] = coldfull
    s['connected_estimate'] = warmfull
    return s

def optimal_classification(s):


    maxr = 0
    max_shelf_class = []
    r2s = []
    mys = s['mys']
    areas = s['areas']

    for i in tqdm(range(100000)):
        shelf_class = random.choices([-1,0,1],k=len(s['shelf_class']))



        wmask = np.asarray((shelf_class))>0#0.0005
        warm_mys=np.asarray(mys)[wmask]
        warm = s['connected_estimate'][wmask]
        warm_xs = np.asarray(([warm])).reshape((-1, 1))
        warm_model = LinearRegression(fit_intercept=False).fit(warm_xs, warm_mys*areas[wmask])
        warm_melts = warm_model.predict(warm_xs)

        cmask = np.asarray(shelf_class)<0#0.0005
        cold_mys=np.asarray(mys)[cmask]
        cold = s['disconnected_estimate'][cmask]
        cold_xs = np.asarray(([cold])).reshape((-1, 1))
        cold_model = LinearRegression(fit_intercept=False).fit(cold_xs, cold_mys*areas[cmask])
        cold_melts = cold_model.predict(cold_xs)


        graymask = np.asarray(shelf_class)==0#0.0005
        graycold_xs = np.asarray(([s['disconnected_estimate'][graymask]])).reshape((-1,1))
        graywarm_xs = np.asarray(([s['connected_estimate'][graymask]])).reshape((-1,1))
        gray_melts = (cold_model.predict(graycold_xs) + warm_model.predict(graywarm_xs))/2.0
        gray_mys  = np.asarray(mys)[graymask]

        r2 = pearsonr(np.concatenate((cold_melts.flatten(),warm_melts.flatten(),gray_melts.flatten())),np.concatenate((cold_mys.flatten()*areas[cmask],warm_mys.flatten()*areas[wmask],gray_mys.flatten()*areas[graymask]))).statistic**2
        r2s.append(r2)
        if r2>maxr:
            maxr=r2
            max_shelf_class = shelf_class

    plt.hist(r2s,bins=50,color="black",alpha=0.75)

    plt.xticks(fontsize=16,rotation=0)
    plt.yticks(fontsize=16,rotation=0)
    plt.xlabel(r"$r^2$",fontsize=18)
    plt.ylabel('# of classification combinations',fontsize=18)

    plt.show()

def breakdown(stats,colorthresh=5,textthresh=5):
    glfreezing = (gsw.CT_freezing(stats['salts'],0,0)-gsw.CT_freezing(stats['salts'],np.abs(stats['avg_drafts']),0))

    rho0 = 1025
    rhoi = 910
    Cp = 4186
    If = 334000
    W0 =  100000
 
    rhomin,rhomax = (rho0/9.8)*((1/stats['fs-1'])*(-stats['Btotal']*(9.8/1027)))**(1/2)/(stats['h_max']),(rho0/9.8)*((1/stats['fs-1'])*(-stats['Btotal']))**(1/2)/(stats['h_min'])
    rhomean = (rho0/9.8)*((1/stats['fs-1'])*(-stats['Btotal']))**(1/2)/(np.nanmean((stats['h_max']+stats['h_min'])/2))
    stratterm = rhomax-rhomin

    rho_s = gsw.beta(stats['salts'],-1.9,0)*rho0

    Spolyna = stats['salts'] + rhomean/rho_s

    Tf = gsw.CT_freezing(stats['salts'],np.abs(stats['avg_drafts']),0)
    Tpolyna = -1.9
    D = (1-(Cp/If)*(Tf-(Tpolyna))/4)
    gprimes_cold = (9.8/1027)*(Spolyna*(1-1/D)*rho_s + (stratterm/6))


    bar_x = []
    wbar_x = []
    cmask = stats['Btotal']>0
    wmask = stats['Btotal']<0
    for k in range(len(stats['labels'])):
        if cmask[k]:
            bar_x.append(stats['labels'][k])
        if wmask[k]:
            wbar_x.append(stats['labels'][k])
    plt.close()
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.bar(bar_x,np.asarray(stats['entrance_thickness'])[cmask])
    ax1.set_title("H term")
    ax2.bar(bar_x,np.asarray(glfreezing)[cmask])
    ax2.set_title("Thermal term")
    ax3.bar(bar_x,(gprimes_cold)[cmask])
    ax3.set_title("g' term")
    ax4.bar(bar_x,np.asarray(stats['slopes'])[cmask])
    ax4.set_title("slope term")

    ax1.tick_params(axis='x', labelrotation=45)
    ax2.tick_params(axis='x', labelrotation=45)
    ax3.tick_params(axis='x', labelrotation=45)
    ax4.tick_params(axis='x', labelrotation=45)

    plt.show()

    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.bar(wbar_x,np.asarray(stats['cdws'])[wmask])
    ax1.set_title("H term")
    ax2.bar(wbar_x,np.asarray(stats["Tcdw"])[wmask])
    ax2.set_title("Thermal term")
    ax3.bar(wbar_x,(stats["gprimes"])[wmask])
    ax3.set_title("g' term")
    ax4.bar(wbar_x,np.asarray(stats['slopes'])[wmask])
    ax4.set_title("slope term")
    ax1.tick_params(axis='x', labelrotation=45)
    ax2.tick_params(axis='x', labelrotation=45)
    ax3.tick_params(axis='x', labelrotation=45)
    ax4.tick_params(axis='x', labelrotation=45)
    plt.show()



 
