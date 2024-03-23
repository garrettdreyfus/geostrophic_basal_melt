import matplotlib.pyplot as plt
import numpy as np
import gsw

lat = -75
Z = np.arange(0,1100,1)
S = np.full_like(Z,1,dtype=float)
S[Z<100]=33.5
S[np.logical_and(Z>=100,Z<200)]=33.5+(Z[np.logical_and(Z>=100,Z<200)]-100)/(100.0)
S[np.logical_and(Z>=200,Z<1100)]=34.5+(Z[np.logical_and(Z>=200,Z<1100)]-200)/(900.0*4)

def t_profile(t_hot=0):
    Z = np.arange(0,1100,1)
    T = np.full_like(Z,1,dtype=float)
    T[Z<100]=-1.8
    T[np.logical_and(Z>=100,Z<200)]=-1.8+(Z[np.logical_and(Z>=100,Z<200)]-100)*(t_hot+1.8)/(100.0)
    T[np.logical_and(Z>=200,Z<1100)]=t_hot
    return T

deltaH = 1000-150
SA = gsw.SA_from_SP(S,Z,0,lat)

melts = []
ts = np.arange(-1.8,2,0.1)
for t_hot in ts:
    print(t_hot)
    T = t_profile(t_hot)
    CT = gsw.CT_from_pt(SA,T)
    rho = gsw.rho(SA,CT,150)
    plt.plot(rho,Z)
    gprime = (np.mean(rho[Z>100]) - np.mean(rho[Z<100]))
    melt = gprime*(t_hot+1.8)*deltaH
    melts.append(melt)

plt.gca().invert_yaxis()
plt.show()

plt.plot(ts,melts)#/(np.max(melts)/9))
plt.show()

