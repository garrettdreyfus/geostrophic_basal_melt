import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr



ds = xr.open_dataset('data/amundsilli.h5')

ds.w_b.plot.pcolormesh()
plt.show()

filename ='data/amundsilli.h5'
is_wb = h5py.File(filename,'r')
print(is_wb)
wb = np.array(is_wb['/w_b'])

x_wb = np.array(is_wb['/x'])
y_wb = np.array(is_wb['/y'])
wb = np.array(is_wb['/w_b'])

fig, ax1 = plt.subplots()
extent = [np.min(is_wb['x']),np.max(is_wb['x']),np.min(is_wb['y']),np.max(is_wb['y'])]
X,Y = np.meshgrid(x_wb,y_wb)
ax1.pcolormesh(X,Y,wb)

with open("data/bedmach.pickle","rb") as f:
    bedmach = pickle.load(f)

icemask = bedmach.icemask_grounded_and_shelves.values
icemask[icemask==1]=np.nan
X,Y = np.meshgrid(bedmach.x.values,bedmach.y.values)
ax1.pcolormesh(X,Y,icemask)

plt.show()
