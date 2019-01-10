import numpy as np
import matplotlib.pyplot as plt

di = np.loadtxt('direct_info.dat')
di_sym = np.loadtxt('direct_info_sym.dat')

plt.figure(figsize=(6,3.2))

plt.subplot2grid((1,2),(0,0))
plt.title('direct information')
plt.imshow(di.T,cmap='rainbow',origin='lower')
plt.xlabel('j')
plt.ylabel('i')
plt.clim(0.,0.05)
plt.colorbar(fraction=0.045, pad=0.05,ticks=[0.,0.5])

plt.subplot2grid((1,2),(0,1))
plt.title('direct info from w symmetry')
plt.imshow(di_sym.T,cmap='rainbow',origin='lower')
plt.xlabel('j')
plt.ylabel('i')
plt.clim(0.,0.05)
plt.colorbar(fraction=0.045, pad=0.05,ticks=[0.,0.5])

plt.savefig('direct_info.pdf', format='pdf', dpi=100)
#plt.show()

