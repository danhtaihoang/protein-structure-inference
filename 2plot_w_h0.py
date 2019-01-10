import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

w_infer = np.loadtxt('w.dat')
h0_infer = np.loadtxt('h0.dat') 

#=========================================================================================
plt.figure(figsize=(11,3.2))

plt.subplot2grid((1,3),(0,0))
plt.title('inferred coupling matrix')
plt.imshow(w_infer,cmap='rainbow',origin='lower')
plt.xlabel('j')
plt.ylabel('i')
plt.clim(-0.5,0.5)
plt.colorbar(fraction=0.045, pad=0.05,ticks=[-0.5,0,0.5])

plt.subplot2grid((1,3),(0,1))
plt.plot(h0_infer,'b-',label='infer')
plt.xlabel('i')
plt.ylabel('h0')

plt.tight_layout(h_pad=1, w_pad=1.5)
plt.savefig('w.pdf', format='pdf', dpi=100)
#plt.show()

