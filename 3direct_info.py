import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#===================================================================================================

# read w, h0:
h0 = np.loadtxt('h0.dat')
w = np.loadtxt('w.dat')

#
s0 = np.loadtxt('s0.txt')

n = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T
      
def direct_info(w):
    di = np.zeros((n,n)) # direct information
    for i in range(n):
        i1,i2 = i1i2[i,0],i1i2[i,1]

        for j in range(n):
            j1,j2 = i1i2[j,0],i1i2[j,1]

            pij = np.exp(w[i1:i2,j1:j2] + h0[i1:i2,np.newaxis] + h0[np.newaxis,j1:j2])            
            pij /= pij.sum()

            pi,pj = pij.sum(axis=1),pij.sum(axis=0)
            di[i,j] = (pij*np.log(pij/np.outer(pi,pj))).sum()
            
    return di        

di = direct_info(w)            
np.savetxt('direct_info.dat',di,fmt='% 3.8f')

# w symmetry
w_sym = (w+w.T)/2.
di_sym = direct_info(w_sym)            
np.savetxt('direct_info_sym.dat',di_sym,fmt='% 3.8f')

#===================================================================================================
# plot
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
