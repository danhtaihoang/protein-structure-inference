# collect w and h0 from each position i
# 2018.12.22:
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

s0 = np.loadtxt('s0.txt')

n = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T
              
#-----------------------------------------------------------------------------------------     
mx_sum = mx.sum()
w = np.zeros((mx_sum,mx_sum))
h0 = np.zeros(mx_sum)
for i0 in range(n):
    i1,i2 = i1i2[i0,0],i1i2[i0,1]

    w1 = np.loadtxt('W/w_%03d.dat'%(i0))    
    w[:i1,i1:i2] = w1[:i1,:]
    w[i2:,i1:i2] = w1[i1:,:]
    
    h01 = np.loadtxt('h0/h0_%03d.dat'%(i0))
    h0[i1:i2] = h01   

np.savetxt('h0.dat',h0,fmt='% 3.8f')
np.savetxt('w.dat',w,fmt='% 3.8f')

#=========================================================================================
# plot:
#w_infer = np.loadtxt('w.dat')
#h0_infer = np.loadtxt('h0.dat') 

plt.figure(figsize=(11,3.2))

plt.subplot2grid((1,3),(0,0))
plt.title('inferred coupling matrix')
plt.imshow(w,cmap='rainbow',origin='lower')
plt.xlabel('j')
plt.ylabel('i')
plt.clim(-0.5,0.5)
plt.colorbar(fraction=0.045, pad=0.05,ticks=[-0.5,0,0.5])

plt.subplot2grid((1,3),(0,1))
plt.title('inferred coupling matrix - sym')
plt.imshow((w+w.T)/2.,cmap='rainbow',origin='lower')
plt.xlabel('j')
plt.ylabel('i')
plt.clim(-0.5,0.5)
plt.colorbar(fraction=0.045, pad=0.05,ticks=[-0.5,0,0.5])

plt.subplot2grid((1,3),(0,2))
plt.plot(h0,'b-',label='infer')
plt.xlabel('i')
plt.ylabel('h0')

plt.tight_layout(h_pad=1, w_pad=1.5)
plt.savefig('w.pdf', format='pdf', dpi=100)
#plt.show()
