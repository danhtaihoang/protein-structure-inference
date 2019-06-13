import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

seed = 2
np.random.seed(seed)

d = np.loadtxt('contact_map.txt')
#d = np.random.randint(0,12,size=(3,3)) 
#print(d)
#=========================================================================================
# find value of top smallest
d1 = d.copy()

i = np.loadtxt('cols_remove.txt').astype(int)
d1 = np.delete(d1,i,axis=0)
d1 = np.delete(d1,i,axis=1)

np.savetxt('contact_removed.dat',d1,'%f')

plt.figure(figsize=(3.2,3.2))
plt.title('contact map')
plt.imshow(d1,cmap='rainbow',origin='lower')
plt.xlabel('j')
plt.ylabel('i')
#plt.clim(-0.5,0.5)
plt.colorbar()
#plt.colorbar(fraction=0.045, pad=0.05,ticks=[-0.5,0,0.5])
plt.savefig('ct_removed.pdf', format='pdf', dpi=100)

#======================================
np.fill_diagonal(d1, 1000)
print(d1)

a = d1.reshape((-1,))
#print(a)

a = np.sort(a)
print(a)

top = 1500
top_value = a[top]
print(top_value)
#=========================================================================================
# fill the top smallest to be 1, other 0
top_pos = d1 < top_value
print(top_pos)
d1[top_pos] = 1.
d1[~top_pos] = 0.
print(d1) 

xy = np.argwhere(d1==1)
print(xy)

np.savetxt('contact_top%s.dat'%top,xy,fmt='% i')
