import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

xy1 = np.loadtxt('contact_top1500.dat')
xy2 = np.loadtxt('direct_info_top200.dat')
xy2_sym = np.loadtxt('direct_info_sym_top200.dat')

#=========================================================================================
plt.figure(figsize=(6,3.2))

plt.subplot2grid((1,2),(0,0))
plt.title('direct information')
plt.scatter(xy1[:,0],xy1[:,1],marker='^',color='b',label='contact',s=5)
plt.scatter(xy2[:,0],xy2[:,1],marker='o',color='r',label='direct-info',s=5)
plt.xlabel('j')
plt.ylabel('i')
plt.legend()

plt.subplot2grid((1,2),(0,1))
plt.title('direct info from w symmetry')
plt.scatter(xy1[:,0],xy1[:,1],marker='^',color='b',label='contact',s=5)
plt.scatter(xy2_sym[:,0],xy2_sym[:,1],marker='o',color='r',label='direct-info',s=5)
plt.xlabel('j')
plt.ylabel('i')
plt.legend()

plt.savefig('contact1500_direct200.pdf', format='pdf', dpi=100)
#plt.show()
