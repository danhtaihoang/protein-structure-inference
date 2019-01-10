# 2018.12.14: replace element having lower probability (than a threshold) 
# by the element having highest prob

import numpy as np
from scipy.stats import itemfreq

def replace_lower_by_highest_prob(s,p0=0.3):
# input: numpy array, 1 dimension s ; threshold p0
# output: s in which element having p < p0 were placed by element with highest p
    f = itemfreq(s)
    
    # element and number of occurence
    i,p = f[:,0],f[:,1]

    # probabilities
    p = p/float(p.sum()) 

    # find element having highest probability
    ipmax = i[np.argmax(p)]

    # find elements having lower probability than threshold p0
    ipmin = i[np.argwhere(p < p0)]

    # replace
    for ii in ipmin:
        s[np.argwhere(s==ii)] = ipmax
    
    return s

#=========================================================================================
s0 = np.loadtxt('s0_11.txt')
l,n = s0.shape

minfreq = np.zeros(n)
for i in range(n):
    f = itemfreq(s0[:,i])
    minfreq[i] = np.min(f[:,1])
    
print(minfreq)    
#---------------------------------------
s = np.zeros((l,n))
for i in range(n):
    s[:,i] = replace_lower_by_highest_prob(s0[:,i],p0=0.004)

minfreq = np.zeros(n)
for i in range(n):
    f = itemfreq(s[:,i])
    minfreq[i] = np.min(f[:,1])
    
print(minfreq)    

np.savetxt('s0.txt',s,fmt='%i')

#=========================================================================================
def number_residues(s):
    # number of residues at each position
    l,n = s.shape
    mi = np.zeros(n)
    for i in range(n):
        s_unique = np.unique(s[:,i])
        mi[i] = len(s_unique)

    return mi

mi = number_residues(s)

print(mi.mean())
