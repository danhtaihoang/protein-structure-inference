import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import inference

#=========================================================================================
np.random.seed(1)

# read integer data
s0 = np.loadtxt('s0.txt')
 
# convert to onehot
onehot_encoder = OneHotEncoder(sparse=False)
s = onehot_encoder.fit_transform(s0) 

n = s0.shape[1]
mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])
mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T
              
#-----------------------------------------------------------------------------------------
#i0 = 1
i0 = sys.argv[1]
print(i0)
i0 = int(i0)        
       
i1,i2 = i1i2[i0,0],i1i2[i0,1]
x = np.hstack([s[:,:i1],s[:,i2:]])
y = s[:,i1:i2]

w,h0,cost = inference.fit_additive(x,y,nloop=10)

np.savetxt('W/w_%03d.dat'%(i0),w,fmt='% 3.8f')
np.savetxt('h0/h0_%03d.dat'%(i0),h0,fmt='% 3.8f')
np.savetxt('cost/cost_%03d.dat'%(i0),cost,fmt='% 3.8f')

print('finished')
