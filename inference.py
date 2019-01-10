##========================================================================================
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder

#=========================================================================================
def itab(n,m):    
    i1 = np.zeros(n)
    i2 = np.zeros(n)
    for i in range(n):
        i1[i] = i*m
        i2[i] = (i+1)*m

    return i1.astype(int),i2.astype(int)
#=========================================================================================
# generate coupling matrix w0: wji from j to i
def generate_interactions(n,m,g,sp):
    nm = n*m
    w = np.random.normal(0.0,g/np.sqrt(nm),size=(nm,nm))
    i1tab,i2tab = itab(n,m)

    # sparse
    for i in range(n):
        for j in range(n):
            if (j != i) and (np.random.rand() < sp): 
                w[i1tab[i]:i2tab[i],i1tab[j]:i2tab[j]] = 0.
    
    # sum_j wji to each position i = 0                
    for i in range(n):        
        i1,i2 = i1tab[i],i2tab[i]              
        w[:,i1:i2] -= w[:,i1:i2].mean(axis=1)[:,np.newaxis]            

    # no self-interactions
    for i in range(n):
        i1,i2 = i1tab[i],i2tab[i]
        w[i1:i2,i1:i2] = 0.   # no self-interactions

    # symmetry
    for i in range(nm):
        for j in range(nm):
            if j > i: w[i,j] = w[j,i]       
        
    return w
#=========================================================================================
def generate_external_local_field(n,m,g):  
    nm = n*m
    h0 = np.random.normal(0.0,g/np.sqrt(nm),size=nm)

    i1tab,i2tab = itab(n,m) 
    for i in range(n):
        i1,i2 = i1tab[i],i2tab[i]
        h0[i1:i2] -= h0[i1:i2].mean(axis=0)

    return h0
#=========================================================================================
# 2018.10.27: generate time series by MCMC
def generate_sequences(w,h0,n,m,l): 
    i1tab,i2tab = itab(n,m)
    
    # initial s (categorical variables)
    s_ini = np.random.randint(0,m,size=(l,n)) # integer values

    # onehot encoder 
    enc = OneHotEncoder(n_values=m)
    s = enc.fit_transform(s_ini).toarray()

    nrepeat = 5*n
    for irepeat in range(nrepeat):
        for i in range(n):
            i1,i2 = i1tab[i],i2tab[i]

            h = h0[np.newaxis,i1:i2] + s.dot(w[:,i1:i2])  # h[t,i1:i2]

            k0 = np.argmax(s[:,i1:i2],axis=1)
            for t in range(l):
                k = np.random.randint(0,m)                
                while k == k0[t]:
                    k = np.random.randint(0,m)
                
                if np.exp(h[t,k] - h[t,k0[t]]) > np.random.rand():
                    s[t,i1:i2],s[t,i1+k] = 0.,1.

        if irepeat%n == 0: print('irepeat:',irepeat) 

    return s

#===================================================================================================
# 2018.12.22: inverse of covariance between values of x
def cov_inv(x,y):
    l,mx = x.shape
    my = y.shape[1]
    cab_inv = np.empty((my,my,mx,mx))
    
    for ia in range(my):
        for ib in range(my):
            if ib != ia:
                eps = y[:,ia] - y[:,ib]

                which_ab = eps !=0.                    
                xab = x[which_ab]          
                xab_av = np.mean(xab,axis=0)
                dxab = xab - xab_av
                cab = np.cov(dxab,rowvar=False,bias=True)

                cab_inv[ia,ib,:,:] = linalg.pinv(cab,rcond=1e-15)    
                
    return cab_inv
#===================================================================================================
# 2018.12.22: fit interactions from x to y
def fit(x,y,nloop=50):
    l,mx = x.shape
    my = y.shape[1]
    
    # inverse of covariance between values of x    
    cab_inv = np.empty((my,my,mx,mx))    
    for ia in range(my):
        for ib in range(my):
            if ib != ia:
                eps = y[:,ia] - y[:,ib]

                which_ab = eps !=0.                    
                xab = x[which_ab]          
                xab_av = np.mean(xab,axis=0)
                dxab = xab - xab_av
                cab = np.cov(dxab,rowvar=False,bias=True)

                cab_inv[ia,ib,:,:] = linalg.pinvh(cab,rcond=1e-15)        
    #------------------------------------------------                                           
    w = np.random.normal(0.0,1./np.sqrt(mx),size=(mx,my))
    h0 = np.random.normal(0.0,1./np.sqrt(mx),size=my)    
    cost = np.full(nloop,100.) 
    for iloop in range(nloop):
        h = h0[np.newaxis,:] + x.dot(w)

        # stopping criterion --------------------
        p = np.exp(h)
        p /= p.sum(axis=1)[:,np.newaxis]

        cost[iloop] = ((y - p)**2).mean()
        if iloop > 1 and cost[iloop] >= cost[iloop-1]: break
        #-----------------------------------------    

        for ia in range(my):
            wa = np.zeros(mx)
            ha0 = 0.
            for ib in range(my):
                if ib != ia:
                    eps = y[:,ia] - y[:,ib]
                    which_ab = eps!=0.

                    eps_ab = eps[which_ab]
                    xab = x[which_ab]

                    # ----------------------------
                    xab_av = xab.mean(axis=0)
                    dxab = xab - xab_av

                    h_ab = h[which_ab,ia] - h[which_ab,ib]

                    which_h = h_ab!=0
                    ha = eps_ab[which_h]*h_ab[which_h]/np.tanh(h_ab[which_h]/2.)

                    dhdx = dxab*((ha - ha.mean())[:,np.newaxis])
                    dhdx_av = dhdx.mean(axis=0)

                    wab = cab_inv[ia,ib,:,:].dot(dhdx_av)   # wa - wb
                    h0ab = ha.mean() - xab_av.dot(wab)      # ha0 - hb0

                    wa += wab
                    ha0 += h0ab

            w[:,ia] = wa/my
            h0[ia] = ha0/my
            
    return w,h0,cost  
    
#=========================================================================================
# 2018.12.28: fit interaction to residues at position i
# additive update
def fit_additive(x,y,nloop=50):    
    mx = x.shape[1]
    my = y.shape[1]
    
    x_av = x.mean(axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)
    c_inv = linalg.pinvh(c)

    w = np.random.normal(0.0,1./np.sqrt(mx),size=(mx,my))
    h0 = np.random.normal(0.0,1./np.sqrt(mx),size=my)

    cost = np.full(nloop,100.)         
    for iloop in range(nloop):
        h = h0[np.newaxis,:] + x.dot(w)

        p = np.exp(h)
        p_sum = p.sum(axis=1)            
        p /= p_sum[:,np.newaxis]

        cost[iloop] = ((y - p)**2).mean()
        print(iloop,cost[iloop])
        if iloop > 1 and cost[iloop] >= cost[iloop-1]: break

        h += y - p

        h_av = h.mean(axis=0)
        dh = h - h_av

        dhdx = dh[:,np.newaxis,:]*dx[:,:,np.newaxis]
        dhdx_av = dhdx.mean(axis=0)
        w = c_inv.dot(dhdx_av)            
        w -= w.mean(axis=0) 

        h0 = h_av - x_av.dot(w)
        h0 -= h0.mean()

    return w,h0,cost                     
