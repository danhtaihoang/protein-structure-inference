import numpy as np

seed = 2
np.random.seed(seed)

d = np.loadtxt('direct_info.dat')
d_sym = np.loadtxt('direct_info_sym.dat')

#=========================================================================================
def direct_top(d,top=100):

    # find value of top biggest
    d1 = d.copy()
    np.fill_diagonal(d1, 0)
    #print(d1)
    
    a = d1.reshape((-1,))
    #print(a)
    
    a = np.sort(a)[::-1] # descresing sort
    #print(a)

    top_value = a[top]
    #print(top_value)
       
    # fill the top largest to be 1, other 0
    top_pos = d1 > top_value
    #print(top_pos)
    d1[top_pos] = 1.
    d1[~top_pos] = 0.
    #print(d1) 
    
    xy = np.argwhere(d1==1)   
    return xy
    
#=========================================================================================
tops = [50,100,200]
for top in tops:
    xy = direct_top(d,top)
    xy_sym = direct_top(d_sym,top) 
    np.savetxt('direct_info_top%s.dat'%top,xy,fmt='%i')
    np.savetxt('direct_info_sym_top%s.dat'%top,xy_sym,fmt='%i')
