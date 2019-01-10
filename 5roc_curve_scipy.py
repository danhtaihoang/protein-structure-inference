# 2018.12.27: Receiver operating characteristic (ROC) curve

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import metrics

#di = np.loadtxt('direct_info.dat').astype(float)
di = np.loadtxt('direct_info_sym.dat').astype(float)
ct = np.loadtxt('contact_removed.dat').astype(float)

#===================================================================================================
# 2018.12.28: calculate ROC using scipy
def roc_curve_scipy(ct,di,ct_thres=10):    
    ct1 = ct.copy()
    
    ct_pos = ct1 < ct_thres           
    ct1[ct_pos] = 1
    ct1[~ct_pos] = 0

    mask = np.triu(np.ones(di.shape[0],dtype=bool), k=1)
    order = di[mask].argsort()[::-1]

    ct_flat = ct1[mask][order]
    di_flat = di[mask][order]

    # scipy
    fp, tp, thresholds = metrics.roc_curve(ct_flat,di_flat)
        
    return tp,fp  
#===================================================================================================
def roc_curve(ct,di,ct_thres=10):    
    ct1 = ct.copy()
    
    ct_pos = ct1 < ct_thres           
    ct1[ct_pos] = 1
    ct1[~ct_pos] = 0

    mask = np.triu(np.ones(di.shape[0],dtype=bool), k=2)
    order = di[mask].argsort()[::-1]

    ct_flat = ct1[mask][order]

    tp = np.cumsum(ct_flat, dtype=float)
    fp = np.cumsum(~ct_flat.astype(int), dtype=float)

    tp /= tp[-1]
    fp /= fp[-1]
        
    return tp,fp     
#===================================================================================================
# find optimal threshold of distance
ct_thres = np.linspace(2,22,41,endpoint=True)
n = ct_thres.shape[0]

auc = np.zeros(n)
for i in range(n):
    #tp,fp = roc_curve(ct,di,ct_thres[i])
    tp,fp = roc_curve_scipy(ct,di,ct_thres[i])
    
    #auc[i] = tp.sum()/tp.shape[0]    
    auc[i] = metrics.auc(fp,tp)
    print(ct_thres[i],auc[i])    

np.savetxt('auc_scipy.dat',(ct_thres,auc),fmt='%f')
#---------------------------------------------------
# plot at optimal threshold:
i0 = np.argmax(auc)    
print('auc max:',ct_thres[i0],auc[i0])
tp0,fp0 = roc_curve(ct,di,ct_thres[i0])

plt.figure(figsize=(6.5,3.2))

plt.subplot2grid((1,2),(0,0))
plt.title('ROC with distance threshold = %3.2f'%(ct_thres[i0]))
plt.plot(fp0,tp0,'r-')
plt.plot([0,1],[0,1],'k--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.subplot2grid((1,2),(0,1))
plt.title('AUC max = %f' %(auc[i0]))
plt.plot(ct_thres,auc,'r-')
plt.xlim([ct_thres.min(),ct_thres.max()])
plt.ylim([0,auc.max()+0.1])
plt.xlabel('distance threshold')
plt.ylabel('AUC')

plt.tight_layout(h_pad=1, w_pad=1.5)
plt.savefig('roc_curve_scipy.pdf', format='pdf', dpi=100)

#plt.show()

#===================================================================================================

