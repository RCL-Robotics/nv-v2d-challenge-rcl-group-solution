import numpy as np, sys
from scipy.spatial import cKDTree
d=np.load(sys.argv[1],allow_pickle=True).item(); pts=d['pts']; res=d['res']
FY=np.diag([-1,1,-1.]); flipidx=cKDTree(pts).query(pts@FY.T)[1]
keys=list(res.keys())
M={k:(np.where(res[k][1][:,None]>0,res[k][0]/np.maximum(res[k][1],1)[:,None],np.nan),res[k][1]) for k in keys}
def corr(a,b,ca,cb):
    ok=(ca>=3)&(cb>=3)
    if ok.sum()<80: return np.nan
    return np.corrcoef(a[ok].mean(1),b[ok].mean(1))[0,1]
print('rows: segment; columns: other segments; value: same-corr / flipped-corr')
for i in keys:
    s=[]
    for j in keys:
        if i==j: s.append('     --     '); continue
        a,ca=M[i]; b,cb=M[j]
        s.append('%5.2f/%5.2f'%(corr(a,b,ca,cb),corr(a,b[flipidx],ca,cb[flipidx])))
    print(str(i).ljust(11),' '.join(s))
