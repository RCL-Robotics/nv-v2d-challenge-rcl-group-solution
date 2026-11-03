import numpy as np, sys
sys.argv=[sys.argv[0]]+sys.argv[1:]
exec(open('/work/scratch/photocons.py').read().split("if __name__")[0])
from scipy.spatial import cKDTree
Tc=load_cam_traj(sys.argv[1])
refsegs=[(0,150),(330,380),(445,565)]
S=np.zeros((len(pts),3)); C=np.zeros(len(pts))
for a,b in refsegs:
    s,c=accumulate(Tc,range(a,b)); S+=s; C+=c
ref=np.where(C[:,None]>0,S/np.maximum(C,1)[:,None],np.nan)
FY=np.diag([-1,1,-1.]); flipidx=cKDTree(pts).query(pts@FY.T)[1]
def corr(a,b,ca,cb):
    ok=(ca>=3)&(cb>=2)
    if ok.sum()<60: return np.nan
    return np.corrcoef(a[ok].mean(1),b[ok].mean(1))[0,1]
for w0 in range(150,445,6):
    s,c=accumulate(Tc,range(w0,w0+6))
    m=np.where(c[:,None]>0,s/np.maximum(c,1)[:,None],np.nan)
    print(w0,'same %.2f flip %.2f'%(corr(ref,m,C,c),corr(ref,m[flipidx],C,c[flipidx])),flush=True)
