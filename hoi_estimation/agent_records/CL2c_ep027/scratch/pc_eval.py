import numpy as np, sys
from scipy.spatial import cKDTree
d=np.load(sys.argv[1],allow_pickle=True).item(); pts=d['pts']; res=d['res']
FY=np.diag([-1,1,-1.])
tree=cKDTree(pts); flipidx=tree.query(pts@FY.T)[1]
def mean(S,C): 
    m=np.where(C[:,None]>0,S/np.maximum(C,1)[:,None],np.nan); return m
keys=list(res.keys())
def corr(a,b,ca,cb):
    ok=(ca>=3)&(cb>=3)
    if ok.sum()<50: return np.nan, ok.sum()
    x=a[ok]; y=b[ok]
    # use luminance and chroma (r/g)
    xl=x.mean(1); yl=y.mean(1)
    xr=x[:,2]/x[:,1]; yr=y[:,2]/y[:,1]
    c1=np.corrcoef(xl,yl)[0,1]; c2=np.corrcoef(xr,yr)[0,1]
    return (round(c1,3),round(c2,3)), ok.sum()
for i in range(len(keys)):
    for j in range(i+1,len(keys)):
        Sa,Ca=res[keys[i]]; Sb,Cb=res[keys[j]]
        ma,mb=mean(Sa,Ca),mean(Sb,Cb)
        same=corr(ma,mb,Ca,Cb)
        fl=corr(ma,mb[flipidx],Ca,Cb[flipidx])
        print(keys[i],keys[j],'same',same,'flipped',fl)
