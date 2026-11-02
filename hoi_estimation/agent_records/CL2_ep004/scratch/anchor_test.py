import numpy as np, sys; sys.path.insert(0,'/work/scratch')
from common import *
from scipy.ndimage import median_filter, uniform_filter1d
ep=Episode('/work/inputs')
J=np.load(sys.argv[1] if len(sys.argv)>1 else '/work/scratch/joint3.npz')['Tc']; zb=ep.Tc_base[:,2,3]
s1,s2=110,706; mid=np.arange(s1+1,s2)
kpc=np.einsum('ij,fkj->fki',ep.Tcw[:3,:3],ep.kp_world)+ep.Tcw[:3,3]
hands=np.r_[21:63]
zs=np.linspace(-0.2,0.33,30); seg=np.stack([np.zeros(30),np.zeros(30),zs,np.ones(30)],1)
def hd(T):
    out=[]
    for i in mid:
        P=(T[i]@seg.T)[:3].T; H=kpc[i,hands]
        out.append(np.linalg.norm(H[:,None]-P[None],axis=-1).min())
    return np.array(out)
def anchor(T,scale):
    T=T.copy(); T[:,:3,3]*=scale[:,None]; return T
r=zb/J[:,2,3]
res={}
res['baseline']=hd(ep.Tc_base)
res['raw (no anchor)']=hd(J)
res['phase factor']=hd(anchor(J,np.full(803,np.median(zb[mid])/np.median(J[mid,2,3]))))
for w in [31,61,121,241]:
    res[f'window median {w}']=hd(anchor(J,median_filter(r,size=w,mode='nearest')))
res['per-frame baseline z']=hd(anchor(J,r))
for k,v in res.items(): print('%-25s median %.3f mean %.3f  p90 %.3f'%(k,np.median(v),v.mean(),np.percentile(v,90)))
print('---')
from scipy.ndimage import gaussian_filter1d
for nm,zt in [('zb raw',zb),('zb median5',median_filter(zb,5,mode='nearest')),('zb median9',median_filter(zb,9,mode='nearest')),
              ('zb med9+gauss2',gaussian_filter1d(median_filter(zb,9,mode='nearest'),2,mode='nearest')),('zb med15+gauss3',gaussian_filter1d(median_filter(zb,15,mode='nearest'),3,mode='nearest'))]:
    v=hd(anchor(J,zt/J[:,2,3])); acc=np.abs(np.diff(zt[mid],2))
    print('%-18s hand median %.3f mean %.3f p90 %.3f | z accel mean %.4f max %.3f'%(nm,np.median(v),v.mean(),np.percentile(v,90),acc.mean(),acc.max()))
