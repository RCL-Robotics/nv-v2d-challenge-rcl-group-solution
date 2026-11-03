import numpy as np, json
from opt import *
d=np.load('../run2/diag.npz'); Rf,tf=d['Rf'],d['tf']; sc=json.loads(str(d['depth_scale']))
tun=tf.copy()
for k,s in sc.items():
    a,b=map(int,k.split('-')); tun[a:b]/=s
tip=np.array([0,0,1.015])-c_mesh
headpts=np.array([[x,0,-0.50] for x in np.linspace(-0.2,0.2,9)])-c_mesh
def heights(R,t,P): return (np.einsum('fij,nj->fni',R,P)+t[:,None])@n_c+d_c
for name,t in [('anchored',tf),('unanchored',tun),('baseline',None)]:
    R=Rf if t is not None else Rb; tt=t if t is not None else tb
    th=heights(R,tt,tip[None])[:,0]; hh=heights(R,tt,headpts).min(1)
    Xh=np.einsum('fij,nj->fni',R,handle_pts)+tt[:,None]
    dh=np.linalg.norm(hands_c[:,:,None]-Xh[:,None],axis=-1).min(-1)
    dh=np.where(contact>0.5,dh,np.nan); dmin=np.nanmin(np.where(np.isnan(dh),9,dh),1); dmin[dmin==9]=np.nan
    print(name,'tip h start %.1f end %.1f cm | head low pt start %.1f end %.1f | hand dist med moving %.1f cm'%(np.median(th[:118])*100,np.median(th[908:])*100,np.median(hh[:118])*100,np.median(hh[908:])*100,np.nanmedian(dmin[118:908])*100))
# depth shift needed to put tip on floor (along ray of the object centre, scaling)
for (a,b) in [(0,118),(908,988)]:
    R=Rf[a]; t=tf[a]
    best=None
    for s in np.linspace(0.9,1.1,2001):
        h=(R@tip+t*s)@n_c+d_c
        if best is None or abs(h)<abs(best[1]): best=(s,h)
    print('phase',a,b,'scale to put tip on floor (from anchored)',best[0],'-> centre depth shift %.1f cm'%((best[0]-1)*t[2]*100))
