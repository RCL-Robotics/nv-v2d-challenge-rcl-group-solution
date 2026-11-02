import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from scipy.spatial import cKDTree
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp); mesh=load_mesh(inp)
o=Twc[:3,3]
v=np.asarray(mesh.vertices)[::10]
kp=rec['human_keypoints_world'].astype(np.float64)
cands={'base':rec['T_world_mesh'].astype(np.float64)}
for p in sys.argv[1:]: cands[p.split('/')[-1]]=np.load(p)['T_world_mesh']
F=len(kp)
hands=list(range(21,63))
for name,T in cands.items():
    D=[];S=[]
    for i in range(110,880,2):
        pv=v@T[i,:3,:3].T+T[i,:3,3]; tr=cKDTree(pv)
        h=kp[i,hands]; dd,ii=tr.query(h)
        # per hand: min distance
        for hs in [slice(0,21),slice(21,42)]:
            k=np.argmin(dd[hs]); D.append(dd[hs][k])
            # signed offset along viewing ray of hand point vs nearest object point
            r=h[hs][k]-o; r/=np.linalg.norm(r); S.append((h[hs][k]-pv[ii[hs][k]])@r)
    D=np.array(D);S=np.array(S)
    print(name,'hand-obj min dist median %.1fcm p25 %.1f p75 %.1f | along-ray hand-minus-obj median %.1fcm'%(100*np.median(D),100*np.percentile(D,25),100*np.percentile(D,75),100*np.median(S)))
print('window: along-ray (hand - nearest obj pt) median cm, and min-dist median cm')
for a in range(100,890,40):
    row=[]
    for name,T in cands.items():
        S=[];D=[]
        for i in range(a,min(a+40,890),2):
            pv=v@T[i,:3,:3].T+T[i,:3,3]; tr=cKDTree(pv)
            h=kp[i,hands]; dd,ii=tr.query(h); k=np.argmin(dd)
            r=h[k]-o; r/=np.linalg.norm(r); S.append((h[k]-pv[ii[k]])@r); D.append(dd[k])
        row.append('%s ray %.1f d %.1f'%(name[:5],100*np.median(S),100*np.median(D)))
    print(a,' | '.join(row))
