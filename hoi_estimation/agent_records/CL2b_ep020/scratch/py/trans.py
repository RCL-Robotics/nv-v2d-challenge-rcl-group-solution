import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
K,Twc,Kp,W,H=load_camera('/work/inputs')
v,f,uv,tex=load_mesh('/work/inputs/object_mesh.glb')
r=np.load('/work/inputs/reconstruction.npz')
Twm=r['T_world_mesh'].astype(np.float64)
Tcm=np.linalg.inv(Twc)@Twm
c=v.mean(0)
vw=np.einsum('fij,nj->fni',Twm[:,:3,:3],v[::50].astype(np.float64))+Twm[:,None,:3,3]
bottom=vw[:,:,1].max(1)  # y-down: max y = lowest
top=vw[:,:,1].min(1)
pl=np.array(__import__('json').load(open('/work/inputs/ground_plane.json'))['plane'])
cw=vw.mean(1)
for i in range(0,553,10):
    pc=Tcm[i,:3,:3]@c+Tcm[i,:3,3]
    u=K@pc; u=u[:2]/u[2]
    print(i,'world ctr',np.round(cw[i],3),'bottom y %.3f'%bottom[i],'floor dist bottom %.3f'%(-(pl[:3]@vw[i].T+pl[3])).min(), 'cam z %.3f'%pc[2],'uv',np.round(u,1))
