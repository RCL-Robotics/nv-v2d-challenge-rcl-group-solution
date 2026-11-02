import sys, time; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'/work/scratch')
from rend import load, Masks
from solve import SilRenderer, project
import numpy as np, torch, trimesh, json, cv2
torch.backends.cudnn.deterministic = True
r,cam=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
Tc=Tcw[None]@r['T_world_mesh'].astype(float)
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
rd=SilRenderer(np.asarray(m.vertices),np.asarray(m.faces),K); MK=Masks(); S=192
c=np.load('/work/scratch/fitcache.npz')
for st in [360,500]:
    ids=list(range(st,st+48)); uv=project(K,Tc[ids,:3,3])
    xy0=c['xy0s'][ids]
    O=[];V=[]
    for k,i in enumerate(ids):
        o,p=MK.crop(i,*xy0[k],S); O.append(o); V.append(~p|o)
    O=np.array(O,np.float32); V=np.array(V,np.float32)
    t0=time.time(); res=[rd.fit(Tc[ids,:3,:3],uv,Tc[ids,2,3],O,V,xy0,S) for _ in range(2)]; dt=(time.time()-t0)/2
    print(st,'time/batch %.1fs'%dt,'R diff',np.abs(res[0][0]-res[1][0]).max(),'t diff mm',1000*np.abs(res[0][1]-res[1][1]).max(), 'iou fd %.3f'%res[0][2].mean())
