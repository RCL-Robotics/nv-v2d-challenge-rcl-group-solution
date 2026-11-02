import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, trimesh, h5py, json
from validate import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
To=cam_from_world(d['T_wc'],np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'])
f5=h5py.File('/work/inputs/masks.h5'); 
gray=np.load('frames_full_gray.npy',mmap_mode='r')
Sobb,_=trimesh.bounds.oriented_bounds(mesh); Sinv=np.linalg.inv(Sobb); Robb=Sinv[:3,:3]; c=Sinv[:3,3]
fn_lab=torch.tensor(np.argmax(np.asarray(mesh.face_normals)@np.concatenate([Robb,-Robb],1),1)+1,device='cuda')
idx=list(range(4,1003,20)); OM=f5['object'][idx]; PM=f5['person'][idx]
PMfull={i:f5['person'][i] for i in idx}
def flip(T,ax):
    # 180 deg about OBB axis ax through the OBB centre (mesh frame)
    Fm=np.eye(4); Rf=Robb@np.diag([1 if k==ax else -1 for k in range(3)])@Robb.T
    Fm[:3,:3]=Rf; Fm[:3,3]=c-Rf@c
    return T@Fm
res={}
for name,ax in [('ours',None),('flipA',0),('flipB',1),('flipC',2)]:
    T=To[idx] if ax is None else flip(To[idx],ax)
    io=mask_iou(R,d,T,OM,PM)
    class G: pass
    ed=edge_chamfer(R,fn_lab,d,np.concatenate([T]),list(range(len(idx))),[gray[i] for i in idx],[PMfull[i] for i in idx])
    res[name]=(io,ed)
phases=json.load(open('/work/outputs/object_poses_world_diag.json'))['phases']
for k,a,b in phases:
    sel=[j for j,i in enumerate(idx) if a<=i<=b]
    print(k,a,b,' '.join(f"{n}: IoU {res[n][0][sel].mean():.3f} edge {np.nanmean(res[n][1][sel]):.2f} |" for n in res))
