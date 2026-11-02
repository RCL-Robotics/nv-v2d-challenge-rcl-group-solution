import sys,time; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoi_lib import *
lam_p=float(sys.argv[1]); lam_r=float(sys.argv[2]); outname=sys.argv[3]
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
center=(mesh.bounds[0]+mesh.bounds[1])/2
R=Renderer(mesh,D['K'],D['W'],D['H'],scale=0.5)
Tc=np.linalg.inv(D['T_wc'])[None]@D['rec']['T_world_mesh'].astype(np.float64)
OM=np.load('om_half.npy'); PM=np.load('pm_half.npy'); BB=np.load('bb.npy')
phases=[('S1',0,252),('M1',252,456),('S2',456,659),('M2',659,720),('S3',720,858)]
def prep(frames):
    bb=BB[frames]; cen=np.stack([(bb[:,0]+bb[:,1])/2,(bb[:,2]+bb[:,3])/2],1)
    off,(omc,pmc)=make_crops([OM[frames],PM[frames]],cen)
    return (torch.tensor(omc,dtype=torch.float32,device='cuda'), torch.tensor(~(pmc&~omc),dtype=torch.float32,device='cuda'),
            (torch.tensor(off,device='cuda',dtype=torch.float32),(192,256)))
out=Tc.copy(); t0=time.time()
for name,s,e in phases:
    if name[0]!='S': continue
    frames=list(range(s,e,3))
    tgt,valid,crop=prep(frames)
    # init: chordal mean of baseline over phase
    from scipy.spatial.transform import Rotation as Rot
    T0=Tc[frames[len(frames)//2]].copy()
    T0[:3,:3]=Rot.from_matrix(Tc[s:e,:3,:3]).mean().as_matrix(); T0[:3,3]=np.median(Tc[s:e,:3,3],0)
    T,L=fit_poses(R,T0[None],tgt,valid,center,iters=300,shared=True,crop=crop)
    out[s:e]=T[0]
    print(name,'static fit',np.round(T[0,:3,3],4),'meanL %.4f'%L.mean(), time.time()-t0)
for k,(name,s,e) in enumerate(phases):
    if name[0]!='M': continue
    frames=list(range(s,e))
    tgt,valid,crop=prep(frames)
    T,L=fit_track(R,Tc[s:e],tgt,valid,center,crop,T_before=out[s-1],T_after=out[e],iters=400,lam_p=lam_p,lam_r=lam_r)
    out[s:e]=T
    print(name,'track fit meanL %.4f'%L.mean(), time.time()-t0)
np.savez(outname,T=out)
