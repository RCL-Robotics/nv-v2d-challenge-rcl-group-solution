"""Evaluate trajectories without GT: silhouette IoU (person pixels excluded), smoothness, floor, depth vs baseline."""
import sys, json, numpy as np, torch, cv2, h5py, trimesh
sys.path.insert(0,'/work/outputs')
from solve import Silhouette, load_masks_half
from scipy.spatial.transform import Rotation as Rot
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
plane=np.array(json.load(open('/work/inputs/ground_plane.json'))['plane'])
mesh=trimesh.load('/work/inputs/object_mesh.glb', force='scene', process=False).to_geometry(); c=mesh.bounds.mean(0)
V=mesh.vertices[::10]
Tb=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
F=len(Tb)
import os
if os.path.exists('/work/scratch/om_eval.npy'):
    om=np.load('/work/scratch/om_eval.npy'); val=np.load('/work/scratch/val_eval.npy')
else:
    om,pm=load_masks_half('/work/inputs/masks.h5',F)
    val=np.array([~(cv2.dilate(pm[i].astype(np.uint8),np.ones((7,7),np.uint8))>0) for i in range(F)])
    np.save('/work/scratch/om_eval.npy',om); np.save('/work/scratch/val_eval.npy',val)
Kh=K.copy(); Kh[:2]*=0.5; Kh[0,2]=(K[0,2]+0.5)*0.5-0.5; Kh[1,2]=(K[1,2]+0.5)*0.5-0.5
sil=Silhouette(mesh,Kh,576,768)
def ious(Tw):
    Tc=Tcw[None]@Tw; out=[]
    for k in range(0,F,32):
        T=torch.tensor(Tc[k:k+32],dtype=torch.float32,device='cuda')
        with torch.no_grad(): A=(sil(T[:,:3,:3],T[:,:3,3])>0.5).cpu().numpy()
        m=om[k:k+32]>0.5; v=val[k:k+32]
        out+=list(((A&m&v).sum((1,2))/np.maximum(((A|m)&v).sum((1,2)),1)))
    return np.array(out)
def stats(Tw):
    Tc=Tcw[None]@Tw; p=Tc[:,:3,:3]@c+Tc[:,:3,3]
    acc=np.r_[0,np.linalg.norm(p[2:]-2*p[1:-1]+p[:-2],axis=1),0]
    R=Rot.from_matrix(Tw[:,:3,:3]); dr=np.r_[0,(R[1:]*R[:-1].inv()).magnitude(),0]; racc=np.r_[0,np.abs(np.diff(dr))[:-1] if False else np.abs(dr[2:]-dr[1:-1]),0] if False else None
    rs=np.degrees(np.r_[0,(R[1:]*R[:-1].inv()).magnitude()])
    h=np.array([((V@T[:3,:3].T+T[:3,3])@plane[:3]+plane[3]).min() for T in Tw])
    return p,acc,rs,h
phases=[('S0 static',0,166),('T1 grasp/shift',167,197),('S1 static held',198,260),('M manipulation',261,596),('S2 on floor',597,793),('ALL',0,793)]
blk=((np.arange(F)//5)%4==2)
res={}
pB,accB,rsB,hB=stats(Tb); iB=ious(Tb)
names=sys.argv[1:]
rows=[]
for nm in names:
    Tw=np.load(nm)['T_world_mesh']; I=ious(Tw); p,acc,rs,h=stats(Tw)
    Rd=np.degrees((Rot.from_matrix(Tw[:,:3,:3])*Rot.from_matrix(Tb[:,:3,:3]).inv()).magnitude())
    print('====',nm)
    for TT,lab in [(Tb,'base'),(Tw,'new')]:
        Pf=np.array([[0.73,0.27,-0.36],[0.72,0.28,0.39]])
        W=np.einsum('fij,pj->fpi',TT[400:,:3,:3],Pf)+TT[400:,None,:3,3]
        sp=np.linalg.norm(W-np.median(W,0),axis=2).mean()*100
        _,_,_,hh=stats(TT); 
        print(f'  [{lab}] pivot foot-end spread 400-793 (cm): {sp:.2f};  lowest-point height std in M (cm): {hh[261:597].std()*100:.2f}')
    print(f'{"phase":18s} {"IoU base":>8s} {"IoU new":>8s} {"hoIoU b":>8s} {"hoIoU n":>8s} {"acc b mm":>8s} {"acc n mm":>8s} {"rstep b":>7s} {"rstep n":>7s} {"floorMin b":>9s} {"floorMin n":>9s} {"dz mean":>7s} {"dz med":>7s} {"dRot":>6s} {"dCtr cm":>7s}')
    for (ph,a,b) in phases:
        s=slice(a,b+1); hb=blk[s]
        print(f'{ph:18s} {iB[s].mean():8.3f} {I[s].mean():8.3f} {iB[s][hb].mean():8.3f} {I[s][hb].mean():8.3f} {accB[s].mean()*1000:8.2f} {acc[s].mean()*1000:8.2f} {rsB[s].mean():7.2f} {rs[s].mean():7.2f} {hB[s].min()*100:9.1f} {h[s].min()*100:9.1f} {(p[s,2]-pB[s,2]).mean()*100:7.2f} {np.median(p[s,2]-pB[s,2])*100:7.2f} {Rd[s].mean():6.2f} {np.linalg.norm(p[s]-pB[s],axis=1).mean()*100:7.2f}')
