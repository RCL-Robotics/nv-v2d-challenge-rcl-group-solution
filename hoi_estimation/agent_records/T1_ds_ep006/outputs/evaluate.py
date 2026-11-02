import json
import numpy as np
import trimesh, torch, nvdiffrast.torch as dr, h5py
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from pathlib import Path
inputs=Path('/work/inputs'); result=Path('/work/outputs/object_poses_world.npz')
ep=json.load(open(inputs/'episode.json')); cam=json.load(open(inputs/'camera.json')); rec=np.load(inputs/'reconstruction.npz'); Tout=np.load(result)['T_world_mesh']; Tbase=rec['T_world_mesh'].astype(float)
with h5py.File(inputs/'masks.h5','r') as h: om=h['object'][:]; pm=h['person'][:]
mesh=trimesh.load(inputs/'object_mesh.glb',force='scene',process=False).to_geometry();v=mesh.vertices
V=torch.tensor(v,dtype=torch.float32,device='cuda');Fa=torch.tensor(mesh.faces,dtype=torch.int32,device='cuda');ctx=dr.RasterizeCudaContext();Tcw=np.linalg.inv(np.array(cam['T_world_cam']));H,W=1152,1536
phases=[('P0 untouched/static',0,190),('P1 approach+turn',191,339),('P2 edge-on flip',340,379),('P3 settle',380,399),('P4 resting/static',400,545)]
def ious(P,K):
 fx,fy=K[0,0],K[1,1];cx,cy=K[0,2],K[1,2];out=[]
 for i in range(len(P)):
  M=Tcw@P[i];vv=V@torch.tensor(M[:3,:3],dtype=torch.float32,device='cuda').T+torch.tensor(M[:3,3],dtype=torch.float32,device='cuda');x,y,z=vv.unbind(1);X=(2*fx/W)*x+(2*cx/W-1)*z;Y=(2*fy/H)*y+(1-2*cy/H)*z;clip=torch.stack([X,Y,torch.zeros_like(z),z],1)[None];rast=dr.rasterize(ctx,clip,Fa,[H,W])[0];mk=(rast[...,3]>0).reshape(H,W).cpu().numpy();vis=mk&~pm[i];out.append((vis&om[i]).sum()/max(1,(vis|om[i]).sum()))
 return np.array(out)
ratings={}
for kn in ['K','K_pipeline']:
 K=np.array(cam[kn]);a=ious(Tbase,K);b=ious(Tout,K);ratings[kn]=(a,b)
# contact: hand keypoints vs sampled object surface
kp=rec['human_keypoints_world'];vs=v[::20];clear={}
for name,P in [('baseline',Tbase),('result',Tout)]:
 ds=[]
 for i in range(100,451,5):
  vw=vs@P[i,:3,:3].T+P[i,:3,3];tree=cKDTree(vw);hands=np.concatenate([kp[i,21:42],kp[i,42:63]]);ds.append(tree.query(hands)[0].min())
 clear[name]=np.array(ds)
# temporal dynamics
dyn={}
for name,P in [('baseline',Tbase),('result',Tout)]:
 d1=np.degrees(np.linalg.norm((Rot.from_matrix(P[1:,:3,:3])*Rot.from_matrix(P[:-1,:3,:3]).inv()).as_rotvec(),axis=1));t1=np.linalg.norm(P[1:,:3,3]-P[:-1,:3,3],axis=1)
 move=np.arange(190,400)
 st=np.concatenate([np.arange(0,191),np.arange(400,546)])
 statrot=np.concatenate([np.degrees(np.linalg.norm((Rot.from_matrix(P[0:191,:3,:3])*Rot.from_matrix(P[0,:3,:3]).inv()).as_rotvec(),axis=1)),np.degrees(np.linalg.norm((Rot.from_matrix(P[400:,:3,:3])*Rot.from_matrix(P[400,:3,:3]).inv()).as_rotvec(),axis=1))])
 dyn[name]={'median_dR_deg':float(np.median(d1)),'p95_dR_deg':float(np.percentile(d1,95)),'max_dR_deg':float(d1.max()),'p95_dt_m':float(np.percentile(t1,95)),'max_static_rot_dev_deg':float(statrot.max()),'static_translation_span_m':float(np.linalg.norm(np.ptp(np.concatenate([P[:191,:3,3],P[400:,:3,3]]),axis=0)))}
# support heights
n=np.array(json.load(open(inputs/'ground_plane.json'))['plane'][:3]);n/=np.linalg.norm(n)
def support(P,i):return float(np.percentile((v@P[i,:3,:3].T+P[i,:3,3])@n,0.1))
report={'phase_iou_K':{},'phase_iou_Kpipeline':{},'contact':{},'dynamics':dyn,'support':{}}
for p,a,b in phases:
 report['phase_iou_K'][p]=[float(ratings['K'][0][a:b+1].mean()),float(ratings['K'][1][a:b+1].mean())]
 report['phase_iou_Kpipeline'][p]=[float(ratings['K_pipeline'][0][a:b+1].mean()),float(ratings['K_pipeline'][1][a:b+1].mean())]
report['overall_iou_K']=[float(ratings['K'][0].mean()),float(ratings['K'][1].mean())]
report['overall_iou_Kpipeline']=[float(ratings['K_pipeline'][0].mean()),float(ratings['K_pipeline'][1].mean())]
for name,d in clear.items():report['contact'][name]={'median_min_hand_distance_m':float(np.median(d)),'p90_m':float(np.percentile(d,90)),'fraction_under_5cm':float(np.mean(d<.05))}
for name,P in [('baseline',Tbase),('result',Tout)]:report['support'][name]={'initial_m':support(P,100),'final_m':support(P,500),'difference_m':support(P,500)-support(P,100)}
print(json.dumps(report,indent=2))
