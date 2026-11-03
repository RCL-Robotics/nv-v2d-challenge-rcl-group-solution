from pathlib import Path
import numpy as np,torch,cv2,json,time
from scipy.spatial.transform import Rotation
from refine import Renderer,make_data,to_torch,optimize,losses

def run(ids,save='multi_results.npz',steps=120):
 torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4);cv2.setNumThreads(1);inputs=Path('/work/inputs');out=Path('/work/outputs');c=json.load(open(inputs/'camera.json'));K=np.array(c['K']);a=np.load(inputs/'reconstruction.npz');T=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];data=make_data(inputs,ids,K,T);d=to_torch(data);r=Renderer(inputs/'object_mesh.glb',K);Rall=[];tall=[];scores=[];stats=[]
 start=time.time()
 for j,f in enumerate(ids):
  # Reuse the two released rotations and nearby rotations as independent starts.
  seeds=[T[f,:3,:3],a['T_cam_mesh_foundationpose'][f,:3,:3]]
  for delta in [-20,-10,-5,5,10,20]:seeds.append(T[np.clip(f+delta,0,len(T)-1),:3,:3])
  # The asymmetric blue latch requires explicit axial-roll alternatives.
  seeds.extend([T[f,:3,:3]@Rotation.from_euler('z',z,degrees=True).as_matrix() for z in [-90,90,180]])
  # Avoid an erroneous cap-end inversion in FoundationPose after hand occlusion.
  seeds.extend([a['T_cam_mesh_foundationpose'][f,:3,:3]@Rotation.from_euler('x',180,degrees=True).as_matrix()])
  seeds=np.array(seeds);n=len(seeds);dd={k:v[j:j+1].repeat(n,*([1]*(v.ndim-1))) for k,v in d.items()};t0=np.repeat(T[f,:3,3][None],n,0)
  RR,tt,ss,st=optimize(r,seeds,t0,dd,steps=steps,texture=True,rot_reg=.012,shift_reg=.004);
  # Compare final candidates with one common baseline-centered regularizer.
  w=Rotation.from_matrix(RR@T[f,:3,:3].T).magnitude();ss0=[]
  with torch.no_grad():l,s,m=losses(r,RR,tt,dd,texture=True)
  cost=l.cpu().numpy()+.018*np.minimum(w*w,2)
  Rall.append(RR);tall.append(tt);scores.append(cost);stats.append(s.cpu().numpy());print('frame',f,'best',np.argmin(cost),'score',np.round(cost.min(),3),'iou',round(s[np.argmin(cost),0].item(),3),'angles',np.round(np.degrees(w[np.argsort(cost)[:3]]),1),flush=True)
 np.savez_compressed(out/save,R=np.array(Rall),t=np.array(tall),scores=np.array(scores),stats=np.array(stats),**data);print('seconds',time.time()-start)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--ids',default='240,250,400,550,600,650,660,670,680,690,700,710');p.add_argument('--save',default='multi_results.npz');p.add_argument('--steps',type=int,default=120);a=p.parse_args();run(np.array(list(map(int,a.ids.split(',')))),a.save,a.steps)
