from pathlib import Path
import numpy as np,torch,cv2,json,time
from scipy.spatial.transform import Rotation
from refine import Renderer,make_data,to_torch,losses; from deterministic_search import search as optimize

def generate(inputs,save,steps=85):
 torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4);cv2.setNumThreads(1);c=json.load(open(inputs/'camera.json'));K=np.array(c['K']);a=np.load(inputs/'reconstruction.npz');T=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];ids=np.arange(len(T));data=make_data(inputs,ids,K,T);r=Renderer(inputs/'object_mesh.glb',K);d=to_torch(data);F=len(T);N=9
 Rall=np.zeros((F,N,3,3));tall=np.zeros((F,N,3));score=np.zeros((F,N));stats=np.zeros((F,N,4));Rall[:,0]=T[:,:3,:3];tall[:,0]=T[:,:3,3];start=time.time()
 for j in range(0,F,4):
  ii=ids[j:j+4];n=len(ii);R0=np.stack([T[ii,:3,:3],a['T_cam_mesh_foundationpose'][ii,:3,:3],T[np.clip(ii-10,0,F-1),:3,:3],T[np.clip(ii+10,0,F-1),:3,:3],T[ii,:3,:3]@Rotation.from_euler('z',90,degrees=True).as_matrix(),T[ii,:3,:3]@Rotation.from_euler('z',-90,degrees=True).as_matrix(),T[ii,:3,:3]@Rotation.from_euler('z',180,degrees=True).as_matrix(),a['T_cam_mesh_foundationpose'][ii,:3,:3]@Rotation.from_euler('x',180,degrees=True).as_matrix()],axis=1)
  t0=np.repeat(T[ii,None,:3,3],N-1,axis=1);dd={k:v[j:j+n].repeat_interleave(N-1,0) for k,v in d.items()}
  RR,tt,ss,st=optimize(r,R0.reshape(-1,3,3),t0.reshape(-1,3),dd,rot_reg=.012,shift_reg=.004)
  Rall[ii,1:]=RR.reshape(n,N-1,3,3);tall[ii,1:]=tt.reshape(n,N-1,3);
  dd={k:v[j:j+n].repeat_interleave(N,0) for k,v in d.items()}
  with torch.no_grad():l,s,_=losses(r,Rall[ii].reshape(-1,3,3),tall[ii].reshape(-1,3),dd,texture=True)
  score[ii]=l.cpu().numpy().reshape(n,N);stats[ii]=s.cpu().numpy().reshape(n,N,4)
  if j%40==0:print('candidates',j,'/',F,'seconds',round(time.time()-start,1),flush=True)
 np.savez_compressed(save,R=Rall,t=tall,scores=score,stats=stats,**data);print('done',time.time()-start,flush=True)

if __name__=='__main__':generate(Path('/work/inputs'),'/work/outputs/det_candidates.npz')
