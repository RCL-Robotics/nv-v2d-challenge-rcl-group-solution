import numpy as np,torch,time
from scipy.spatial.transform import Rotation
from deterministic_search import search
from refine import to_torch,losses
from blue_metrics import compute_blue

def add_roll_candidates(c,renderer):
 R=c['R'];t=c['t'];F,N=R.shape[:2];blue=compute_blue(c,renderer);c['blue_iou']=blue;count=c['blue'].sum((1,2));
 ids=np.where((count>80)&(blue[:,0]<.20)&(c['stats'][:,:,0].max(1)>.70))[0];Nadd=8
 RR=np.repeat(R[:,0,None],Nadd,axis=1);tt=np.repeat(t[:,0,None],Nadd,axis=1);ss=np.repeat(c['scores'][:,0,None],Nadd,axis=1);st=np.repeat(c['stats'][:,0,None],Nadd,axis=1);d=to_torch(c);start=time.time();print('axial search frames',len(ids),flush=True)
 for j in range(0,len(ids),4):
  ii=ids[j:j+4];n=len(ii);best=np.argmax(c['stats'][ii,:,0]-.2*c['stats'][ii,:,2],axis=1);seed=R[ii,best];ts=t[ii,best]
  delta=Rotation.from_euler('z',np.arange(8)*45,degrees=True).as_matrix();R0=seed[:,None]@delta[None];t0=np.repeat(ts[:,None],Nadd,axis=1);dd={k:v[ii].repeat_interleave(Nadd,0) for k,v in d.items()}
  r1,t1,s1,st1=search(renderer,R0.reshape(-1,3,3),t0.reshape(-1,3),dd,rot_reg=.012,shift_reg=.004,rounds=2)
  with torch.no_grad():l,stats,_=losses(renderer,r1,t1,dd,texture=True)
  RR[ii]=r1.reshape(n,Nadd,3,3);tt[ii]=t1.reshape(n,Nadd,3);ss[ii]=l.cpu().numpy().reshape(n,Nadd);st[ii]=stats.cpu().numpy().reshape(n,Nadd,4)
 for name,val in [('R',RR),('t',tt),('scores',ss),('stats',st)]:c[name]=np.concatenate([c[name],val],axis=1)
 c['blue_iou']=compute_blue(c,renderer);print('axial search seconds',time.time()-start,flush=True);return c

if __name__=='__main__':
 import json,cv2
 from render_utils import Renderer
 torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(0);np.random.seed(0)
 a=np.load('/work/outputs/det_candidates.npz');c={k:a[k] for k in a.files};K=np.array(json.load(open('/work/inputs/camera.json'))['K']);r=Renderer('/work/inputs/object_mesh.glb',K);c=add_roll_candidates(c,r);np.savez_compressed('/work/outputs/final_candidates.npz',**c)
