import os,time,json,cv2,h5py,numpy as np,torch
from scipy.optimize import minimize
from render_utils import Renderer
np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(4)
t0=time.perf_counter();p='/work/inputs/';K=np.array(json.load(open(p+'camera.json'))['K']);C=np.array(json.load(open(p+'camera.json'))['T_world_cam']);poses=np.linalg.inv(C)@np.load(p+'reconstruction.npz')['T_world_mesh'];ren=Renderer(p+'object_mesh.glb');cap=cv2.VideoCapture(p+'video.mp4')
n=len(poses);side=160
centres=(poses[:,:3,3]@K.T);centres=centres[:,:2]/centres[:,2:];origins=np.floor(centres-side/2).astype(int);Ks=np.repeat(K[None],n,0);Ks[:,:2,2]-=origins
renders=[]
with torch.no_grad():
 for a in range(0,n,24):
  m=ren.render(poses[a:a+24,:3,:3],poses[a:a+24,:3,3],Ks[a:a+24],(side,side),False)
  renders.append(m.cpu().numpy())
renders=np.concatenate(renders)
shifts=[];scores=[];ious=[];coverage=[];areas=[];rawmasks=[];valids=[];rgbcrops=[]
with h5py.File(p+'masks.h5') as f:
 for i in range(n):
  ok,bgr=cap.read();assert ok;x,y=origins[i];rgbcrops.append(cv2.cvtColor(bgr[y:y+side,x:x+side],cv2.COLOR_BGR2RGB))
  obs=f['object'][i,y:y+side,x:x+side].astype(np.uint8);human=f['person'][i,y:y+side,x:x+side].astype(np.uint8)
  # Bowl is convex in silhouette. Filling interior cutouts prevents the chair leg from attracting the fit.
  ys,xs=np.where(obs);hull=np.zeros_like(obs)
  if len(xs):cv2.fillConvexPoly(hull,cv2.convexHull(np.stack([xs,ys],1)),1)
  valid=1-cv2.dilate(human,np.ones((7,7),np.uint8))
  target=hull.astype(np.float32);valid=valid.astype(np.float32)
  m=renders[i];areas.append([obs.sum(),m.sum()]);coverage.append(float((m*valid).sum()/max(m.sum(),1)))
  rawmasks.append(obs);valids.append(valid)
  def score(v):
   s=cv2.warpAffine(m,np.array([[1,0,v[0]],[0,1,v[1]]],np.float32),(side,side),flags=cv2.INTER_LINEAR)
   return float(((s-target)**2*valid).sum()/max((target*valid).sum(),1))+.00005*float(np.dot(v,v))
  fit=minimize(score,[0,0],method='Powell',bounds=[(-6,6),(-6,6)],options={'xtol':.03,'ftol':1e-6,'maxiter':30})
  shifts.append(fit.x);scores.append([score([0,0]),score(fit.x)]);
  s=cv2.warpAffine(m,np.array([[1,0,fit.x[0]],[0,1,fit.x[1]]],np.float32),(side,side),flags=cv2.INTER_LINEAR)
  def iou(a,b):return ((a>.5)&(b>.5)&(valid>0)).sum()/max(((a>.5)|(b>.5))[valid>0].sum(),1)
  ious.append([iou(m,target),iou(s,target)])
np.savez_compressed('/work/outputs/diagnostics/alignment.npz',shifts=shifts,scores=scores,ious=ious,coverage=coverage,areas=areas,render=renders,object=rawmasks,valid=valids,rgb=rgbcrops,origins=origins,Ks=Ks)
shifts=np.array(shifts);scores=np.array(scores);ious=np.array(ious);coverage=np.array(coverage)
for a,b in [(0,150),(150,220),(220,490),(490,620),(620,790),(790,870)]:
 print(a,b,'shift median',np.median(shifts[a:b],axis=0),'mean score',scores[a:b].mean(0),'iou',ious[a:b].mean(0),'coverage',coverage[a:b].mean(),flush=True)
print('elapsed',time.perf_counter()-t0)
