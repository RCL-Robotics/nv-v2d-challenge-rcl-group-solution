import cv2,h5py,trimesh,numpy as np,json,time
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

def refine_cpu(inputs, initial, tracks, start=0):
 p=Path(inputs);camera=json.load(open(p/'camera.json'));K=np.array(camera['K']);base=np.linalg.inv(camera['T_world_cam'])@np.load(p/'reconstruction.npz')['T_world_mesh'];mesh=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry()
 # Fixed area-stratified surface samples, no stochastic sampling.
 area=np.asarray(mesh.area_faces);cdf=np.cumsum(area);ids=np.searchsorted(cdf,(np.arange(2600)+.5)/2600*cdf[-1]);tris=np.asarray(mesh.triangles)[ids]
 frac=(np.arange(len(ids))*.6180339887498949)%1;frac2=(np.arange(len(ids))*.4142135623730951)%1;sq=np.sqrt(frac);vertices=(1-sq[:,None])*tris[:,0]+(sq*(1-frac2))[:,None]*tris[:,1]+(sq*frac2)[:,None]*tris[:,2]
 points=np.array([[-.00386614,-.0140458,-.44275588],[.00168146,.01075304,.99291701]])
 masks=h5py.File(p/'masks.h5');Q=initial.copy();timer=time.time();params=[];losses=[]
 for i in range(start,len(base)):
  mask=masks['object'][i].astype('uint8');person=masks['person'][i].astype('uint8');yy,xx=np.where(mask);minx=max(0,xx.min()-120);maxx=min(1536,xx.max()+121);miny=max(0,yy.min()-120);maxy=min(1152,yy.max()+121)
  om=mask[miny:maxy,minx:maxx];pm=person[miny:maxy,minx:maxx];allowed=om|pm;dt=cv2.distanceTransform((1-allowed).astype('uint8'),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
  # Deterministic stratification gives each observed segment support.
  pick=np.linspace(0,len(xx)-1,min(240,len(xx))).astype(int);target=np.c_[xx[pick],yy[pick]].astype(float)
  obs=np.stack([tracks['head'][i],tracks['tip'][i]]);valid=np.isfinite(obs[:,0]);dist=np.linalg.norm(base[i,:3,3]);R0=initial[i,:3,:3];t0=initial[i,:3,3];uv0=t0[:2]/t0[2]
  def pose(x):
   R=Rotation.from_rotvec(x[:3]).as_matrix()@R0;ray=np.r_[uv0+x[3:]/K[0,0],1.];t=ray/np.linalg.norm(ray)*dist;return R,t
  def project(v,R,t):
   xyz=v@R.T+t;uv=xyz@K.T;return uv[:,:2]/uv[:,2:]
  def objective(x):
   R,t=pose(x);uv=project(vertices,R,t);xy=(uv-[minx,miny]).astype(np.float32)
   outside=cv2.remap(dt,xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=70).ravel()
   # Beyond a 1.5px tolerance, penalize model pixels outside all allowed
   # support; person regions permit genuine object occlusion.
   a=np.maximum(outside-1.,0)/np.sqrt(len(outside))
   distances=cKDTree(uv).query(target)[0];b=np.maximum(distances-2.,0)/np.sqrt(len(target))
   puv=project(points,R,t);land=(puv[valid]-obs[valid]).ravel()*.38
   # Strong orientation trust region before failure, lighter late.
   reg=x[:3]*(10 if i<660 else 6);transreg=x[3:]*.012
   return np.r_[a,b,land,reg,transreg]
  # Differencing must exceed cv2's 1/32px interpolation table spacing.
  # Central finite differences are explicit, avoiding quantization traps.
  def jac(x):
   columns=[]
   for j in range(5):
    step=.0004 if j<3 else .08;e=np.zeros(5);e[j]=step;columns.append((objective(x+e)-objective(x-e))/(2*step))
   return np.stack(columns,axis=1)
  result=least_squares(objective,np.zeros(5),jac=jac,loss='soft_l1',f_scale=2.,max_nfev=20,xtol=3e-5,ftol=3e-5,gtol=3e-5,bounds=([-0.3]*3+[-60]*2,[.3]*3+[60]*2))
  Q[i,:3,:3],Q[i,:3,3]=pose(result.x);params.append(result.x);losses.append(result.cost)
  if i%100==0:print('CPU fit',i,'seconds',round(time.time()-timer,1),flush=True)
 # Smooth only residual corrections relative to the original trajectory.
 # Baseline camera distance remains exact after filtering the image ray.
 pars=gaussian_filter1d(np.array(params),.8,axis=0)
 for j,i in enumerate(range(start,len(base))):
  R0=initial[i,:3,:3];t0=initial[i,:3,3];x=pars[j];Q[i,:3,:3]=Rotation.from_rotvec(x[:3]).as_matrix()@R0;ray=np.r_[t0[:2]/t0[2]+x[3:]/K[0,0],1.];Q[i,:3,3]=ray/np.linalg.norm(ray)*np.linalg.norm(base[i,:3,3])
 masks.close();return Q,losses,time.time()-timer

if __name__=='__main__':
 a=np.load('/work/outputs/feature_tracks_full.npz');tracks={k:a[k] for k in a.files};initial=np.load('/work/outputs/endpoint_initial.npz')['T_cam_mesh'];q,loss,secs=refine_cpu('/work/inputs',initial,tracks);np.savez('/work/outputs/cpu_candidate.npz',T_cam_mesh=q);print('DONE',secs)
