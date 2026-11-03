import os,time,json,argparse,random
import numpy as np,cv2,h5py,torch
from scipy.spatial.transform import Rotation
from render_utils import Renderer

def data(inputs,step=1):
 cam=json.load(open(inputs+'/camera.json'));K=np.array(cam['K']);W=np.array(cam['T_world_cam']);base=np.load(inputs+'/reconstruction.npz')['T_world_mesh'].astype(np.float64);T=np.linalg.inv(W)@base;F=len(T);S=128
 uv=T[:,:3,3]@K.T;uv=uv[:,:2]/uv[:,2:];orig=np.rint(uv-S/2).astype(int)
 images=[];masks=[];people=[]
 c=cv2.VideoCapture(inputs+'/video.mp4')
 with h5py.File(inputs+'/masks.h5') as h:
  for f in range(F):
   ok,im=c.read()
   if f%step:continue
   x,y=orig[f];images.append(im[y:y+S,x:x+S,::-1].copy()/255.);masks.append(h['object'][f,y:y+S,x:x+S]);people.append(h['person'][f,y:y+S,x:x+S])
 return cam,base,T,orig[::step],np.array(images),np.array(masks),np.array(people)

def rot6(x):
 a=torch.nn.functional.normalize(x[...,:3],dim=-1);b=x[...,3:];b=torch.nn.functional.normalize(b-a*(a*b).sum(-1,keepdim=True),dim=-1);return torch.stack([a,b,torch.cross(a,b,dim=-1)],-1)
def six(R):return np.concatenate([R[...,0],R[...,1]],-1)
def translation(x,t0,K):
 # Conserve the Euclidean camera-to-mesh-origin distance exactly.
 u=t0[:,0]/t0[:,2]+x[:,0]/K[0,0];v=t0[:,1]/t0[:,2]+x[:,1]/K[1,1];ray=torch.stack([u,v,torch.ones_like(u)],-1);return torch.nn.functional.normalize(ray,dim=-1)*t0.norm(dim=-1,keepdim=True)

def loss_fn(rgb,alpha,img,mask,person,dist):
 a=alpha[...,0];area=mask.sum((1,2)).clamp(min=1);visible=(1-person)+mask;visible=visible.clamp(max=1)
 sil=((a-mask).square()*visible).sum((1,2))/area
 outside=(a*dist*visible).sum((1,2))/area
 # Fit an exposure gain and offset, then compare the longitudinal silver/black pattern.
 weights=(a*mask).detach();den=weights.sum((1,2)).clamp(min=1);x=rgb.sum(-1)/3;y=img.sum(-1)/3
 mx=(x*weights).sum((1,2))/den;my=(y*weights).sum((1,2))/den
 dx=x-mx[:,None,None];dy=y-my[:,None,None]
 gain=((dx*dy*weights).sum((1,2))/((dx*dx*weights).sum((1,2))+1e-5)).clamp(.3,3).detach()
 err=(gain[:,None,None]*dx-dy).square();photo=(err*weights).sum((1,2))/den
 return sil+0.12*outside+12*photo,torch.stack([sil,outside,photo],-1)

def fit(inputs,out,step=4,iters=45):
 torch.manual_seed(0);np.random.seed(0);random.seed(0);torch.set_num_threads(4)
 start=time.time();cam,base,T,orig,imgs,masks,people=data(inputs,step);idx=np.arange(0,len(T),step);K=np.array(cam['K']);ren=Renderer(inputs+'/object_mesh.glb',K);D='cuda'
 dist=np.array([np.minimum(cv2.distanceTransform((~m).astype('uint8'),cv2.DIST_L2,5),12)**2 for m in masks]);frames=[];scores=[];components=[];trans=[];scales=[]
 # Cover both polarities of the long axis and all handle azimuths. No axial symmetry assumption.
 local=np.array([Rotation.from_euler('x',flip,degrees=True).as_matrix()@Rotation.from_euler('z',angle,degrees=True).as_matrix() for flip in [0,180] for angle in range(0,360,30)])
 # A mug never inverts in this clip. The visible silhouette gives its long-axis
 # direction in the image; a 9-frame median suppresses isolated depth-axis flips
 # without estimating depth from object size. These are initializations only.
 from scipy.ndimage import median_filter
 zseed=median_filter(T[:,2,2],size=9,mode='nearest')
 initialR=T[:,:3,:3].copy()
 for n,f in enumerate(idx):
  yy,xx=np.where(masks[n]);points=np.stack([xx,yy],-1).astype(float)
  _,vectors=np.linalg.eigh(np.cov(points.T));axis=vectors[:,-1]
  if axis[1]<0:axis=-axis
  z=np.clip(zseed[f],-.85,.6);desired=np.r_[axis*np.sqrt(1-z*z),z]
  old=initialR[f,:,2];cross=np.cross(old,desired);sin=np.linalg.norm(cross);cos=np.dot(old,desired)
  if sin>1e-8: initialR[f]=Rotation.from_rotvec(cross/sin*np.arctan2(sin,cos)).as_matrix()@initialR[f]
 B=24
 for n,f in enumerate(idx):
  init=initialR[f][None]@local;rr=torch.tensor(six(init),device=D,dtype=torch.float32,requires_grad=True);xy=torch.zeros((B,2),device=D,requires_grad=True);scale=torch.full((B,),1.08,device=D,requires_grad=True);t0=torch.tensor(T[f,:3,3],device=D,dtype=torch.float32)[None].expand(B,-1);o=torch.tensor(orig[n],device=D,dtype=torch.float32)[None].expand(B,-1)
  im=torch.tensor(imgs[n],device=D,dtype=torch.float32)[None].expand(B,-1,-1,-1);m=torch.tensor(masks[n],device=D,dtype=torch.float32)[None].expand(B,-1,-1);p=torch.tensor(people[n],device=D,dtype=torch.float32)[None].expand(B,-1,-1);dt=torch.tensor(dist[n],device=D,dtype=torch.float32)[None].expand(B,-1,-1)
  opt=torch.optim.Adam([{'params':[rr],'lr':.025},{'params':[xy],'lr':.20},{'params':[scale],'lr':.005}])
  for it in range(iters):
   R=rot6(rr);t=translation(xy,t0,K);rgb,a=ren.render(R,t,o,scale=scale.clamp(.95,1.18));loss,co=loss_fn(rgb,a,im,m,p,dt)
   # Small image-plane corrections, no depth or scale fitting.
   total=loss+.0008*xy.square().sum(-1)+2*(scale-1.08).square();opt.zero_grad();total.sum().backward();opt.step()
  with torch.no_grad():
   R=rot6(rr);t=translation(xy,t0,K);rgb,a=ren.render(R,t,o,scale=scale.clamp(.95,1.18));loss,co=loss_fn(rgb,a,im,m,p,dt)
  scales.append(scale.detach().cpu().numpy());frames.append(R.detach().cpu().numpy());trans.append(t.detach().cpu().numpy());scores.append(loss.detach().cpu().numpy());components.append(co.detach().cpu().numpy())
  if n%10==0:print(n,len(idx),'frame',f,'sec',round(time.time()-start,1),'best',np.round(np.sort(scores[-1])[:3],3),flush=True)
 np.savez(out,idx=idx,scale=np.array(scales),R=np.array(frames),t=np.array(trans),loss=np.array(scores),components=np.array(components));print('done',time.time()-start,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inputs',default='/work/inputs');p.add_argument('--out',default='/work/outputs/candidates.npz');p.add_argument('--step',type=int,default=4);p.add_argument('--iters',type=int,default=45);a=p.parse_args();fit(a.inputs,a.out,a.step,a.iters)
