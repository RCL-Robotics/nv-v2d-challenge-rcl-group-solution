import os
os.environ['PYOPENGL_PLATFORM']='egl'
import trimesh,pyrender,cv2,numpy as np,json,h5py,time
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
root='/work/inputs/';mesh=trimesh.load(root+'object_mesh.glb',force='scene',process=False).to_geometry();cam=json.load(open(root+'camera.json'));K=np.array(cam['K']);wc=np.array(cam['T_world_cam']);poses=np.linalg.inv(wc)@np.load(root+'reconstruction.npz')['T_world_mesh']
scene=pyrender.Scene(ambient_light=[1,1,1],bg_color=[0,0,0,0]);node=scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False));camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],128,128,znear=.01,zfar=20);cn=scene.add(camera,pose=np.diag([1,-1,-1,1]));renderer=pyrender.OffscreenRenderer(256,256); masks=h5py.File(root+'masks.h5','r');res=[];t=time.time()
baseline=np.load('/work/scratch/alignment.npz')['metrics']
for f,T0 in enumerate(poses):
 if not (150<f<440 and baseline[f,3]>.8 and baseline[f,4]>.55): continue
 lo=max(0,f-4);hi=min(len(poses),f+5);rots=Rotation.from_matrix(poses[lo:hi,:3,:3]);R=rots.mean().as_matrix();ang=Rotation.from_matrix(R@T0[:3,:3].T).magnitude()*180/np.pi
 if not (3<ang<20): continue
 if max((rots.inv()*Rotation.from_matrix(T0[:3,:3])).magnitude())>np.pi/3:continue
 T=T0.copy();center=mesh.bounds.mean(0);c0=T[:3,:3]@center+T[:3,3];T[:3,:3]=R;T[:3,3]=c0-R@center
 c=(T[:3,:3]@mesh.bounds.mean(0)+T[:3,3]);q=K@c;q=q[:2]/q[2];x0=int(round(q[0]))-128;y0=int(round(q[1]))-128;camera.cx=K[0,2]-x0;camera.cy=K[1,2]-y0;scene.set_pose(node,T);col,dep=renderer.render(scene,flags=pyrender.RenderFlags.FLAT);pred=(dep>0).astype('uint8');obs=masks['object'][f,y0:y0+256,x0:x0+256].astype('uint8');human=masks['person'][f,y0:y0+256,x0:x0+256].astype('uint8');
 # Ignore person overlap to prevent bias towards the visible fraction.
 valid=1-cv2.dilate(human,np.ones((5,5),'uint8'));valid[obs>0]=1
 dtout=cv2.distanceTransform(1-obs,cv2.DIST_L2,5);dtin=cv2.distanceTransform(obs,cv2.DIST_L2,5);signed=dtout-dtin
 def cost(x):
  p=cv2.warpAffine(pred.astype('float32'),np.array([[1,0,x[0]],[0,1,x[1]]],dtype='float32'),(256,256),flags=cv2.INTER_LINEAR)
  return float(np.sum(p*np.clip(signed,-5,5)*valid)/max(obs.sum(),1))
 opt=minimize(cost,[0,0],method='Powell',bounds=[(-8,8),(-8,8)],options={'xtol':.05,'ftol':.0001});shift=opt.x;visible=(pred*valid).sum()/max(pred.sum(),1);iou=(pred*obs).sum()/max(((pred+obs)>0).sum(),1);res.append([f,ang,*shift,cost([0,0])-opt.fun,visible,iou,obs.sum(),pred.sum(),opt.fun]);print(f,'angle',round(ang,1),'baseiou',round(baseline[f,4],3),'smootiou',round(iou,3),'gain',round(cost([0,0])-opt.fun,3),flush=True);
 if f%100==0:print(f,round(time.time()-t,1),np.round(res[-1][:5],3),flush=True)
np.savez('/work/scratch/orient_alignment.npz',metrics=res);renderer.delete();print('seconds',time.time()-t)
