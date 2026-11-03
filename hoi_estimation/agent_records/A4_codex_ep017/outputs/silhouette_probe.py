import numpy as np,cv2,json,h5py,trimesh,time
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
root='/work/inputs'; r=np.load(root+'/reconstruction.npz');cam=json.load(open(root+'/camera.json'));K=np.array(cam['K']);C=np.linalg.inv(cam['T_world_cam']);T=C@r['T_world_mesh'];m=trimesh.load(root+'/object_mesh.glb',force='scene',process=False).to_geometry().convex_hull;v=m.vertices
h=h5py.File(root+'/masks.h5');results=[];start=time.time()
for f in range(len(T)):
 obj=h['object'][f];ys,xs=np.where(obj)
 if len(xs)<1000:results.append([f,0,0,0,0,0]);continue
 p=v@T[f,:3,:3].T+T[f,:3,3]; pp=p@K.T;pp=pp[:,:2]/pp[:,2:];x0=max(0,min(xs.min(),int(pp[:,0].min()))-12);x1=min(1536,max(xs.max(),int(pp[:,0].max()))+13);y0=max(0,min(ys.min(),int(pp[:,1].min()))-12);y1=min(1152,max(ys.max(),int(pp[:,1].max()))+13)
 obs=obj[y0:y1,x0:x1];person=h['person'][f,y0:y1,x0:x1];occ=cv2.dilate((person&~obs).astype('uint8'),np.ones((3,3),np.uint8))>0;occ[obs]=False
 best=(-1,0,0,0);base=0
 for angle in np.arange(-45,45.01,3):
  R=T[f,:3,:3]@Rotation.from_rotvec([0,0,np.deg2rad(angle)]).as_matrix();p=v@R.T+T[f,:3,3];pp=p@K.T;pp=pp[:,:2]/pp[:,2:]-[x0,y0];hull=cv2.convexHull(pp.astype('float32'))
  for dx in [-2,0,2]:
   for dy in [-2,0,2]:
    pred=np.zeros(obs.shape,np.uint8);cv2.fillConvexPoly(pred,np.round(hull+[dx,dy]).astype('int32'),1);pred=pred.astype(bool)&~occ
    iou=(pred&obs).sum()/max(1,(pred|obs).sum())
    if angle==0 and dx==dy==0:base=iou
    if iou>best[0]:best=(iou,angle,dx,dy)
 results.append([f,base,*best]);
 if f%50==0:print(f,results[-1],round(time.time()-start,1),flush=True)
np.save('/work/outputs/silhouette_probe.npy',results)
print('done',time.time()-start)
