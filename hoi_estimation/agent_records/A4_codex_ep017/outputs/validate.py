import os,json,cv2,numpy as np,h5py,torch,trimesh
from scipy.spatial.transform import Rotation
from solve import Silhouette
root='/work/inputs';cam=json.load(open(root+'/camera.json'));K=np.array(cam['K']);C=np.linalg.inv(cam['T_world_cam']);b=np.load(root+'/reconstruction.npz')['T_world_mesh'].astype(float);o=np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];T=C@b;U=C@o;h=h5py.File(root+'/masks.h5');m=trimesh.load(root+'/object_mesh.glb',force='scene',process=False).to_geometry();ren=Silhouette(m);results=[]
for f in range(len(T)):
 obs=h['object'][f]
 if obs.sum()<1500:continue
 p=K@T[f,:3,3];xy=np.clip(np.round(p[:2]/p[2]-128).astype(int),[0,0],[1280,896]);x,y=xy;obj=obs[y:y+256,x:x+256];person=h['person'][f,y:y+256,x:x+256];occ=cv2.dilate((person&~obj).astype('uint8'),np.ones((3,3),np.uint8))>0;occ[obj]=False
 with torch.no_grad():
  arr=np.stack([T[f],U[f]]);mask=ren.render(torch.tensor(arr[:,:3,:3],device='cuda',dtype=torch.float32),torch.tensor(arr[:,:3,3],device='cuda',dtype=torch.float32),K,torch.tensor(np.stack([xy,xy]),device='cuda',dtype=torch.float32)).cpu().numpy()
 iou=[float((p*obj).sum()/((p+obj-p*obj)*(~occ)).sum()) for p in mask];results.append([f,*iou])
results=np.array(results);np.save('/work/outputs/validation_iou.npy',results)
print('mean',results[:,1:].mean(axis=0),'median',np.median(results[:,1:],axis=0),'worse >.03',results[results[:,2]<results[:,1]-.03,0].tolist())
print('static beginning',results[results[:,0]<117,1:].mean(axis=0),'static end',results[results[:,0]>=340,1:].mean(axis=0))
for a,z in [(0,117),(117,190),(190,250),(250,290),(290,340),(340,440)]:
 idx=(np.arange(440)>=a)&(np.arange(440)<z);d=(Rotation.from_matrix(T[idx,:3,:3]).inv()*Rotation.from_matrix(U[idx,:3,:3])).magnitude()*180/np.pi
 print(a,z,'rotchange mean',d.mean(),'transchange mean cm',np.mean(np.linalg.norm(T[idx,:3,3]-U[idx,:3,3],axis=1))*100)
d=Rotation.from_matrix(U[:,:3,:3]);print('max rotation jump',np.max((d[:-1].inv()*d[1:]).magnitude())*180/np.pi)
print('distance error',np.max(abs(np.linalg.norm(U[:,:3,3],axis=1)-np.linalg.norm(T[:,:3,3],axis=1))))
