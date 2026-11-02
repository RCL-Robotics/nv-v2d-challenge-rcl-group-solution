import numpy as np,cv2,h5py,trimesh,json
p='/work/inputs/';a=np.load(p+'reconstruction.npz');c=json.load(open(p+'camera.json'));K=np.array(c['K']);T=np.linalg.inv(c['T_world_cam'])[None]@a['T_world_mesh'];m=trimesh.load(p+'object_mesh.glb',force='scene',process=False).to_geometry();pts=np.array(m.vertices)[::12];h=h5py.File(p+'masks.h5');v=cv2.VideoCapture(p+'video.mp4'); tiles=[]
for i in [0,105,210,245,315,385,490,560,630,700,770,805,840,885]:
 v.set(1,i);_,im=v.read();ms=h['object'][i];yy,xx=np.where(ms);x0=max(0,xx.min()-50);x1=min(1536,xx.max()+51);y0=max(0,yy.min()-50);y1=min(1152,yy.max()+51);res=[]
 for name,A in [('BASE',T),('FP',a['T_cam_mesh_foundationpose']),('COCO',a['T_cam_mesh_coconet'])]:
  R=A[i,:3,:3];t=A[i,:3,3];t=t/np.linalg.norm(t)*np.linalg.norm(T[i,:3,3]);q=pts@R.T+t;q=q@K.T;uv=q[:,:2]/q[:,2:];ov=im.copy();ov[ms]=(.7*im[ms]+.3*np.array([0,255,0])).astype('uint8');u=uv.astype(int);g=(u[:,0]>=0)&(u[:,0]<1536)&(u[:,1]>=0)&(u[:,1]<1152);u=u[g];ov[u[:,1],u[:,0]]=[255,0,255];crop=ov[y0:y1,x0:x1];s=min(280/crop.shape[1],320/crop.shape[0]);crop=cv2.resize(crop,None,fx=s,fy=s);canvas=np.full((350,290,3),235,np.uint8);canvas[25:25+crop.shape[0],5:5+crop.shape[1]]=crop;cv2.putText(canvas,f'{i} {name}',(5,20),0,.6,(0,0,0),1);res.append(canvas)
 tiles.append(np.concatenate(res,1))
for j in range(0,len(tiles),4):cv2.imwrite('/work/outputs/candidates_'+str(j//4)+'.jpg',np.concatenate(tiles[j:j+4],0))
