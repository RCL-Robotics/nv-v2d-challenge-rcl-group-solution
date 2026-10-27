import numpy as np,cv2,h5py,trimesh,json,argparse
p=argparse.ArgumentParser();p.add_argument('file');p.add_argument('--prefix',default='fitted');args=p.parse_args()
f=np.load(args.file);ids=f['indices'];T=f['T_cam_mesh'];p='/work/inputs/';a=np.load(p+'reconstruction.npz');c=json.load(open(p+'camera.json'));K=np.array(c['K']);B=np.linalg.inv(c['T_world_cam'])[None]@a['T_world_mesh'];m=trimesh.load(p+'object_mesh.glb',force='scene',process=False).to_geometry();pts=np.array(m.vertices)[::9];h=h5py.File(p+'masks.h5');v=cv2.VideoCapture(p+'video.mp4');tiles=[]
for idx,i in enumerate(ids):
 v.set(1,int(i));_,im=v.read();ms=h['object'][i];yy,xx=np.where(ms);x0=max(0,xx.min()-50);x1=min(1536,xx.max()+51);y0=max(0,yy.min()-50);y1=min(1152,yy.max()+51);res=[]
 for name,A in [('IMAGE',None),('BASE',B[i]),('FIT',T[idx])]:
  ov=im.copy()
  if A is not None:
   q=pts@A[:3,:3].T+A[:3,3];q=q@K.T;uv=q[:,:2]/q[:,2:];ov[ms]=(.7*im[ms]+.3*np.array([0,255,0])).astype('uint8');u=uv.astype(int);g=(u[:,0]>=0)&(u[:,0]<1536)&(u[:,1]>=0)&(u[:,1]<1152);u=u[g];ov[u[:,1],u[:,0]]=[255,0,255]
  crop=ov[y0:y1,x0:x1];s=min(240/crop.shape[1],300/crop.shape[0]);crop=cv2.resize(crop,None,fx=s,fy=s);canvas=np.full((330,250,3),235,np.uint8);canvas[25:25+crop.shape[0],5:5+crop.shape[1]]=crop;cv2.putText(canvas,f'{i} {name}',(5,20),0,.6,(0,0,0),1);res.append(canvas)
 tiles.append(np.concatenate(res,1))
for j in range(0,len(tiles),6):cv2.imwrite('/work/outputs/'+args.prefix+'_'+str(j//6)+'.jpg',np.concatenate(tiles[j:j+6],0))
