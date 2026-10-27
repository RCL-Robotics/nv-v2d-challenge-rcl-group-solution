import numpy as np, cv2, json, trimesh, h5py
from pathlib import Path
from PIL import Image, ImageDraw
p=Path('/work/inputs'); out=Path('/work/outputs'); a=np.load(p/'reconstruction.npz'); cam=json.load(open(p/'camera.json')); K=np.array(cam['K']); C=np.linalg.inv(cam['T_world_cam']); T=C@a['T_world_mesh']; mesh=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry(); verts=np.array(mesh.vertices)[::20]
frames=[0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,858]
cap=cv2.VideoCapture(str(p/'video.mp4')); h=h5py.File(p/'masks.h5'); panels=[]
for f in frames:
 cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,f); ok,im=cap.read(); assert ok
 pts=verts@T[f,:3,:3].T+T[f,:3,3]; uv=pts@K.T; uv=uv[:,:2]/uv[:,2:]
 m=h['object'][f]; ys,xs=np.where(m); bb=[xs.min(),ys.min(),xs.max(),ys.max()] if len(xs) else [0,0,0,0]
 xyz=np.array([[0,0,0],[.06,0,0],[0,.06,0],[0,0,.17]])@T[f,:3,:3].T+T[f,:3,3]; ax=xyz@K.T; ax=ax[:,:2]/ax[:,2:]
 center=ax[0]; x,y=np.round(center).astype(int); half=125; x0=max(0,x-half);y0=max(0,y-half); crop=im[y0:y0+250,x0:x0+250].copy()
 contours,_=cv2.findContours(m.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(crop,[c-np.array([[[x0,y0]]]) for c in contours],-1,(0,255,255),1)
 hull=cv2.convexHull(np.round(uv-[x0,y0]).astype('int32'));cv2.polylines(crop,[hull],True,(255,0,255),1)
 ax=np.round(ax-[x0,y0]).astype(int)
 for j,col in [(1,(0,0,255)),(2,(0,255,0)),(3,(255,0,0))]:cv2.arrowedLine(crop,ax[0],ax[j],col,1,tipLength=.15)
 crop=cv2.resize(crop,(375,375)); cv2.putText(crop,f'frame {f}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),3);cv2.putText(crop,f'frame {f}',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),1);panels.append(crop)
 cv2.imwrite(str(out/f'frame_{f:04d}.jpg'),im) if f in [0,300,600] else None
canvas=np.concatenate([np.concatenate(panels[i:i+6],axis=1) for i in range(0,18,6)],axis=0);cv2.imwrite(str(out/'baseline_montage.jpg'),canvas)
np.savez_compressed(out/'inspection_data.npz',T_cam_mesh=T)
print('translation ranges',T[:,:3,3].min(axis=0),T[:,:3,3].max(axis=0)); print('contact',a['contact_prob_right_left'].min(axis=0),a['contact_prob_right_left'].max(axis=0))
print('material',mesh.visual.material); tex=mesh.visual.material.baseColorTexture; print('texture',tex.size);tex.save(out/'texture.png')
