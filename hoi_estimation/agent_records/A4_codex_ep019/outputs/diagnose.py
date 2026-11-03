import json, os, cv2, h5py, numpy as np, trimesh
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
p='/work/inputs/';out='/work/outputs/'
a=np.load(p+'reconstruction.npz');cam=json.load(open(p+'camera.json'));K=np.array(cam['K']);W=np.array(cam['T_world_cam']);T=np.linalg.inv(W)[None]@a['T_world_mesh']
m=trimesh.load(p+'object_mesh.glb',force='scene',process=False).to_geometry(); pts=np.array(m.vertices)[::7]
print('translation stats',np.min(T[:,:3,3],axis=0),np.max(T[:,:3,3],axis=0))
rot=Rotation.from_matrix(T[:,:3,:3]); dr=np.degrees((rot[:-1].inv()*rot[1:]).magnitude());print('rot step quantiles',np.percentile(dr,[0,50,90,95,99,100]));print('large steps',[(int(i+1),round(float(dr[i]),2)) for i in np.where(dr>15)[0]])
v=cv2.VideoCapture(p+'video.mp4');h=h5py.File(p+'masks.h5','r')
indices=list(range(0,len(T),35))+[len(T)-1];tiles=[];boxes=[]
for i in indices:
 v.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=v.read();mask=h['object'][i];person=h['person'][i]
 q=pts@T[i,:3,:3].T+T[i,:3,3];uv=q@K.T;uv=uv[:,:2]/uv[:,2:]
 yy,xx=np.where(mask); b=[max(0,int(min(xx.min() if len(xx) else 768,uv[:,0].min()))-50),max(0,int(min(yy.min() if len(yy) else 576,uv[:,1].min()))-50),min(1536,int(max(xx.max() if len(xx) else 768,uv[:,0].max()))+50),min(1152,int(max(yy.max() if len(yy) else 576,uv[:,1].max()))+50)]
 boxes.append(b);x0,y0,x1,y1=b
 overlay=im.copy();overlay[mask]=(.6*im[mask]+.4*np.array([0,255,0])).astype(np.uint8)
 for u in uv.astype(int):
  if 0<=u[0]<1536 and 0<=u[1]<1152: overlay[u[1],u[0]]=[255,0,255]
 def tile(img):
  c=img[y0:y1,x0:x1];scale=min(300/c.shape[1],340/c.shape[0]);c=cv2.resize(c,None,fx=scale,fy=scale);o=np.full((370,310,3),235,np.uint8);o[25:25+c.shape[0],5:5+c.shape[1]]=c;cv2.putText(o,str(i),(8,20),0,.6,(20,20,20),1);return o
 tiles.append(np.concatenate([tile(im),tile(overlay)],axis=1))
for k in range(0,len(tiles),8):
 group=tiles[k:k+8];group+=[np.full_like(tiles[0],235)]*(8-len(group)); montage=np.concatenate([np.concatenate(group[j:j+2],axis=1) for j in range(0,8,2)],axis=0);cv2.imwrite(out+f'baseline_montage_{k//8}.jpg',montage)
# Mesh orthographic views with red +z end and blue -z end.
vs=np.asarray(m.vertices); uv0=np.asarray(m.visual.uv);tex=np.asarray(m.visual.material.baseColorTexture.convert('RGB'));colors=tex[np.clip(((1-uv0[:,1])*(len(tex)-1)).astype(int),0,len(tex)-1),np.clip((uv0[:,0]*(tex.shape[1]-1)).astype(int),0,tex.shape[1]-1)][:,::-1]
views=[]
for a0,a1,depth in [(0,2,1),(1,2,0),(0,1,2)]:
 arr=np.full((800,400,3),240,np.uint8);xy=vs[:,[a0,a1]];xy=xy*700+np.array([200,360]);idx=np.argsort(vs[:,depth]);ip=xy.astype(int);good=(ip[:,0]>=0)&(ip[:,0]<400)&(ip[:,1]>=0)&(ip[:,1]<800);idx=idx[good[idx]];arr[ip[idx,1],ip[idx,0]]=colors[idx];cv2.putText(arr,f'axes {a0},{a1}',(10,25),0,.6,(0,0,0),1);views.append(arr)
cv2.imwrite(out+'mesh_views.jpg',np.concatenate(views,axis=1))
np.savez('/work/scratch/diagnose.npz',T=T,rot_step=dr)
