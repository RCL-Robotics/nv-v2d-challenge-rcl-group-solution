import numpy as np,cv2,trimesh,h5py
from fit_experiment import prepare_mesh
p='/work/inputs/';m,pts=prepare_mesh(p+'object_mesh.glb',1600)
# texture lookup by nearest vertex (sample points are close compared to texture band width).
from scipy.spatial import cKDTree
_,ix=cKDTree(m.vertices).query(pts);uv=m.visual.uv[ix];tex=np.array(m.visual.material.baseColorTexture.convert('RGB'));rgb=tex[np.clip(((1-uv[:,1])*(len(tex)-1)).astype(int),0,len(tex)-1),np.clip((uv[:,0]*(tex.shape[1]-1)).astype(int),0,tex.shape[1]-1)];hsv=cv2.cvtColor(rgb[None],cv2.COLOR_RGB2HSV)[0];orange=(hsv[:,0]<28)&(hsv[:,0]>3)&(hsv[:,1]>110)&(hsv[:,2]>70);print('orange',orange.mean());np.savez('/work/scratch/meshcolor.npz',pts=pts,rgb=rgb,orange=orange)
v=cv2.VideoCapture(p+'video.mp4');h=h5py.File(p+'masks.h5');tiles=[]
for f in [0,70,140,210,245,315,385,490,560,630,700,770,805,840,875]:
 v.set(1,f);_,im=v.read();ms=h['object'][f];hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);og=(hsv[:,:,0]>3)&(hsv[:,:,0]<28)&(hsv[:,:,1]>110)&(hsv[:,:,2]>70)&(cv2.dilate(ms.astype('uint8'),np.ones((5,5),np.uint8))>0);yy,xx=np.where(ms);b=[max(0,xx.min()-30),max(0,yy.min()-30),min(1536,xx.max()+30),min(1152,yy.max()+30)];ov=im.copy();ov[og]=[255,0,255];crop=ov[b[1]:b[3],b[0]:b[2]];s=min(240/crop.shape[1],310/crop.shape[0]);crop=cv2.resize(crop,None,fx=s,fy=s);can=np.full((340,250,3),230,np.uint8);can[25:25+crop.shape[0],5:5+crop.shape[1]]=crop;cv2.putText(can,str(f),(5,20),0,.6,(0,0,0),1);tiles.append(can)
cv2.imwrite('/work/outputs/orange.jpg',np.concatenate([np.concatenate(tiles[j:j+5],1) for j in range(0,15,5)],0))
