import cv2,numpy as np,h5py,json,trimesh
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
root='/work/inputs/'
a=np.load(root+'reconstruction.npz'); cam=json.load(open(root+'camera.json'));K=np.array(cam['K']); wc=np.array(cam['T_world_cam']); poses=np.linalg.inv(wc)@a['T_world_mesh'];mesh=trimesh.load(root+'object_mesh.glb',force='scene',process=False).to_geometry();v=mesh.vertices
cap=cv2.VideoCapture(root+'video.mp4');mask=h5py.File(root+'masks.h5','r')['object']
frames=[0,40,80,120,160,200,240,280,320,360,400,440,480,520,564]
out=Image.new('RGB',(5*350,3*300))
for i,f in enumerate(frames):
 cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,f);ret,im=cap.read(); y,x=np.where(mask[f]);p=v@poses[f,:3,:3].T+poses[f,:3,3];uv=p@K.T;uv=uv[:,:2]/uv[:,2:];
 xmin=max(0,min(int(uv[:,0].min()),x.min())-60);xmax=min(1536,max(int(uv[:,0].max()),x.max())+60);ymin=max(0,min(int(uv[:,1].min()),y.min())-60);ymax=min(1152,max(int(uv[:,1].max()),y.max())+60)
 cv2.polylines(im,[cv2.convexHull(uv.astype(np.float32)).astype(np.int32)],True,(0,0,255),1)
 contours,_=cv2.findContours(mask[f].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,(0,255,0),1)
 im=cv2.cvtColor(im[ymin:ymax,xmin:xmax],cv2.COLOR_BGR2RGB);tile=Image.fromarray(im);tile.thumbnail((350,270));out.paste(tile,(i%5*350,i//5*300+25));ImageDraw.Draw(out).text((i%5*350+5,i//5*300+5),f'Frame {f}',fill='white')
out.save('/work/scratch/montage.jpg')
# pose stats
for f in frames: print(f,'t',np.round(poses[f,:3,3],3),'r',np.round(Rotation.from_matrix(poses[f,:3,:3]).as_euler('xyz',degrees=True),1),'contact',np.round(a['contact_prob_right_left'][f],2))
