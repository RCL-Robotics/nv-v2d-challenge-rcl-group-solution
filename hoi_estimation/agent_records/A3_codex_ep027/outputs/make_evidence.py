import os
os.environ['PYOPENGL_PLATFORM']='egl'
import cv2,json,trimesh,pyrender,numpy as np
from PIL import Image,ImageDraw
root='/work/inputs/';cam=json.load(open(root+'camera.json'));K=np.array(cam['K']);C=np.linalg.inv(np.array(cam['T_world_cam']));base=C@np.load(root+'reconstruction.npz')['T_world_mesh'];new=C@np.load('/work/outputs/object_poses_world.npz')['T_world_mesh'];mesh=trimesh.load(root+'object_mesh.glb',force='scene',process=False).to_geometry();scene=pyrender.Scene();node=scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False));camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],128,128);scene.add(camera,pose=np.diag([1,-1,-1,1]));renderer=pyrender.OffscreenRenderer(256,256);cap=cv2.VideoCapture(root+'video.mp4');frames=[40,100,130,323,349,385];out=Image.new('RGB',(3*384,6*230),(25,25,25));draw=ImageDraw.Draw(out)
for i,f in enumerate(frames):
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);_,im=cap.read();p=base[f,:3,:3]@mesh.bounds.mean(0)+base[f,:3,3];uv=K@p;cx,cy=np.rint(uv[:2]/uv[2]).astype(int);x0,y0=cx-128,cy-128;camera.cx=K[0,2]-x0;camera.cy=K[1,2]-y0;crop=im[y0:y0+256,x0:x0+256];tiles=[crop.copy()]
 for j,T in enumerate([base[f],new[f]]):
  scene.set_pose(node,T);mask=(renderer.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)>0).astype('uint8');contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);tile=crop.copy();cv2.drawContours(tile,contours,-1,(255,0,255) if j==0 else (255,255,0),1);tiles.append(tile)
 for j,tile in enumerate(tiles):
  tile=tile[80:176,48:208];tile=Image.fromarray(cv2.cvtColor(tile,cv2.COLOR_BGR2RGB)).resize((384,208));out.paste(tile,(j*384,i*230+22));draw.text((j*384+5,i*230+5),f'Frame {f}: '+['video','baseline','refined'][j],fill='white')
renderer.delete();out.save('/work/outputs/alignment_examples.jpg',quality=94)
