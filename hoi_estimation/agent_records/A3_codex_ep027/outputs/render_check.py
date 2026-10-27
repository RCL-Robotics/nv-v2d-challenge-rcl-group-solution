import os
os.environ['PYOPENGL_PLATFORM']='egl'
import trimesh,pyrender,cv2,numpy as np,json
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
root='/work/inputs/';mesh=trimesh.load(root+'object_mesh.glb',force='scene',process=False).to_geometry();print(type(mesh.visual.material),mesh.visual.material)
cam=json.load(open(root+'camera.json'));K=np.array(cam['K']);wc=np.array(cam['T_world_cam']);poses=np.linalg.inv(wc)@np.load(root+'reconstruction.npz')['T_world_mesh']
scene=pyrender.Scene(ambient_light=[1,1,1],bg_color=[0,0,0,0]);node=scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False));camera=pyrender.IntrinsicsCamera(K[0,0],K[1,1],K[0,2],K[1,2],znear=.01,zfar=20);scene.add(camera,pose=np.diag([1,-1,-1,1]));renderer=pyrender.OffscreenRenderer(1536,1152)
cap=cv2.VideoCapture(root+'video.mp4');frames=[0,80,140,160,180,200,240,280,320,360,400,440,480,564];out=Image.new('RGB',(300*4,230*len(frames)),(30,30,30));draw=ImageDraw.Draw(out)
for i,f in enumerate(frames):
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);_,im=cap.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);T=poses[f];v=mesh.vertices@T[:3,:3].T+T[:3,3];uv=v@K.T;uv=uv[:,:2]/uv[:,2:];c=uv.mean(0);x0=int(c[0]-95);x1=x0+190;y0=int(c[1]-70);y1=y0+140
 ims=[im[y0:y1,x0:x1]]
 for deg in [0,180]:
  S=np.eye(4);S[:3,:3]=Rotation.from_euler('y',deg,degrees=True).as_matrix();scene.set_pose(node,T@S);col,dep=renderer.render(scene,flags=pyrender.RenderFlags.FLAT);blend=im.copy();blend[dep>0]=col[dep>0,:3];ims.append(blend[y0:y1,x0:x1]);
  if deg==0: ims.append(col[y0:y1,x0:x1,:3])
 for j,img in enumerate(ims):out.paste(Image.fromarray(img).resize((300,220)),(300*j,230*i+10))
 draw.text((0,230*i),str(f),fill='white')
out.save('/work/scratch/render_checks.jpg');renderer.delete()
