import cv2,numpy as np,h5py,json,trimesh
from PIL import Image,ImageDraw
p='/work/inputs/';cap=cv2.VideoCapture(p+'video.mp4');f=h5py.File(p+'masks.h5');a=np.load(p+'reconstruction.npz');c=json.load(open(p+'camera.json'));K=np.array(c['K']);T=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];m=trimesh.load(p+'object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices[::20]; fs=[0,125,150,175,200,300,450,500,550,600,650,700,725,750,775,800,850,900,950,987];canvas=Image.new('RGB',(1600,5*450),'white');d=ImageDraw.Draw(canvas)
for j,i in enumerate(fs):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();mask=f['object'][i];yy,xx=np.where(mask);x0,x1=max(0,xx.min()-40),min(1536,xx.max()+40);y0,y1=max(0,yy.min()-40),min(1152,yy.max()+40);crop=im[y0:y1,x0:x1];h,w=crop.shape[:2];s=min(400/w,420/h);crop=cv2.resize(crop,(int(w*s),int(h*s)));canvas.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),((j%4)*400,(j//4)*450+30));d.text(((j%4)*400+5,(j//4)*450+5),str(i),fill='black')
 for name in ['T_cam_mesh_coconet','T_cam_mesh_foundationpose']:
  U=a[name][i]; print(i,name, 't',U[:3,3].round(3),'baseline t',T[i,:3,3].round(3),'deltaR',np.trace(T[i,:3,:3].T@U[:3,:3]))
canvas.save('/work/outputs/rgb_details.jpg')
