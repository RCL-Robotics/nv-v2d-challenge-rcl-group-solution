import numpy as np, json,cv2,torch
from render_utils import Renderer
p='/work/inputs/';a=np.load(p+'reconstruction.npz');c=json.load(open(p+'camera.json'));K=np.array(c['K']);T=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];r=Renderer(p+'object_mesh.glb',K,160);cap=cv2.VideoCapture(p+'video.mp4');panels=[]
for f in [0,240,250,400,550,570,600,650,680,700]:
 uv=K@T[f,:3,3];uv=uv[:2]/uv[2];box=np.r_[np.round(uv)-80,1.]
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,im=cap.read();x,y=box[:2].astype(int);im=im[y:y+160,x:x+160]
 Rs=np.array([T[f,:3,:3],a['T_cam_mesh_foundationpose'][f,:3,:3]]);ts=np.repeat(T[f,:3,3][None],2,0)
 with torch.no_grad():m,attr,col=r.render(Rs,ts,np.tile(box,(2,1)),texture=True)
 col=col.cpu().numpy()[...,::-1];m=m.cpu().numpy();imgs=[]
 for j in range(2):
  b=im.copy();b[m[j]>0]=col[j][m[j]>0]*255;imgs.append(b)
 panel=np.concatenate([im]+imgs,axis=1);panel=cv2.resize(panel,(720,240));cv2.putText(panel,str(f),(5,22),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2);panels.append(panel)
cv2.imwrite('/work/outputs/render_compare.jpg',np.concatenate(panels,axis=0))
