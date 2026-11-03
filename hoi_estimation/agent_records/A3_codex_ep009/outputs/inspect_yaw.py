import json,cv2,numpy as np,torch
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from render_utils import Renderer
p='/work/inputs/';K=np.array(json.load(open(p+'camera.json'))['K']);C=np.array(json.load(open(p+'camera.json'))['T_world_cam']);poses=np.linalg.inv(C)@np.load(p+'reconstruction.npz')['T_world_mesh'];ren=Renderer(p+'object_mesh.glb');cap=cv2.VideoCapture(p+'video.mp4')
for i in [0,300,400,820]:
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,bgr=cap.read(); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);pc=K@poses[i,:3,3];pc=pc[:2]/pc[2];x0,y0=np.floor(pc-64).astype(int);kk=K.copy();kk[0,2]-=x0;kk[1,2]-=y0;kk[:2]*=3
 angles=np.arange(0,360,30);R=poses[i,:3,:3]@Rotation.from_euler('y',angles,degrees=True).as_matrix();t=np.repeat(poses[i,None,:3,3],len(angles),axis=0)
 with torch.no_grad(): rr,mm=ren.render(R,t,kk,(384,384))
 sheet=Image.new('RGB',(6*384,3*384),'#303030');d=ImageDraw.Draw(sheet)
 im=Image.fromarray(rgb[y0:y0+128,x0:x0+128]).resize((384,384));sheet.paste(im,(0,0))
 for j,a in enumerate(angles):
  x=(j%6)*384;y=(1+j//6)*384
  sheet.paste(Image.fromarray((rr[j].cpu().numpy().clip(0,1)*255).astype(np.uint8)),(x,y));d.text((x+5,y+5),f'{a:+d}',fill='white')
 sheet.save(f'/work/outputs/diagnostics/yaw_{i}.jpg')
