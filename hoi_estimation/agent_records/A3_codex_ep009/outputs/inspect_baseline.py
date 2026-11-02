import os,json,cv2,h5py,numpy as np,torch
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from render_utils import Renderer
p='/work/inputs/'
K=np.array(json.load(open(p+'camera.json'))['K']); C=np.array(json.load(open(p+'camera.json'))['T_world_cam']);r=np.load(p+'reconstruction.npz');poses=np.linalg.inv(C)@r['T_world_mesh']
np.set_printoptions(precision=3,suppress=True)
for i in range(0,870,25):
 R=poses[i,:3,:3];print(i,'centre',poses[i,:3,3],'y',R[:,1],'rot',Rotation.from_matrix(R).as_euler('xyz',degrees=True))
ren=Renderer(p+'object_mesh.glb'); cap=cv2.VideoCapture(p+'video.mp4'); masks=h5py.File(p+'masks.h5')
frames=[0,100,170,200,230,260,300,350,400,450,500,550,600,630,650,680,710,740,770,820,869]
w=192
sheet=Image.new('RGB',(7*w*2,3*w),'white');d=ImageDraw.Draw(sheet)
for j,i in enumerate(frames):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,bgr=cap.read(); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 ys,xs=np.where(masks['object'][i]);cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2
 x0,y0=int(cx-64),int(cy-64);crop=rgb[y0:y0+128,x0:x0+128]
 kk=K.copy();kk[0,2]-=x0;kk[1,2]-=y0;kk[:2]*=1.5
 with torch.no_grad(): rr,mm=ren.render(poses[i,:3,:3],poses[i,:3,3],kk,(w,w))
 rr=rr[0].cpu().numpy();mm=mm[0].cpu().numpy();im=cv2.resize(crop,(w,w)).astype(float)/255
 overlay=im*.55+rr*.45+(1-mm[:,:,None])*im*.45
 # Cyan silhouette around the renderer.
 cs,_=cv2.findContours((mm>.5).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,cs,-1,(0.,1.,1.),1)
 out=(overlay.clip(0,1)*255).astype(np.uint8)
 x=(j%7)*w*2;y=(j//7)*w
 sheet.paste(Image.fromarray(out),(x,y));sheet.paste(Image.fromarray((rr*255).astype(np.uint8)),(x+w,y));d.text((x+5,y+5),str(i),fill='white',stroke_fill='black',stroke_width=2)
sheet.save('/work/outputs/diagnostics/baseline_renders.jpg')
