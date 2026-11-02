import json,numpy as np,cv2,torch
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from render_utils import Renderer
p='/work/inputs/';cam=json.load(open(p+'camera.json'));K=np.array(cam['K']);T=np.linalg.inv(cam['T_world_cam'])@np.load(p+'reconstruction.npz')['T_world_mesh'];ren=Renderer(p+'object_mesh.glb',K);c=cv2.VideoCapture(p+'video.mp4');out=Image.new('RGB',(128*9,160*5),'white');d=ImageDraw.Draw(out)
for i,f in enumerate([0,160,300,440,720]):
 t=T[f,:3,3];uv=K@t;uv=uv[:2]/uv[2];origin=(uv-64).astype(int);c.set(cv2.CAP_PROP_POS_FRAMES,f);_,im=c.read();im=im[origin[1]:origin[1]+128,origin[0]:origin[0]+128,::-1];out.paste(Image.fromarray(im),(0,i*160+25));d.text((0,i*160+5),str(f),fill='black')
 R=T[f,:3,:3][None]@Rotation.from_euler('z',np.arange(8)*45,degrees=True).as_matrix();rgb,mask=ren.render(R,t,origin)
 for j,a in enumerate(rgb.detach().cpu().numpy()):
  out.paste(Image.fromarray((a*255).astype('uint8')),((j+1)*128,i*160+25));d.text(((j+1)*128,i*160+5),str(j*45),fill='black')
out.save('/work/outputs/rotation_candidates.jpg')
