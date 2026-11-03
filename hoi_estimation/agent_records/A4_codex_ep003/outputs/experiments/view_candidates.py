import sys,json,cv2,numpy as np
from PIL import Image,ImageDraw
from render_utils import Renderer
p='/work/inputs/';cam=json.load(open(p+'camera.json'));K=np.array(cam['K']);T=np.linalg.inv(cam['T_world_cam'])@np.load(p+'reconstruction.npz')['T_world_mesh'];q=np.load(sys.argv[1]);ren=Renderer(p+'object_mesh.glb',K);c=cv2.VideoCapture(p+'video.mp4');N=len(q['idx']);out=Image.new('RGB',(128*7,150*N),'white');d=ImageDraw.Draw(out)
for n,f in enumerate(q['idx']):
 t=T[f,:3,3];uv=K@t;origin=np.rint(uv[:2]/uv[2]-64).astype(int);c.set(cv2.CAP_PROP_POS_FRAMES,int(f));_,im=c.read();im=im[origin[1]:origin[1]+128,origin[0]:origin[0]+128,::-1];out.paste(Image.fromarray(im),(0,n*150+22));d.text((0,n*150+4),str(f),fill='black');ids=np.argsort(q['loss'][n])[:5];R=np.concatenate([T[f,:3,:3][None],q['R'][n,ids]]);tt=np.concatenate([t[None],q['t'][n,ids]]);rgb,mask=ren.render(R,tt,origin)
 for j,a in enumerate(rgb.detach().cpu().numpy()):
  out.paste(Image.fromarray((a*255).astype('uint8')),((j+1)*128,n*150+22));d.text(((j+1)*128,n*150+4),'baseline' if j==0 else '%s %.3f'%(ids[j-1],q['loss'][n,ids[j-1]]),fill='black')
out.save('/work/outputs/fit_contact_sheet.jpg')
