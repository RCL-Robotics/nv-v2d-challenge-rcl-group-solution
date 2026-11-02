import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py, torch
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec=load_inputs(inp)
mesh=load_mesh(inp); Rd=Renderer(mesh)
Tw=np.load(sys.argv[1])['T_world_mesh'] if len(sys.argv)>1 and sys.argv[1]!='base' else rec['T_world_mesh'].astype(float)
out=sys.argv[2] if len(sys.argv)>2 else '/work/scratch/viz/ov_base.jpg'
frames=[int(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else [0,60,110,125,135,150,180,220,250,264,280,300,318,330,342,354,366,380,400,476]
om=h5py.File(inp+'/masks.h5')['object']
Tc=np.linalg.inv(Twc)@Tw
ims=[]
for i in frames:
    im=cv2.imread(f'/work/scratch/frames/{i+1:04d}.jpg')
    p=K@Tc[i,:3,3]; cx,cy=int(p[0]/p[2]),int(p[1]/p[2])
    box=(cx-60,cy-80,cx+60,cy+80)
    rgb,m,z=Rd.render(torch.tensor(Tc[i:i+1],device='cuda'),K,box,(160,120))
    rgb=(rgb[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); m=m[0].cpu().numpy()
    c=im[box[1]:box[3],box[0]:box[2]].copy()
    if c.shape!=(160,120,3): c=np.zeros((160,120,3),np.uint8)
    bl=c.copy(); bl[m]=(0.5*bl[m]+0.5*rgb[m]).astype(np.uint8)
    mm=om[i][box[1]:box[3],box[0]:box[2]].astype(np.uint8) if c.any() else np.zeros((160,120),np.uint8)
    cnt,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(bl,cnt,-1,(0,0,255),1)
    rr=np.zeros_like(c); rr[m]=rgb[m]
    t=np.hstack([c,bl,rr]); t=cv2.resize(t,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    cv2.putText(t,str(i),(5,25),0,0.8,(0,255,0),2); ims.append(t)
while len(ims)%4: ims.append(np.zeros_like(ims[0]))
cv2.imwrite(out,np.vstack([np.hstack(ims[i:i+4]) for i in range(0,len(ims),4)]))
