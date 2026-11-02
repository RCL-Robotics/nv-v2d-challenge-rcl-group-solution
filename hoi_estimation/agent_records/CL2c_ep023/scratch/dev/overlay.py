import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py, torch
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp)
mesh=load_mesh(inp)
R=Renderer(mesh,K,1536,1152)
poses=np.load(sys.argv[1])['T_world_mesh'] if len(sys.argv)>1 and sys.argv[1].endswith('.npz') else rec['T_world_mesh']
tag=sys.argv[2] if len(sys.argv)>2 else 'base'
idx=[int(x) for x in sys.argv[3].split(',')] if len(sys.argv)>3 else list(range(0,946,60))
frames=read_video(inp+'/video.mp4',idx)
mf=h5py.File(inp+'/masks.h5')
for i,fr in zip(idx,frames):
    Tc=np.linalg.inv(Twc)@poses[i].astype(np.float64)
    o=R.render(Tc)
    m=o['mask'][0].cpu().numpy(); rgb=(o['rgb'][0].cpu().numpy()*255).astype(np.uint8)
    img=fr.copy()
    img[m]=(0.5*img[m]+0.5*rgb[m]).astype(np.uint8)
    om=mf['object'][i].astype(np.uint8)
    cs,_=cv2.findContours(om,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,cs,-1,(0,255,0),1)
    ys,xs=np.nonzero(om|m)
    if len(xs)==0: ys,xs=np.array([576]),np.array([768])
    cx,cy=int(xs.mean()),int(ys.mean()); s=150
    x0,y0=max(0,cx-s),max(0,cy-s)
    crop=img[y0:y0+2*s,x0:x0+2*s]; raw=fr[y0:y0+2*s,x0:x0+2*s]
    both=np.concatenate([raw,crop],1)
    both=cv2.resize(both,None,fx=1.5,fy=1.5)
    cv2.putText(both,str(i),(5,25),0,0.8,(255,255,0),2)
    cv2.imwrite(f'/work/scratch/frames/{tag}_{i:04d}.jpg',both[:,:,::-1])
