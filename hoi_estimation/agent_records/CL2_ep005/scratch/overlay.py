import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch
from hoi_lib import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
frames=np.load('frames_half.npy', mmap_mode='r'); om=np.load('objmask_half.npy', mmap_mode='r')
def overlay(T_wm, idx, out, label=''):
    Tc=cam_from_world(d['T_wc'], T_wm[idx])
    ims=[]
    for k,i in enumerate(idx):
        r=R.render(Tc[k:k+1], d['K'], d['H'], d['W'], scale=0.5)
        rgb=r['rgb'][0].cpu().numpy(); m=r['mask'][0].cpu().numpy()
        im=frames[i].astype(np.float32)/255
        im2=im.copy(); im2[m]=0.4*im[m]+0.6*np.clip(rgb[m]*2.5,0,1)
        cnt,_=cv2.findContours(om[i].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        im2=(im2*255).astype(np.uint8).copy(); cv2.drawContours(im2,cnt,-1,(0,255,0),1)
        ys,xs=np.where(om[i]|m); 
        y0,y1,x0,x1=max(ys.min()-30,0),ys.max()+30,max(xs.min()-30,0),xs.max()+30
        crop=cv2.resize(im2[y0:y1,x0:x1],(256,256))
        cv2.putText(crop,f'{label}{i}',(5,20),0,0.6,(255,255,0),2)
        ims.append(crop)
    while len(ims)%6: ims.append(np.zeros_like(ims[0]))
    cv2.imwrite(out, np.vstack([np.hstack(ims[j:j+6]) for j in range(0,len(ims),6)])[...,::-1])
if __name__=='__main__':
    T=d['rec']['T_world_mesh'].astype(np.float64)
    idx=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else list(range(0,1003,42))
    overlay(T, idx, sys.argv[1])
