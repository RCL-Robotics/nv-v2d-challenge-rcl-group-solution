import sys; sys.path.insert(0,'/work/scratch')
from common import *
from render import Renderer
import cv2
_R=None
def overlay(i, Ts, labels, out=None, crop=(520,600,320,448)):
    global _R
    if _R is None: _R=Renderer(INP+'/object_mesh.glb')
    r,K,Twc,Kp=load()
    x0,y0,W,H=crop
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[:,:,::-1].astype(np.float32)/255
    c=im[y0:y0+H,x0:x0+W]
    tiles=[c]
    for T,l in zip(Ts,labels):
        rgb,m,z=_R.render(T,K,x0,y0,W,H); rgb=rgb[0].cpu().numpy(); m=m[0].cpu().numpy()
        ov=c.copy(); ov[m]=0.35*ov[m]+0.65*rgb[m]
        e=cv2.Canny((m*255).astype(np.uint8),50,150)>0; ov2=c.copy(); ov2[e]=[1,0,0]
        t=np.hstack([ov,ov2]); t=(t*255).astype(np.uint8).copy(); cv2.putText(t,f'{i} {l}',(5,20),0,0.6,(255,255,0),2); tiles.append(t.astype(np.float32)/255)
    img=(np.hstack(tiles)*255).astype(np.uint8)[:,:,::-1]
    if out: cv2.imwrite(out,img)
    return img
