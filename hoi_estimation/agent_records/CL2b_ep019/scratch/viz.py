import sys; sys.path.insert(0,'/work/outputs')
from common import *
import h5py, cv2
def overlay(frames, Tcs, out, K=None, size=360, labels=None):
    """Tcs: list of (F,4,4) camera poses arrays (one per row-type)."""
    if K is None: K=load_camera()[0]
    R=Renderer(); vid=np.load('/work/scratch/video.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
    cols=[]
    for i in frames:
        om=f['object'][i]; ys,xs=np.where(om)
        if len(xs)==0: cy,cx=576,768
        else: cy,cx=int((ys.min()+ys.max())/2),int((xs.min()+xs.max())/2)
        y0=int(np.clip(cy-size//2,0,1152-size)); x0=int(np.clip(cx-size//2,0,1536-size))
        img=vid[i,y0:y0+size,x0:x0+size].astype(np.float32)/255
        col=[img]
        for T in Tcs:
            o=R.render(K,T[i],w=size,h=size,x0=x0,y0=y0)
            m=o['mask'][0].cpu().numpy(); rgb=o['rgb'][0].cpu().numpy()
            comp=np.where(m[...,None],0.5*rgb+0.5*img,img*0.6)
            e=cv2.Canny((om[y0:y0+size,x0:x0+size]*255).astype(np.uint8),50,100)>0
            comp[e]=(0,1,0)
            col.append(comp)
        c=np.concatenate(col,0); cv2.putText(c,str(i),(3,15),0,0.5,(1,1,0),1)
        cols.append(c)
    cv2.imwrite(out,(np.concatenate(cols,1)[...,::-1]*255).clip(0,255).astype(np.uint8))
