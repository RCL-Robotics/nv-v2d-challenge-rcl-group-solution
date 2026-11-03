import sys;sys.path.insert(0,'/work/outputs')
from common import *
import h5py,cv2
def zoom(frames,Ts,out,center=None,size=120,scale=3):
    K=load_camera()[0]; R=Renderer(); vid=np.load('/work/scratch/video.npy',mmap_mode='r'); f=h5py.File('/work/inputs/masks.h5')
    cols=[]
    for i in frames:
        om=f['object'][i]
        cx,cy=center
        x0,y0=cx-size//2,cy-size//2
        img=vid[i,y0:y0+size,x0:x0+size].astype(np.float32)/255
        col=[img]
        e=cv2.Canny((om[y0:y0+size,x0:x0+size]*255).astype(np.uint8),50,100)>0
        for T in Ts:
            o=R.render(K,T[i] if T.ndim==3 else T,w=size,h=size,x0=x0,y0=y0)
            m=o['mask'][0].cpu().numpy(); rgb=o['rgb'][0].cpu().numpy()
            comp=np.where(m[...,None],0.6*rgb+0.4*img,img*0.7); comp[e]=(0,1,0); col.append(comp)
        c=np.concatenate(col,0); c=cv2.resize(c,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST); cv2.putText(c,str(i),(3,15),0,0.5,(1,1,0),1)
        cols.append(c)
    cv2.imwrite(out,(np.concatenate(cols,1)[...,::-1]*255).clip(0,255).astype(np.uint8))
