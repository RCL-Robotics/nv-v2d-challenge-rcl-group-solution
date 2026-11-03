import numpy as np,cv2,h5py,sys
from rend import *
cam,r=load_all(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
Rn=None
H5=h5py.File('/work/inputs/masks.h5')
def panel(i,Tcm,label='',S=500):
    global Rn
    if Rn is None: Rn=Renderer(K)
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    om=H5['object'][i]; pm=H5['person'][i]
    out=[]
    ys,xs=np.where(om)
    col,m,z=Rn.render(Tcm); m=m[0].cpu().numpy(); c=(col[0].cpu().numpy()[...,::-1]*255)
    ys2,xs2=np.where(m)
    allx=np.concatenate([xs,xs2]); ally=np.concatenate([ys,ys2])
    if len(allx)==0: allx=np.array([768]); ally=np.array([576])
    x0,x1=allx.min()-20,allx.max()+20; y0,y1=ally.min()-20,ally.max()+20
    s=max(x1-x0,y1-y0); cx,cy=(x0+x1)//2,(y0+y1)//2
    x0=int(np.clip(cx-s//2,0,1536-s)) if s<1536 else 0; y0=int(np.clip(cy-s//2,0,1152-s)) if s<1152 else 0
    s=min(s,1152)
    ov=im.copy(); ov[m]=(0.35*im[m]+0.65*c[m]).astype(np.uint8)
    cnt=im.copy()
    for mm,colr in [(om,(0,0,255)),(m,(0,255,0)),(pm,(255,0,0))]:
        cs,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(cnt,cs,-1,colr,1)
    t=np.hstack([cv2.resize(a[y0:y0+s,x0:x0+s],(S,S)) for a in (im,ov,cnt)])
    cv2.putText(t,f'{i} {label}',(10,30),0,0.9,(0,255,255),2)
    return t
if __name__=='__main__':
    T=np.linalg.inv(Twc)@r['T_world_mesh'].astype(np.float64)
    fr=[int(a) for a in sys.argv[2].split(',')]
    cv2.imwrite(sys.argv[1],np.vstack([panel(i,T[i],'base',S=int(sys.argv[3]) if len(sys.argv)>3 else 400) for i in fr]))
