import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py
from hoilib import *
K,Twc,rec,mesh,plane=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
poses=sys.argv[1]; idx=[int(x) for x in sys.argv[2].split(',')]; out=sys.argv[3]
if poses=='base': T=np.einsum('ij,fjk->fik',Tcw,rec['T_world_mesh'].astype(float))
else: T=np.einsum('ij,fjk->fik',Tcw,np.load(poses)['T_world_mesh'])
rd=Renderer(mesh,K,0.5)
ims=read_video('/work/inputs/video.mp4',idx,0.5)
m,z,rgb=rd.render(T[idx])
m=m.cpu().numpy(); rgb=rgb.cpu().numpy()
f=h5py.File('/work/inputs/masks.h5')
tiles=[]
for k,i in enumerate(idx):
    im=ims[k].astype(float)/255
    a=im.copy(); a[m[k]]=0.5*a[m[k]]+0.5*rgb[k][m[k]]
    om=cv2.resize(f['object'][i].astype(np.uint8),(rd.w,rd.h),interpolation=cv2.INTER_NEAREST)>0
    cnt,_=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    a=(a*255).astype(np.uint8)[:,:,::-1].copy()
    cv2.drawContours(a,cnt,-1,(0,255,0),1)
    cnt2,_=cv2.findContours(m[k].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(a,cnt2,-1,(0,0,255),1)
    ys,xs=np.where(om|m[k])
    if len(ys):
        c=[int((ys.min()+ys.max())/2),int((xs.min()+xs.max())/2)]
    else: c=[rd.h//2,rd.w//2]
    s=180; y0=np.clip(c[0]-s,0,rd.h-2*s); x0=np.clip(c[1]-s,0,rd.w-2*s)
    t=a[y0:y0+2*s,x0:x0+2*s].copy(); cv2.putText(t,str(i),(5,25),0,0.8,(255,255,0),2); tiles.append(t)
while len(tiles)%5: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(out,np.vstack([np.hstack(tiles[i:i+5]) for i in range(0,len(tiles),5)]))
