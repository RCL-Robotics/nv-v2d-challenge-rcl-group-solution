from common import *; from render import Renderer; import cv2, torch
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
R=Renderer(m,K)
T=r['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@T
Ry=np.diag([-1,1,-1,1.])
msk=h5py.File('/work/inputs/masks.h5')
# which end has green wheels? render top-down labels: use texture colour of vertices below deck
F=len(T); res=np.full((F,2),np.nan); 
for i in range(0,F,3):
    o=msk['object'][i]; per=msk['person'][i]; ys,xs=np.nonzero(o)
    if len(xs)<50: continue
    x0,y0=xs.min()-20,ys.min()-20; w,h=xs.max()-x0+20,ys.max()-y0+20
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+h,x0:x0+w][...,::-1].astype(np.float32)/255
    Gi=im[...,1]-0.5*(im[...,0]+im[...,2])
    pc=per[y0:y0+h,x0:x0+w]
    cols,mks,_=R.render(np.stack([Tc[i],Tc[i]@Ry]),crop=(x0,y0,w,h))
    for k in range(2):
        c=cols[k].cpu().numpy(); mk=mks[k].cpu().numpy()>0
        Gr=c[...,1]-0.5*(c[...,0]+c[...,2])
        green=mk&(Gr>0.15)&~pc
        grey=mk&(Gr<0.03)&(c.mean(-1)>0.2)&(c.mean(-1)<0.6)&(np.abs(c[...,0]-c[...,2])<0.05)&~pc  # grey wheel/truck
        if green.sum()>5 and grey.sum()>5: res[i,k]=Gi[green].mean()-Gi[grey].mean()
np.save('/work/scratch/an/wheel.npy',res)
for a in range(0,F,30):
    b=min(a+30,F); s=res[a:b]; print(a,b,np.nanmean(s,0).round(4), np.nanmean((s[:,0]>s[:,1])[~np.isnan(s[:,0])]).round(2))
