from common import *; from render import Renderer; import cv2, torch
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
R=Renderer(m,K)
T=r['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@T
Ry=np.diag([-1,1,-1,1.])
msk=h5py.File('/work/inputs/masks.h5')
def ncc(a,b,w):
    if w.sum()<20: return np.nan
    a=a[w]-a[w].mean(); b=b[w]-b[w].mean()
    return (a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9)
F=len(T); res=np.full((F,2,3),np.nan)
SS=4
for i in range(0,F,6):
    o=msk['object'][i]; per=msk['person'][i]; ys,xs=np.nonzero(o)
    if len(xs)<50: continue
    x0,y0=xs.min()-25,ys.min()-25; w,h=xs.max()-x0+25,ys.max()-y0+25
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[...,::-1].astype(np.float32)/255
    g=im.mean(-1)
    cols,mks,_=R.render(np.stack([Tc[i],Tc[i]@Ry]),crop=(x0,y0,w,h),res=(h*SS,w*SS))
    for k in range(2):
        c=cv2.resize(cols[k].cpu().numpy(),(w,h),interpolation=cv2.INTER_AREA); mk=cv2.resize(mks[k].cpu().numpy(),(w,h),interpolation=cv2.INTER_AREA)
        gr=c.mean(-1)
        Gr=c[...,1]-0.5*(c[...,0]+c[...,2])
        logo=cv2.dilate(((gr>0.45)&(mk>0.9)).astype(np.uint8),np.ones((3,3)))>0
        # restrict logo to central part of board (exclude deck rim): central blob near centroid
        whl=cv2.dilate(((Gr>0.12)|((gr>0.3)&(np.abs(c[...,0]-c[...,2])<0.05)&~logo)).astype(np.uint8)*(mk>0.5),np.ones((3,3)))>0
        best=[-9,-9,-9]
        for dx in range(-4,5):
            for dy in range(-4,5):
                gi=g[y0+dy:y0+dy+h,x0+dx:x0+dx+w]; pi=per[y0+dy:y0+dy+h,x0+dx:x0+dx+w]
                for j,reg in enumerate([mk>0.5,logo,whl]):
                    v=ncc(gr,gi,reg&~pi)
                    if v>best[j]: best[j]=v
        res[i,k]=best
np.save('/work/scratch/an/regionncc.npy',res)
for a in range(0,F,30):
    b=min(a+30,F); s=res[a:b]; 
    print(a,b,' all',np.nanmean(s[:,:,0],0).round(3),' logo',np.nanmean(s[:,:,1],0).round(3),' wheels',np.nanmean(s[:,:,2],0).round(3))
