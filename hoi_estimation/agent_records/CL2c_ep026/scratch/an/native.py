from common import *; from render import Renderer; import cv2, sys
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
R=Renderer(m,K)
T=r['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@T
Ry=np.diag([-1,1,-1,1.])
rows=[]
for i in [int(x) for x in sys.argv[1].split(',')]:
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    o=h5py.File('/work/inputs/masks.h5')['object'][i]; ys,xs=np.nonzero(o)
    x0,y0=xs.min()-15,ys.min()-15; w,h=xs.max()-x0+15,ys.max()-y0+15
    t=[im[y0:y0+h,x0:x0+w]]
    for P in [Tc[i],Tc[i]@Ry]:
        c,mk,_=R.render(P,crop=(x0,y0,w,h),res=(h*4,w*4))
        c=(c[0].cpu().numpy()[...,::-1]*255).astype(np.float32); mk=mk[0].cpu().numpy()[...,None]
        c=cv2.resize(c,(w,h),interpolation=cv2.INTER_AREA); mk=cv2.resize(mk,(w,h),interpolation=cv2.INTER_AREA)[...,None]
        c=c*mk+t[0]*(1-mk)
        t.append(c.astype(np.uint8))
    row=np.hstack(t); rows.append(cv2.resize(row,(row.shape[1]*5,row.shape[0]*5),interpolation=cv2.INTER_CUBIC))
W=max(x.shape[1] for x in rows)
cv2.imwrite('/work/scratch/viz/native.jpg',np.vstack([np.pad(x,((0,0),(0,W-x.shape[1]),(0,0))) for x in rows]))
