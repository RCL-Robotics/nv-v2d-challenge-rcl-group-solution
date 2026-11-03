from common import *; from render import Renderer; import cv2, sys
r,cam,pl,m=load(); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
R=Renderer(m,K)
T=r['T_world_mesh'].astype(np.float64); Tc=np.linalg.inv(Twc)[None]@T
Ry=np.diag([-1,1,-1,1.])
ids=[int(x) for x in sys.argv[1].split(',')]
rows=[]
for i in ids:
    im=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    o=h5py.File('/work/inputs/masks.h5')['object'][i]; ys,xs=np.nonzero(o)
    x0,x1,y0,y1=xs.min()-10,xs.max()+10,ys.min()-10,ys.max()+10
    S=600/(x1-x0); sz=(600,int((y1-y0)*S))
    t=[cv2.resize(im[y0:y1,x0:x1],sz,interpolation=cv2.INTER_CUBIC)]
    for P in [Tc[i],Tc[i]@Ry]:
        c,mk,_=R.render(P,crop=(x0,y0,x1-x0,y1-y0),res=(sz[1],sz[0]))
        c=(c[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); t.append(c)
    rows.append(np.hstack(t))
cv2.imwrite('/work/scratch/viz/zoomflip.jpg',np.vstack([cv2.resize(x,(1800,int(x.shape[0]*1800/x.shape[1]))) for x in rows]))
