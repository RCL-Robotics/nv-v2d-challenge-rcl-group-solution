# side-by-side: image crop | render (opaque over image) crop
import sys; sys.path.insert(0,'/work/outputs')
from common import *
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--poses',default='base'); ap.add_argument('--frames',default='300,350,400'); ap.add_argument('--out',default='/work/scratch/frames/sbs.png')
a=ap.parse_args()
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
V=np.load('/work/scratch/vid_half.npy',mmap_mode='r'); M=np.load('/work/scratch/mobj_half.npy',mmap_mode='r')
Tws=[]
for p in a.poses.split(','):
    Tws.append(D['rec']['T_world_mesh'].astype(np.float64) if p=='base' else np.load(p)['T_world_mesh'])
K=D['K'].copy(); K[:2]*=0.5
R=Renderer(mesh,K,576,768)
fr=[int(x) for x in a.frames.split(',')]
rows=[]
for f in fr:
    im=V[f].astype(np.float32)/255
    ys,xs=np.nonzero(M[f]); cy,cx=(int(np.median(ys)),int(np.median(xs))) if len(ys) else (288,384)
    y0=np.clip(cy-120,0,576-240); x0=np.clip(cx-160,0,768-320)
    tl=[im[y0:y0+240,x0:x0+320]]
    for Tw in Tws:
        Tc=np.linalg.inv(D['Twc'])@Tw[f]
        o=R.render(Tc[None]); rgb=o['rgb'][0].cpu().numpy(); m=o['mask'][0].cpu().numpy()
        out=im*0.3; out[m]=rgb[m]
        tl.append(out[y0:y0+240,x0:x0+320])
    t=(np.concatenate(tl,1)*255).astype(np.uint8).copy()
    cv2.putText(t,str(f),(5,20),0,0.6,(255,0,0),2)
    rows.append(t)
img=np.concatenate(rows,0)
img=cv2.resize(img,None,fx=1.5,fy=1.5)
cv2.imwrite(a.out,img[:,:,::-1])
