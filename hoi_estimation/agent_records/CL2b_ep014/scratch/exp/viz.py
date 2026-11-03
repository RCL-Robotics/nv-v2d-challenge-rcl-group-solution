import sys; sys.path.insert(0,'/work/outputs')
from common import *
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--poses',default='base'); ap.add_argument('--frames',default='0:993:50'); ap.add_argument('--out',default='/work/scratch/frames/ov.png'); ap.add_argument('--crop',type=int,default=1)
a=ap.parse_args()
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
V=np.load('/work/scratch/vid_half.npy',mmap_mode='r'); M=np.load('/work/scratch/mobj_half.npy',mmap_mode='r')
if a.poses=='base': Tw=D['rec']['T_world_mesh'].astype(np.float64)
else: Tw=np.load(a.poses)['T_world_mesh']
Tc=np.linalg.inv(D['Twc'])@Tw
K=D['K'].copy(); K[:2]*=0.5
R=Renderer(mesh,K,576,768)
s,e,st=map(int,a.frames.split(':')); fr=list(range(s,e,st))
tiles=[]
for f in fr:
    o=R.render(Tc[f:f+1])
    rgb=o['rgb'][0].cpu().numpy(); m=o['mask'][0].cpu().numpy()
    im=V[f].astype(np.float32)/255
    out=im.copy(); out[m]=0.5*im[m]+0.5*rgb[m]
    cs,_=cv2.findContours(M[f].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    out=(out*255).astype(np.uint8).copy(); cv2.drawContours(out,cs,-1,(0,255,0),1)
    if a.crop:
        ys,xs=np.nonzero(M[f]|m)
        if len(ys):
            cy,cx=int(np.median(ys)),int(np.median(xs))
        else: cy,cx=288,384
        y0=np.clip(cy-120,0,576-240); x0=np.clip(cx-160,0,768-320)
        out=out[y0:y0+240,x0:x0+320]
    else: out=cv2.resize(out,(320,240))
    cv2.putText(out,str(f),(5,20),0,0.6,(255,0,0),2)
    tiles.append(out)
while len(tiles)%5: tiles.append(np.zeros_like(tiles[0]))
rows=[np.concatenate(tiles[i:i+5],1) for i in range(0,len(tiles),5)]
cv2.imwrite(a.out,np.concatenate(rows,0)[:,:,::-1])
