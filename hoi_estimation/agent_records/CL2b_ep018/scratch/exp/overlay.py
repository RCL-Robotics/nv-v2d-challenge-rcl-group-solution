import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2
from hoi_lib import *
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
sc=0.5
R=Renderer(mesh,D['K'],D['W'],D['H'],scale=sc)
key=sys.argv[1] if len(sys.argv)>1 else 'base'
frames=list(range(0,858,60))
if len(sys.argv)>2: frames=[int(x) for x in sys.argv[2].split(',')]
if key=='base':
    T=np.linalg.inv(D['T_wc'])[None]@D['rec']['T_world_mesh'].astype(np.float64)
else:
    T=np.linalg.inv(D['T_wc'])[None]@np.load(key)['T_world_mesh']
imgs=read_video('/work/inputs/video.mp4',scale=sc,frames=set(frames))
om=read_masks('/work/inputs/masks.h5','object',sc,frames)
tiles=[]
for k,i in enumerate(frames):
    rgb,m,d,_=R.render(torch.tensor(T[i:i+1],dtype=torch.float32,device='cuda'))
    rgb=rgb[0].cpu().numpy(); m=m[0].cpu().numpy()
    im=imgs[k].astype(np.float32)/255
    out=im.copy(); out[m]=0.5*im[m]+0.5*rgb[m]
    cnt,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    out=(out*255).astype(np.uint8)[:,:,::-1].copy()
    cv2.drawContours(out,cnt,-1,(0,255,0),1)
    cnt2,_=cv2.findContours(om[k].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out,cnt2,-1,(0,0,255),1)
    ys,xs=np.where(m|om[k])
    if len(ys)==0: ys,xs=np.array([288]),np.array([384])
    cy,cx=int(ys.mean()),int(xs.mean())
    y0=np.clip(cy-120,0,out.shape[0]-240); x0=np.clip(cx-100,0,out.shape[1]-200)
    crop=out[y0:y0+240,x0:x0+200].copy()
    rr=(rgb*255).astype(np.uint8)[:,:,::-1].copy(); rr[~m]=(imgs[k][:,:,::-1][~m]*0.3).astype(np.uint8)
    crop=np.vstack([crop, rr[y0:y0+240,x0:x0+200]])
    cv2.putText(crop,str(i),(3,18),0,0.6,(255,255,0),2)
    tiles.append(crop)
while len(tiles)%8: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[r:r+8]) for r in range(0,len(tiles),8)]
cv2.imwrite(sys.argv[3] if len(sys.argv)>3 else '/work/scratch/exp/overlay.jpg',np.vstack(rows))
