import sys; sys.path.insert(0,'/work/outputs')
from common import *
import argparse
p=argparse.ArgumentParser(); p.add_argument('--poses',default='/work/inputs/reconstruction.npz'); p.add_argument('--key',default='T_world_mesh')
p.add_argument('--frames',default='0,100,200'); p.add_argument('--out',default='ov.jpg'); p.add_argument('--cam',default='world')
p.add_argument('--size',type=int,default=320); p.add_argument('--K',default='K')
a=p.parse_args()
K,Kp,Twc=load_camera('/work/inputs')
m=load_mesh('/work/inputs/object_mesh.glb')
T=np.load(a.poses)[a.key].astype(np.float64)
Tc=np.linalg.inv(Twc)@T if a.cam=='world' else T
KK=K if a.K=='K' else Kp
r=Renderer(m,KK)
f=h5py.File('/work/inputs/masks.h5')
frames=[int(x) for x in a.frames.split(',')]
tiles=[]
for i in frames:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    om=f['object'][i]
    mk,rgb,d,_=r.render(Tc[i][None])
    mk=mk[0].cpu().numpy(); rgb=(rgb[0].cpu().numpy()[:,:,::-1]*255).astype(np.uint8)
    ys,xs=np.nonzero(mk|om)
    if len(xs)==0: ys,xs=np.array([576]),np.array([768])
    cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2; s=max(xs.max()-xs.min(),ys.max()-ys.min())*1.3+40
    x0,y0=int(cx-s/2),int(cy-s/2); s=int(s)
    blend=img.copy(); blend[mk]=(0.5*img[mk]+0.5*rgb[mk]).astype(np.uint8)
    cs,_=cv2.findContours(om.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(blend,cs,-1,(0,255,0),1)
    cs,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(blend,cs,-1,(255,0,255),1)
    pad=cv2.copyMakeBorder(blend,s,s,s,s,cv2.BORDER_CONSTANT)
    raw=cv2.copyMakeBorder(img,s,s,s,s,cv2.BORDER_CONSTANT)
    t1=cv2.resize(pad[y0+s:y0+2*s,x0+s:x0+2*s],(a.size,a.size)); t0=cv2.resize(raw[y0+s:y0+2*s,x0+s:x0+2*s],(a.size,a.size))
    cv2.putText(t0,str(i),(5,25),0,0.8,(0,0,255),2)
    tiles.append(np.hstack([t0,t1]))
import os; cols=min(int(os.environ.get("COLS",3)),len(tiles))
while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
g=np.vstack([np.hstack(tiles[k:k+cols]) for k in range(0,len(tiles),cols)])
cv2.imwrite(a.out,g)
