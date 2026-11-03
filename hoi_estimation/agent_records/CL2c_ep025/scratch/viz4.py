# grid viz: python viz4.py out.jpg poses.npz frames [poses2.npz]  -> per frame: image | overlay1 [| overlay2]
import sys
from viz import *
ds=[np.load(a) for a in sys.argv[4:]] if len(sys.argv)>4 else []
d=np.load(sys.argv[2]); fr=[int(a) for a in sys.argv[3].split(',')]
tiles=[]
for f in fr:
    img=cv2.imread(f'/work/scratch/frames/{f:04d}.jpg'); S=120
    c=np.load('/work/scratch/maskstats.npz')['c'][f]
    x0=int(np.clip(c[0]-S/2,0,1536-S)); y0=int(np.clip(c[1]-S/2,0,1152-S))
    crop=img[y0:y0+S,x0:x0+S]; out=[crop]
    for dd in [d]+ds:
        T=np.eye(4); T[:3,:3]=dd['R'][f]; T[:3,3]=dd['t'][f]
        col,mm,_=R.render(T,K,S,S,x0,y0); col=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mm=mm[0].cpu().numpy()
        o=crop.copy(); o[mm]=col[mm]; out.append(o)
    p=cv2.resize(np.hstack(out),None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST); cv2.putText(p,str(f),(4,20),0,0.7,(0,0,255),2)
    tiles.append(np.pad(p,((2,2),(2,2),(0,0))))
nc=3 if len(ds)==0 else 2
while len(tiles)%nc: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[1],np.vstack([np.hstack(tiles[i:i+nc]) for i in range(0,len(tiles),nc)]))
