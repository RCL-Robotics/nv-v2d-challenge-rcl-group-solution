import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py, torch
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
key=sys.argv[1]; frames=[int(x) for x in sys.argv[2].split(',')]
if key=='base': Tcm=Tcw@rec['T_world_mesh'].astype(np.float64)
else:
    Tcm=np.load(key)['T_world_mesh']; Tcm=Tcw@Tcm
m=h5py.File('/work/inputs/masks.h5','r')
tiles=[]
for i in frames:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    p=K@Tcm[i,:3,3]; u,v=p[:2]/p[2]
    S=200; x0=int(u-S/2); y0=int(v-S/2)
    col,mk,_=R.render(Tcm[i:i+1],crop=(x0,y0,S,S),scale=2)
    col=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=mk[0].cpu().numpy()
    crop=np.zeros((S,S,3),np.uint8)
    ys,xs=slice(max(y0,0),min(y0+S,1152)),slice(max(x0,0),min(x0+S,1536))
    crop[ys.start-y0:ys.stop-y0, xs.start-x0:xs.stop-x0]=img[ys,xs]
    crop=cv2.resize(crop,(2*S,2*S))
    om=np.zeros((S,S),np.uint8); om[ys.start-y0:ys.stop-y0, xs.start-x0:xs.stop-x0]=m['object'][i][ys,xs]
    om=cv2.resize(om,(2*S,2*S),interpolation=cv2.INTER_NEAREST)
    a=crop.copy(); b=crop.copy(); b[mk]=col[mk]
    cnt,_=cv2.findContours(om,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(a,cnt,-1,(0,255,0),1)
    cnt,_=cv2.findContours(mk.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(a,cnt,-1,(0,0,255),1)
    t=np.hstack([a,b]); cv2.putText(t,str(i),(5,25),0,0.8,(0,255,255),2); tiles.append(t)
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3], np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))
