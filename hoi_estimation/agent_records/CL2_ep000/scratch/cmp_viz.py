import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
K,Twc,rec,mesh,plane=load_inputs('/work/inputs')
Tcw=np.linalg.inv(Twc)
Tb=np.einsum('ij,fjk->fik',Tcw,rec['T_world_mesh'].astype(float))
Tr=np.einsum('ij,fjk->fik',Tcw,np.load(sys.argv[1])['T_world_mesh'])
idx=[int(x) for x in sys.argv[2].split(',')]
rf=Renderer(mesh,K,1.0)
ims=read_video('/work/inputs/video.mp4',idx)
rows=[]
for j,i in enumerate(idx):
    tiles=[]
    for T,col in [(Tb,(0,0,255)),(Tr,(255,0,255))]:
        im=ims[j][:,:,::-1].copy()
        m,_,_=rf.render(T[i:i+1],need_rgb=False); m=m[0].cpu().numpy().astype(np.uint8)
        c,_=cv2.findContours(m,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE); cv2.drawContours(im,c,-1,col,1)
        ys,xs=np.where(m); cy,cx=int(ys.mean()),int(xs.mean())
        y0=np.clip(cy-200,0,1152-400); x0=np.clip(cx-200,0,1536-400)
        t=im[y0:y0+400,x0:x0+400]; cv2.putText(t,str(i),(5,20),0,0.7,(0,255,255),2); tiles.append(t)
    rows.append(np.hstack(tiles))
n=len(rows); half=(n+1)//2
while len(rows)<2*half: rows.append(np.zeros_like(rows[0]))
cv2.imwrite(sys.argv[3],np.hstack([np.vstack(rows[:half]),np.vstack(rows[half:])]))
