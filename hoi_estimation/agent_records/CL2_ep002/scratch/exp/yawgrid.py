import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
from scipy.spatial.transform import Rotation as Rot
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
i=int(sys.argv[1]); T0=Tcw@rec['T_world_mesh'][i].astype(np.float64)
c=V.mean(0)*0  # rotate about mesh origin y axis
tiles=[]
img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
p=K@T0[:3,3]; u,v=p[:2]/p[2]; S=110; x0=int(u-S/2); y0=int(v-S/2)
tiles.append(cv2.resize(img[y0:y0+S,x0:x0+S],(330,330),interpolation=cv2.INTER_CUBIC))
for k in range(int(sys.argv[2]) if len(sys.argv)>2 else 11):
    a=k*360/11
    Ry=np.eye(4); Ry[:3,:3]=Rot.from_euler('y',a,degrees=True).as_matrix()
    T=T0@Ry
    col,mk,_=R.render(T[None],crop=(x0,y0,S,S),scale=3)
    col=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8)
    cv2.putText(col,f'{a:.0f}',(5,25),0,0.8,(0,255,255),2); tiles.append(col)
cv2.imwrite(sys.argv[3] if len(sys.argv)>3 else '/work/scratch/vis/yaw.jpg', np.vstack([np.hstack(tiles[k:k+6]) for k in range(0,12,6)]))
