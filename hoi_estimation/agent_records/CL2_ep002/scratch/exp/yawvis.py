import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
AX=np.array([0.004,0,0.003])
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]
    M[:3,3]=AX-M[:3,:3]@AX; return M
i=int(sys.argv[1]); offs=[float(x) for x in sys.argv[2].split(',')]
T0=Tcw@rec['T_world_mesh'][i].astype(np.float64)
if len(sys.argv)>4: T0=Tcw@np.load(sys.argv[4])['T_world_mesh'][i]
S=int(sys.argv[5]) if len(sys.argv)>5 else 96; p=K@T0[:3,3]; u,v=p[:2]/p[2]; x0=int(u-S/2); y0=int(v-S/2)
img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+S,x0:x0+S]
tiles=[cv2.resize(img,(300,300),interpolation=cv2.INTER_CUBIC)]
for a in offs:
    col,mk,_=R.render((T0@yawrot(np.radians(a)))[None],crop=(x0,y0,S,S),scale=4)
    c=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mk=mk[0].cpu().numpy()
    c=cv2.resize(c,(S,S),interpolation=cv2.INTER_AREA); mk=cv2.resize(mk.astype(np.uint8),(S,S),interpolation=cv2.INTER_AREA)>0
    b=img.copy(); b[mk]=c[mk]
    b=cv2.resize(b,(300,300),interpolation=cv2.INTER_CUBIC)
    cv2.putText(b,f'{a:.0f}',(5,25),0,0.8,(0,255,255),2); tiles.append(b)
while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3], np.vstack([np.hstack(tiles[k:k+4]) for k in range(0,len(tiles),4)]))
