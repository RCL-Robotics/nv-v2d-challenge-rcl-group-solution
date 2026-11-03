import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
g,V,F,UV,TEX=load_mesh('/work/inputs')
R=Renderer(V,F,UV,TEX,K,1152,1536)
Tcw=np.linalg.inv(Twc)
AXC=np.array([0.004,0,0.003])
def wyaw(R): x=R[:,0]; return np.degrees(np.arctan2(x[2],x[0]))
def yawrot(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]]; M[:3,3]=AXC-M[:3,:3]@AXC; return M
i=int(sys.argv[1]); yaws=[float(x) for x in sys.argv[2].split(',')]
T=np.load('/work/scratch/fitA.npz')['T_world_mesh'][i]
S=int(sys.argv[4]) if len(sys.argv)>4 else 80
p=K@(Tcw@T)[:3,3]; u,v=p[:2]/p[2]; x0=int(u-S/2); y0=int(v-S/2)
img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[y0:y0+S,x0:x0+S]
Z=320
tiles=[cv2.resize(img,(Z,Z),interpolation=cv2.INTER_CUBIC)]
for ay in yaws:
    y0w=wyaw(T[:3,:3]); dl=np.radians(ay-y0w); Tn=T@yawrot(-dl)
    if abs(((wyaw(Tn[:3,:3])-ay+180)%360)-180)>4: Tn=T@yawrot(dl)
    col,mk,_=R.render((Tcw@Tn)[None],crop=(x0,y0,S,S),scale=4)
    c=(col[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); m=mk[0].cpu().numpy()
    big=cv2.resize(img,(4*S,4*S),interpolation=cv2.INTER_CUBIC); big[m]=c[m]
    t=cv2.resize(big,(Z,Z),interpolation=cv2.INTER_AREA); cv2.putText(t,f'{ay:.0f}',(5,25),0,0.8,(0,255,255),2); tiles.append(t)
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[3],np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]))
