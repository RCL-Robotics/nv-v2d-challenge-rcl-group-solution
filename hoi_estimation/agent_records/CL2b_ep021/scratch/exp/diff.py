import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
R=Renderer(mesh,K,W,H)
import h5py; fm=h5py.File('/work/inputs/masks.h5','r')
src=sys.argv[3] if len(sys.argv)>3 else 'base'
Tc=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64) if src=='base' else np.load(src)['T_world_mesh'])
tiles=[]
for i in [int(a) for a in sys.argv[1].split(',')]:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    m=R.render(Tc[i:i+1],need_color=False)['mask'][0].cpu().numpy()
    M=fm['object'][i]; P=fm['person'][i]
    ov=(img*0.4).astype(np.uint8)
    ov[M&m]=(255,255,255); ov[M&~m]=(0,0,255); ov[m&~M]=(255,0,0); ov[P&~M&~m]=ov[P&~M&~m]//2+(0,60,0)
    ys,xs=np.nonzero(m|M); cx,cy=int(xs.mean()),int(ys.mean()); hs=int(max(xs.ptp(),ys.ptp())*0.6)
    a=cv2.copyMakeBorder(ov,hs,hs,hs,hs,0)[cy:cy+2*hs,cx:cx+2*hs]
    t=cv2.resize(a,(450,450)); cv2.putText(t,str(i),(5,20),0,0.6,(0,255,255),2); tiles.append(t)
while len(tiles)%4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2],np.vstack([np.hstack(tiles[k:k+4]) for k in range(0,len(tiles),4)]))
