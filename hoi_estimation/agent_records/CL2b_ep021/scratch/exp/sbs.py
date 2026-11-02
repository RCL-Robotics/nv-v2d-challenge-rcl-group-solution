import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
R=Renderer(mesh,K,W,H,tex_scale=0.5)
src=sys.argv[3] if len(sys.argv)>3 else 'base'
if src=='base': Tc=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
elif src in ('fp','coco'):
    Tc=rec['T_cam_mesh_foundationpose' if src=='fp' else 'T_cam_mesh_coconet'].astype(np.float64); R.set_K(Kp)
else: Tc=cam_poses_from_world(Twc,np.load(src)['T_world_mesh'])
frames=[int(a) for a in sys.argv[1].split(',')]
tiles=[]
for i in frames:
    img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')
    o=R.render(Tc[i:i+1]); m=o['mask'][0].cpu().numpy(); c=(o['color'][0].cpu().numpy()[...,::-1]*255).astype(np.uint8)
    ys,xs=np.nonzero(m); cx,cy=int(xs.mean()),int(ys.mean()); hs=int(max(xs.ptp(),ys.ptp())*0.65)
    def crop(a):
        a=cv2.copyMakeBorder(a,hs,hs,hs,hs,0); return cv2.resize(a[cy:cy+2*hs,cx:cx+2*hs],(300,300))
    e=cv2.Canny(m.astype(np.uint8)*255,50,100)>0
    ov=img.copy(); ov[e]=(0,255,0)
    t=np.hstack([crop(ov),crop(c)]); cv2.putText(t,str(i),(5,20),0,0.6,(0,0,255),2); tiles.append(t)
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
g=np.vstack([np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)])
cv2.imwrite(sys.argv[2],g)
