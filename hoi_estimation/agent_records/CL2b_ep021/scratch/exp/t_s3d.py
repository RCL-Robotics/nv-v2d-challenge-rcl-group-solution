import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'.')
from common import *; from fit import SilFitter; from stages import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s; w,h=int(W*s),int(H*s)
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']; valid=~(per&~obj)
Tb=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
fz=SilFitter(mesh,Ks,w,h); Rb,pb=fz.split(Tb)
a,b=int(sys.argv[1]),int(sys.argv[2]); fr=np.arange(a,b)[::2]; te=np.arange(a,b)[1::2]
Rm=chordal_mean(Rb[a:b]); pm=np.median(pb[a:b],0)
print('heldout base per-frame %.4f'%fz.iou(Rb[te],pb[te],obj[te],valid[te]).mean())
for cfg in [dict(),dict(w_prior_r=3,w_prior_z=30,w_prior_uv=30)]:
  for it in [50,150,300]:
    Rg,pg,l=fz.fit(Rm[None],pm[None],np.zeros(len(fr),int),obj[fr],valid[fr],iters=it,lr_r=0.005,lr_t=0.0015,lr_z=0.001,**cfg)
    print(cfg,it,'%.4f'%l.mean(),'heldout %.4f'%fz.iou(np.repeat(Rg,len(te),0),np.repeat(pg,len(te),0),obj[te],valid[te]).mean(), '%.2f'%rot_angle_deg(Rm,Rg[0]), np.round(pg[0]-pm,4))
