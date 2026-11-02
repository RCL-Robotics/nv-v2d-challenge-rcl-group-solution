import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'.')
from common import *; from fit import SilFitter; from stages import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s; w,h=int(W*s),int(H*s)
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']; valid=~(per&~obj)
Tb=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
fz=SilFitter(mesh,Ks,w,h); Rb,pb=fz.split(Tb)
for a,b in [(483,510),(582,678),(317,375)]:
  fr=np.arange(a,b)[::2]; te=np.arange(a,b)[1::2]
  Rm=chordal_mean(Rb[a:b]); pm=np.median(pb[a:b],0)
  print(a,b,'heldout base per-frame %.4f'%fz.iou(Rb[te],pb[te],obj[te],valid[te]).mean())
  for cfg in [dict(sigma=(4,0.5)),dict(sigma=(4,0.5),lr_r=0.003,lr_t=0.001,lr_z=0.0005)]:
    for it in [100,200]:
      c=dict(lr_r=0.005,lr_t=0.0015,lr_z=0.001); c.update(cfg)
      Rg,pg,l=fz.fit(Rm[None],pm[None],np.zeros(len(fr),int),obj[fr],valid[fr],iters=it,**c)
      print(cfg,it,'heldout %.4f'%fz.iou(np.repeat(Rg,len(te),0),np.repeat(pg,len(te),0),obj[te],valid[te]).mean(), '%.2f'%rot_angle_deg(Rm,Rg[0]), np.round(pg[0]-pm,4))
