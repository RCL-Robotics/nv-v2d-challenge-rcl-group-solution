import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'.')
from common import *; from fit import SilFitter; from stages import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s; w,h=int(W*s),int(H*s)
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']; valid=~(per&~obj)
Tb=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
fz=SilFitter(mesh,Ks,w,h); Rb,pb=fz.split(Tb)
a,b=483,510; fr=np.arange(a,b)[::int(sys.argv[1])]
Rm=chordal_mean(Rb[a:b]); pm=np.median(pb[a:b],0)
print('IoU base per-frame',fz.iou(Rb[fr],pb[fr],obj[fr],valid[fr]).mean(),'mean pose',fz.iou(np.repeat(Rm[None],len(fr),0),np.repeat(pm[None],len(fr),0),obj[fr],valid[fr]).mean())
J=np.load('/work/scratch/joint_a.npz'); print('joint',fz.iou(J['R'][fr],J['p'][fr],obj[fr],valid[fr]).mean())
for it in [0,1,10,50,150]:
    Rg,pg,l=fz.fit(Rm[None],pm[None],np.zeros(len(fr),int),obj[fr],valid[fr],iters=max(it,1),lr_r=0.005 if it>0 else 0,lr_t=0.005 if it>0 else 0)
    print(it,l.mean(),fz.iou(np.repeat(Rg,len(fr),0),np.repeat(pg,len(fr),0),obj[fr],valid[fr]).mean(), rot_angle_deg(Rm,Rg[0]), pg[0]-pm)
