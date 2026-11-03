import sys; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'.')
from common import *; from fit import SilFitter; from stages import *; from cmp import compare
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s; w,h=int(W*s),int(H*s)
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']; valid=~(per&~obj)
Tb=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
fz=SilFitter(mesh,Ks,w,h); Rb,pb=fz.split(Tb)
a,b=483,510; fr=np.arange(a,b)[::2]
Rm=chordal_mean(Rb[a:b]); pm=np.median(pb[a:b],0)
out={'raw':fz.join(Rm[None],pm[None])[0]}
for name,R0 in [('base',Rm),('z90',Rm@rot_about_mesh_axis('z',90))]:
    Rg,pg,l=fz.fit(R0[None],pm[None],np.zeros(len(fr),int),obj[fr],valid[fr],iters=150)
    out[name]=fz.join(Rg,pg)[0]
np.savez('/work/scratch/s3_hyp.npz',**out)
compare(mesh,K,out,[486,495,505],'/work/scratch/s3.jpg')
