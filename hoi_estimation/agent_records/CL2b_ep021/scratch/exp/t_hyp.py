import sys, time; sys.path.insert(0,'/work/outputs')
from common import *; from fit import SilFitter; from stages import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s; w,h=int(W*s),int(H*s)
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']; valid=~(per&~obj)
gray=np.load('/work/scratch/gray_half.npy')
st,segs,_=static_segments(gray,obj,per); print(segs)
Tb=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
fz=SilFitter(mesh,Ks,w,h)
Rb,pb=fz.split(Tb)
t0=time.time()
for a,b,f in segs:
    if not f: continue
    fr=np.arange(a,b)[::2]
    Rm=chordal_mean(Rb[a:b]); pm=np.median(pb[a:b],0)
    hyps=[('base',Rm)]+[(f'z{d}',Rm@rot_about_mesh_axis('z',d)) for d in (90,180,270)]+[('x180',Rm@rot_about_mesh_axis('x',180)),('y180',Rm@rot_about_mesh_axis('y',180))]
    res=[]
    for name,R0 in hyps:
        Rg,pg,l=fz.fit(R0[None],pm[None],np.zeros(len(fr),int),obj[fr],valid[fr],iters=150)
        res.append((name,l.mean(),rot_angle_deg(Rm,Rg[0]),np.round(pg[0]-pm,3)))
    print(a,b,[(n,'%.4f'%l,'%.1f'%da,dp) for n,l,da,dp in res])
print('time',time.time()-t0)
