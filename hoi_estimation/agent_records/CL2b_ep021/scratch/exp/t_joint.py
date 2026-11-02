import sys, time; sys.path.insert(0,'/work/outputs'); sys.path.insert(0,'.')
from common import *; from fit import SilFitter; from stages import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s; w,h=int(W*s),int(H*s)
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']; valid=~(per&~obj)
gray=np.load('/work/scratch/gray_half.npy')
st,segs,_=static_segments(gray,obj,per)
Tb=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
fz=SilFitter(mesh,Ks,w,h); Rb,pb=fz.split(Tb)
F=len(Tb); group=np.zeros(F,int); R0=[];p0=[]
for a,b,f in segs:
    if f: group[a:b]=len(R0); R0.append(chordal_mean(Rb[a:b])); p0.append(np.median(pb[a:b],0))
    else:
        for i in range(a,b): group[i]=len(R0); R0.append(Rb[i]); p0.append(pb[i])
R0=np.array(R0); p0=np.array(p0)
wr,wt,iters=float(sys.argv[1]),float(sys.argv[2]),int(sys.argv[3]); tag=sys.argv[4]
wts=np.ones(F); wts[1::2]=0
t0=time.time()
Rg,pg,l=fz.fit(R0,p0,group,obj,valid,iters=iters,w_smooth_r=wr,w_smooth_t=wt,weights=wts)
print('time',time.time()-t0)
R=Rg[group]; p=pg[group]
np.savez(f'/work/scratch/joint_{tag}.npz',R=R,p=p)
ib=fz.iou(Rb,pb,obj,valid); io=fz.iou(R,p,obj,valid)
np.savez(f'/work/scratch/joint_{tag}_iou.npz',ib=ib,io=io)
for a,b,f in segs:
    te=np.arange(a,b); te=te[te%2==1]
    ang=rot_angles_deg(Rb[a:b],R[a:b])
    print(a,b,'S' if f else 'M','heldout IoU base %.4f new %.4f'%(ib[te].mean(),io[te].mean()),'rot chg med %.2f max %.2f'%(np.median(ang),ang.max()),'dz med %.3f'%np.median(p[a:b,2]-pb[a:b,2]))
