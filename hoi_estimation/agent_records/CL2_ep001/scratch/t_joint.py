exec(open('/work/scratch/common_exp.py').read())
from traj import *
from scipy.ndimage import median_filter, gaussian_filter1d
import time
holdout=int(sys.argv[1]); tag=sys.argv[2]
lam=float(sys.argv[3]) if len(sys.argv)>3 else 5.0
fit=CropFitter('/work/inputs/object_mesh.glb',res=160)
h=np.load('/work/scratch/hyp2.npz'); s=np.load('/work/scratch/hyp2_scores.npz'); hid=list(h['ids'])
ks=['b0','b90','b180','b270','f0','f90','f180','f270']
a,b=126,612
j=[hid.index(i) for i in range(a,b)]
U=np.stack([1-s[k][j,0]+2*s[k][j,1]+s[k][j,2]+s[k][j,3] for k in ks],1); U=np.nan_to_num(U,nan=9)
R=np.stack([np.nan_to_num(h[k+'/R'][j]) for k in ks],1); P=np.stack([h[k+'/p'][j] for k in ks],1)
path=viterbi(U,R,P)
print('viterbi choices',{k:int((path==i).sum()) for i,k in enumerate(ks)})
R0=R[np.arange(len(j)),path]; p0=P[np.arange(len(j)),path]
Rb,pb=fit.to_rp(Tb)
z=gaussian_filter1d(median_filter(pb[:,2],9,mode='nearest'),3,mode='nearest')
p0[:,2]=z[a:b]
S1=np.load('/work/scratch/static2_0_rz-30_0.0.npy'); S2=np.load('/work/scratch/static2_640_rz0_0.0.npy')
R1,p1=fit.to_rp(S1[None]); R2,p2=fit.to_rp(S2[None])
ids=list(range(a,b))
fd=make_fd(fit,ids,Tb[ids],R=160)
iw=np.ones(len(ids)); 
if holdout: iw[::5]=0
t0=time.time()
Rj,pj=joint_refine(fit,fd,R0,p0,iw,np.repeat(R1,2,0),np.repeat(p1,2,0),np.repeat(R2,2,0),np.repeat(p2,2,0),iters=150,lam_acc_r=lam,log=print,global_scale=bool(int(sys.argv[4])) if len(sys.argv)>4 else False)
print('time',time.time()-t0)
E_j=eval_frames(fit,fd,Rj,pj); E_b=eval_frames(fit,fd,Rb[a:b],pb[a:b]); E_v=eval_frames(fit,fd,R0,p0)
U_=lambda E:1-E[:,0]+2*E[:,1]+E[:,2]+E[:,3]
m=iw==0 if holdout else np.ones(len(ids),bool)
print('frames',m.sum(),'iou base %.3f vit %.3f joint %.3f'%(E_b[m,0].mean(),E_v[m,0].mean(),E_j[m,0].mean()),'U base %.3f vit %.3f joint %.3f'%(U_(E_b)[m].mean(),U_(E_v)[m].mean(),U_(E_j)[m].mean()))
np.savez(f'/work/scratch/joint_{tag}.npz',a=a,b=b,R=Rj,p=pj,E_j=E_j,E_b=E_b,E_v=E_v,iw=iw,R0=R0,p0=p0)
