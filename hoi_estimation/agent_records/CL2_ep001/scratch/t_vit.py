import numpy as np,sys; sys.path.insert(0,'/work/outputs')
from lib import rot_angle
h=np.load('/work/scratch/hyp2.npz'); ids=h['ids']
ks=[k[:-2] for k in h.files if k.endswith('/R')]
R=np.stack([np.nan_to_num(h[k+'/R']) for k in ks],1); P=np.stack([h[k+'/p'] for k in ks],1)
lc=float(sys.argv[1]) if len(sys.argv)>1 else 1.5; g=float(sys.argv[2]) if len(sys.argv)>2 else 0.02
U=np.stack([(1-h[k+'/iou'])+2*h[k+'/dt']+lc*h[k+'/chr'] for k in ks],1)
U[~np.isfinite(U)]=10
T,N=U.shape
D=np.zeros((T,N)); bp=np.zeros((T,N),int); D[0]=U[0]
for t in range(1,T):
    ang=rot_angle(R[t-1][:,None],R[t][None,:])  # (N,N)
    C=g*np.maximum(ang-3,0)
    tot=D[t-1][:,None]+C
    bp[t]=tot.argmin(0); D[t]=tot.min(0)+U[t]
path=np.zeros(T,int); path[-1]=D[-1].argmin()
for t in range(T-1,0,-1): path[t-1]=bp[t,path[t]]
Rs=R[np.arange(T),path]
np.savez('/work/scratch/vit.npz',ids=ids,R=Rs,p=P[np.arange(T),path],path=path,ks=np.array(ks))
step=rot_angle(Rs[1:],Rs[:-1])
print('mean unary chosen %.3f vs b0 %.3f'%(U[np.arange(T),path].mean(),U[:,0].mean()))
for t in range(0,T,6): print(ids[t],ks[path[t]],'U %.3f b0U %.3f'%(U[t,path[t]],U[t,0]),'ang_to_b0 %.0f'%rot_angle(Rs[t],R[t,0]),'step %.1f'%(step[t] if t<T-1 else 0))
