from common import *
sys.path.insert(0,'/work/outputs')
from traj_opt import *
import time
d1=np.load('/work/scratch/motion.npy')[:,0]
phases=segment_phases(d1); print(phases)
crop=(690,545,128,112)
t0=time.time()
imgs=[frame(i) for i in range(553)]; oms=[OM[i] for i in range(553)]; pms=[PM[i] for i in range(553)]
print('load',time.time()-t0)
# init: baseline yaw (unwrapped), baseline uv of origin, baseline z
o=Tcm[:,:3,3]; uv=(o@K.T); uv=uv[:,:2]/uv[:,2:]
init_uvz=np.concatenate([uv,o[:,2:]],1)
opt=TrajectoryOptimizer(Rd,K,Rcw,B,crop,imgs,oms,pms,phases,yaw_base,init_uvz)
fix_z='fixz' in sys.argv
opt.optimize(iters=int(sys.argv[1]),fix_z=fix_z)
print('time',time.time()-t0)
T,th,uvz=opt.result()
iou,ncc=opt.eval_frames()
np.savez('/work/scratch/joint_%s.npz'%('fixz' if fix_z else 'free'),T=T,th=th,uvz=uvz,iou=iou,ncc=ncc,phases=np.array(phases))
for a,b,m in phases:
    print(a,b,m,'iou %.4f ncc %.4f'%(iou[a:b+1].mean(),ncc[a:b+1].mean()),'z med %.4f'%np.median(uvz[a:b+1,2]),'base z med %.4f'%np.median(Tcm[a:b+1,2,3]),
          'dyaw med %.1f'%np.degrees(np.median(th[a:b+1]-yaw_base[a:b+1])))
