exec(open('/work/scratch/common_exp.py').read())
from scipy.spatial.transform import Rotation as Rot
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
import sys
a,b=int(sys.argv[1]),int(sys.argv[2])
ids=list(range(a,b,4))
Rb,pb=fit.to_rp(Tb[ids])
# robust init: chordal mean rotation, median position
Rm0=Rot.from_matrix(Rb).mean().as_matrix(); pm0=np.median(pb,0)
T0=fit.to_T(Rm0[None],pm0[None])[0]
fd=make_fd(fit,ids,[T0]*len(ids))
hyps={'base':Rm0}
for ax in range(3):
    e=np.zeros(3); e[ax]=np.pi
    hyps[f'flip{"xyz"[ax]}']=Rm0@Rot.from_rotvec(e).as_matrix()   # 180 about mesh axis
for name,Rh in hyps.items():
    for wc in [0.0,0.5]:
        R1,p1,L=refine_grouped(fit,Rh[None],pm0[None],fd,np.zeros(len(ids),int),iters=100,w_col=wc)
        print(name,'wcol',wc,'iou %.3f ncc %.3f dt %.4f'%(L['iou'].mean(),L['ncc'].mean(),L['dt'].mean()),'dR_from_init %.1f'%rot_angle(R1[0],Rh),'dR_from_base %.1f'%rot_angle(R1[0],Rm0),'dp',np.round(p1[0]-pm0,3))
        np.save(f'/work/scratch/static_{a}_{name}_{wc}.npy',fit.to_T(R1,p1)[0])
# baseline per-frame for reference
with torch.no_grad():
    L=losses(fit,torch.tensor(Rb,dtype=torch.float32,device='cuda'),torch.tensor(pb,dtype=torch.float32,device='cuda'),fd,w_col=1)
print('baseline perframe iou %.3f ncc %.3f'%(L['iou'].mean(),L['ncc'].mean()), 'spread of baseline rot', np.percentile(rot_angle(Rb,Rm0[None]),[50,90,100]))
