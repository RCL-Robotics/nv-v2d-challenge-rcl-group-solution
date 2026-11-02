exec(open('/work/scratch/common_exp.py').read())
from scipy.spatial.transform import Rotation as Rot
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
import sys
a,b=int(sys.argv[1]),int(sys.argv[2])
ids=list(range(a,b,4))
Rb,pb=fit.to_rp(Tb[ids])
Rm0=Rot.from_matrix(Rb).mean().as_matrix(); pm0=np.median(pb,0)
T0=fit.to_T(Rm0[None],pm0[None])[0]
fd=make_fd(fit,ids,[T0]*len(ids))
hyps={f'rz{d}':Rm0@Rot.from_euler('z',d,degrees=True).as_matrix() for d in [0,-30,30,90,180,270]}
hyps['fx']=Rm0@Rot.from_euler('x',180,degrees=True).as_matrix(); hyps['fy']=Rm0@Rot.from_euler('y',180,degrees=True).as_matrix()
for name,Rh in hyps.items():
  for wc in [0.0]:
    R1,p1,L=refine_grouped(fit,Rh[None],pm0[None],fd,np.zeros(len(ids),int),iters=100,w_col=wc)
    print(name,'wc',wc,'iou %.3f chr %.3f dt %.4f'%(L['iou'].mean(),L['chr'].mean(),L['dt'].mean()),'dR_base %.1f'%rot_angle(R1[0],Rm0),'roll-ish: axis dev %.1f'%np.degrees(np.arccos(np.clip(R1[0][:,2]@Rm0[:,2],-1,1))),'dp',np.round(p1[0]-pm0,3))
    np.save(f'/work/scratch/static2_{a}_{name}_{wc}.npy',fit.to_T(R1,p1)[0])
with torch.no_grad():
    L=losses(fit,torch.tensor(Rb,dtype=torch.float32,device='cuda'),torch.tensor(pb,dtype=torch.float32,device='cuda'),fd,w_col=1)
print('baseline perframe iou %.3f chr %.3f'%(L['iou'].mean(),L['chr'].mean()))
