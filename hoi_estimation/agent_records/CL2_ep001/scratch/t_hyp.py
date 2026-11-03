exec(open('/work/scratch/common_exp.py').read())
from scipy.spatial.transform import Rotation as Rot
import time
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
F=747; ids=np.arange(F)
Rb,pb=fit.to_rp(Tb); Rf,_=fit.to_rp(Tfp)
def flip(Rm,ax):
    e=np.zeros(3); e[ax]=np.pi; return Rm@Rot.from_rotvec(e).as_matrix()
H={'base':Rb,'fp':Rf,'base_fz':flip(Rb,2),'fp_fz':flip(Rf,2),'base_fy':flip(Rb,1)}
res={}
t0=time.time()
for c0 in range(0,F,125):
    cid=ids[c0:c0+125]
    fd=make_fd(fit,cid,Tb[cid])
    for k,Rh in H.items():
        R1,p1,L=refine(fit,Rh[cid],pb[cid],fd,iters=80,w_col=0.3,fix_depth=True)
        r=res.setdefault(k,{'R':[],'p':[],'iou':[],'ncc':[],'dt':[]})
        r['R'].append(R1); r['p'].append(p1); r['iou'].append(L['iou']); r['ncc'].append(L['ncc']); r['dt'].append(L['dt'])
    print(c0,time.time()-t0,flush=True)
out={}
for k,r in res.items():
    for q in r: out[f'{k}/{q}']=np.concatenate(r[q])
np.savez('/work/scratch/hyp.npz',**out)
