exec(open('/work/scratch/common_exp.py').read())
from scipy.spatial.transform import Rotation as Rot
import time
fit=CropFitter('/work/inputs/object_mesh.glb',res=128)
F=747; ids=np.arange(100,656)
Rb,pb=fit.to_rp(Tb); Rf,_=fit.to_rp(Tfp)
H={}
for src,Rs in [('b',Rb),('f',Rf)]:
    for d in [0,90,180,270]:
        H[f'{src}{d}']=Rs@Rot.from_euler('z',d,degrees=True).as_matrix()
res={}
t0=time.time()
for c0 in range(0,len(ids),140):
    cid=ids[c0:c0+140]
    fd=make_fd(fit,cid,Tb[cid],R=128)
    for k,Rh in H.items():
        R1,p1,L=refine(fit,Rh[cid],pb[cid],fd,iters=50,w_col=0.0,fix_depth=True,lr_r=0.03)
        r=res.setdefault(k,{'R':[],'p':[],'iou':[],'chr':[],'dt':[]})
        for q,v in [('R',R1),('p',p1),('iou',L['iou']),('chr',L['chr']),('dt',L['dt'])]: r[q].append(v)
    print(c0,time.time()-t0,flush=True)
out={'ids':ids}
for k,r in res.items():
    for q in r: out[f'{k}/{q}']=np.concatenate(r[q])
np.savez('/work/scratch/hyp2.npz',**out)
