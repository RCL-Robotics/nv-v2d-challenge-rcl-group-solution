exec(open('/work/scratch/common_exp.py').read())
import time
fit=CropFitter('/work/inputs/object_mesh.glb',res=192)
ids=list(range(0,747,3))
t0=time.time()
fd=make_fd(fit,ids,Tb[ids])
print('prep',time.time()-t0)
R0,p0=fit.to_rp(Tb[ids])
with torch.no_grad():
    L0=losses(fit,torch.tensor(R0,dtype=torch.float32,device='cuda'),torch.tensor(p0,dtype=torch.float32,device='cuda'),fd,w_col=1.0)
t0=time.time()
R1,p1,L1=refine(fit,R0,p0,fd,iters=80)
print('refine',time.time()-t0, torch.cuda.max_memory_allocated()/1e9)
ang=rot_angle(R1,R0); dpp=np.linalg.norm(p1-p0,axis=1)
for j,i in enumerate(ids):
    if j%3==0: print(i,'iou %.2f->%.2f ncc %.2f->%.2f dR %.1f dp %.3f dz %.3f'%(L0['iou'][j],L1['iou'][j],L0['ncc'][j],L1['ncc'][j],ang[j],dpp[j],p1[j,2]-p0[j,2]))
print('mean iou',L0['iou'].mean().item(),L1['iou'].mean())
np.savez('/work/scratch/r1.npz',ids=ids,T=fit.to_T(R1,p1))
