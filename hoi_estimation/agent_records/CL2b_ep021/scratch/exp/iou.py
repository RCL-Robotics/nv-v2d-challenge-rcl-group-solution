import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s; w,h=int(W*s),int(H*s)
R=Renderer(mesh,Ks,w,h)
obj,per=load_masks('/work/inputs',s)
np.savez_compressed('/work/scratch/masks_half.npz',obj=obj,per=per)
def ious(Tc,Rr):
    out=[]
    for i in range(0,len(Tc),32):
        m=Rr.render(Tc[i:i+32],need_color=False)['mask'].cpu().numpy()
        for j in range(len(m)):
            M=obj[i+j]; P=per[i+j]&~M; v=~P
            out.append(((m[j]&M&v).sum()/max(1,((m[j]|M)&v).sum()), M.sum(), (per[i+j]).sum()))
    return np.array(out)
Tb=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
a=ious(Tb,R)
Ksp=Kp.copy(); Ksp[:2]*=s; R.set_K(Ksp)
b=ious(rec['T_cam_mesh_foundationpose'].astype(np.float64),R)
c=ious(rec['T_cam_mesh_coconet'].astype(np.float64),R)
np.savez('/work/scratch/iou_base.npz',base=a,fp=b,coco=c)
for i in range(0,678,15): print(i, '%.3f %.3f %.3f'%(a[i,0],b[i,0],c[i,0]), int(a[i,1]), int(a[i,2]))
print('mean',a[:,0].mean(),b[:,0].mean(),c[:,0].mean())
