import sys, time; sys.path.insert(0,'/work/outputs')
from common import *; from fit import SilFitter
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s; w,h=int(W*s),int(H*s)
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']
Tb=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
fz=SilFitter(mesh,Ks,w,h)
a,b=int(sys.argv[1]),int(sys.argv[2])
fr=np.arange(a,b); tr=fr[::2]; te=fr[1::2]
valid=~(per&~obj)
t0=time.time()
T,l=fz.fit(Tb[tr],obj[tr],valid[tr],shared=True,iters=200,lr_r=0.005,lr_t=0.005)
print('time',time.time()-t0)
Ts=T[0]
from fit import torch
def iou(Ts_list,frames):
    out=[]
    for i,Tq in zip(frames,Ts_list):
        R,p=fz.split(Tq[None]); r=fz.render_soft(torch.tensor(R,dtype=torch.float32,device='cuda'),torch.tensor(p,dtype=torch.float32,device='cuda'))[0].detach().cpu().numpy()>0.5
        M=obj[i]; V=valid[i]; out.append((r&M&V).sum()/((r|M)&V).sum())
    return np.array(out)
ib=iou(Tb[te],te); io=iou([Ts]*len(te),te)
# baseline mean pose
R,p=fz.split(Tb[fr]); U,_,Vt=np.linalg.svd(R.sum(0)); Tm=fz.join((U@Vt)[None],np.median(p,0)[None])[0]
im=iou([Tm]*len(te),te)
print('heldout IoU base %.4f basemean %.4f static-fit %.4f'%(ib.mean(),im.mean(),io.mean()))
print('rot change vs base mean %.2f deg, centre shift'%rot_angle_deg(Tm[:3,:3],Ts[:3,:3]), np.round((Ts[:3,3]+Ts[:3,:3]@fz.center)-(Tm[:3,3]+Tm[:3,:3]@fz.center),4))
print('base per-frame rot spread', np.percentile(rot_angles_deg(R,np.repeat((U@Vt)[None],len(R),0)),[50,90,100]))
np.save(f'/work/scratch/static_{a}_{b}.npy',Ts)
