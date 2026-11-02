import sys, time; sys.path.insert(0,'/work/outputs')
from common import *; from fit import SilFitter, torch
K,Kp,Twc,rec,gp=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs')
s=0.5; Ks=K.copy(); Ks[:2]*=s; w,h=int(W*s),int(H*s)
d=np.load('/work/scratch/masks_half.npz'); obj=d['obj']; per=d['per']
Tb=cam_poses_from_world(Twc,rec['T_world_mesh'].astype(np.float64))
fz=SilFitter(mesh,Ks,w,h)
a,b=int(sys.argv[1]),int(sys.argv[2]); wr,wt=float(sys.argv[3]),float(sys.argv[4]); iters=int(sys.argv[5]) if len(sys.argv)>5 else 150
fr=np.arange(a,b); valid=~(per&~obj)
wts=np.zeros(len(fr)); wts[::2]=1
t0=time.time()
T,l=fz.fit(Tb[fr],obj[fr],valid[fr],shared=False,iters=iters,lr_r=0.005,lr_t=0.005,w_smooth_r=wr,w_smooth_t=wt,weights=wts)
print('time',time.time()-t0)
def iou(Ts_list,frames):
    out=[]
    for i,Tq in zip(frames,Ts_list):
        R,p=fz.split(Tq[None]); r=fz.render_soft(torch.tensor(R,dtype=torch.float32,device='cuda'),torch.tensor(p,dtype=torch.float32,device='cuda'))[0].detach().cpu().numpy()>0.5
        M=obj[i]; V=valid[i]; out.append((r&M&V).sum()/max(1,((r|M)&V).sum()))
    return np.array(out)
te=fr[1::2]; ib=iou(Tb[te],te); io=iou(T[1::2],te); itr=iou(T[::2],fr[::2]); ibtr=iou(Tb[fr[::2]],fr[::2])
print('heldout IoU base %.4f fit %.4f | train base %.4f fit %.4f'%(ib.mean(),io.mean(),ibtr.mean(),itr.mean()))
ang=rot_angles_deg(Tb[fr,:3,:3],T[:,:3,:3]); print('rot change vs base: med %.2f p90 %.2f max %.2f'%tuple(np.percentile(ang,[50,90,100])))
dp=(T[:,:3,3]-Tb[fr,:3,3]); print("dp mean",np.round(dp.mean(0),4),"absmed",np.round(np.median(np.abs(dp),0),4))
np.save(f'/work/scratch/track_{a}_{b}_{wr}_{wt}.npy',T)
