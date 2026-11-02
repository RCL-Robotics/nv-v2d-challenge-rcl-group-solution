from fitlib import *
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--lr',type=float,default=1.0); ap.add_argument('--lt',type=float,default=1.0)
ap.add_argument('--zs',type=float,default=1.0); ap.add_argument('--noshare',type=int,default=0); ap.add_argument('--fixrot',type=int,default=0); ap.add_argument('--dil',type=int,default=2); ap.add_argument('--tgt',default='sam'); ap.add_argument('--iters',type=int,default=300); ap.add_argument('--holdout',type=int,default=1); ap.add_argument('--out',default='an/joint.npz')
args=ap.parse_args()
torch.manual_seed(0)
cam,K,Twc,rec=load_inputs(); TM=Mesh(); SM=SilMesh(TM.tm)
d=np.load('an/targets.npz'); tg_np,valid_np=d['tg'],d['valid']
if args.tgt=='sam': tg_np=np.load('an/cropmasks.npz')['om']
import torch.nn.functional as Fn
def dil(s_):
    return s_ if args.dil==0 else Fn.max_pool2d(s_[:,None],2*args.dil+1,1,args.dil)[:,0]
F=len(tg_np)
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
S1=(0,134); S2=(598,F)   # provisional phase boundaries
group=np.arange(F); 
# param index per frame: static frames share
pid=np.zeros(F,int); k=0
for i in range(F):
    if args.noshare: pid[i]=2+i
    elif S1[0]<=i<S1[1]: pid[i]=0
    elif S2[0]<=i<S2[1]: pid[i]=1
    else: pid[i]=2+(i-S1[1])
NP=pid.max()+1
T0=np.zeros((NP,4,4))
if not args.noshare: T0[0]=mean_pose(Tc[S1[0]:S1[1]]); T0[1]=mean_pose(Tc[S2[0]:S2[1]])
for i in range(F):
    if pid[i]>=2: T0[pid[i]]=Tc[i]
T0[:,:3,3]*=args.zs
T0t=torch.tensor(T0,device=dev,dtype=torch.float32)
w=torch.zeros(NP,3,device=dev,requires_grad=True); t=torch.zeros(NP,3,device=dev,requires_grad=True)
opt=torch.optim.Adam([t] if args.fixrot else [w,t],lr=0.003)
tg=torch.tensor(tg_np,device=dev).float(); valid=torch.tensor(valid_np,device=dev).float()
use=np.ones(F,bool)
if args.holdout: use[1::2]=False   # odd frames held out for validation
pidt=torch.tensor(pid,device=dev)
tsc=torch.tensor([1,1,5.],device=dev)
CH=192
for it in range(args.iters):
    opt.zero_grad()
    Tall=pose_from(T0t,w,t*tsc)[pidt]       # (F,4,4)
    # smoothness: second differences of rotation (via relative rotvec approx) and translation
    R=Tall[:,:3,:3]; tr=Tall[:,:3,3]
    acc_t=tr[2:]-2*tr[1:-1]+tr[:-2]
    # rotation accel: compare R[i+1] with R[i] (R[i]^T R[i-1])^T  -> small; use matrix diff
    pred=R[1:-1]@R[:-2].transpose(1,2)@R[1:-1]
    acc_r=(R[2:]-pred).flatten(1)
    lsm=args.lr*1e3*(acc_r**2).sum()+args.lt*1e5*(acc_t[:,:2]**2).sum()+args.lt*1e4*(acc_t[:,2]**2).sum()
    lsm.backward()
    tot=0.
    idx=np.nonzero(use)[0]
    for c in range(0,len(idx),CH):
        ii=torch.tensor(idx[c:c+CH],device=dev)
        Tc_=pose_from(T0t,w,t*tsc)[pidt[ii]]
        sil,_=render_soft(SM,Tc_.contiguous(),K,X0,Y0,W,H)
        iou=soft_iou(dil(sil),tg[ii],valid[ii]); l=(1-iou).sum(); l.backward(); tot+=l.item()
    opt.step()
    if it%50==0 or it==args.iters-1: print(it,'sil',tot/len(idx),'smooth',lsm.item(),flush=True)
with torch.no_grad():
    Tall=pose_from(T0t,w,t*tsc)[pidt].cpu().numpy().astype(float)
    ious=[]
    for c in range(0,F,CH):
        sil,_=render_soft(SM,torch.tensor(Tall[c:c+CH],device=dev,dtype=torch.float32),K,X0,Y0,W,H)
        ious.append(soft_iou((dil(sil)>0.5).float(),tg[c:c+CH],valid[c:c+CH]).cpu().numpy())
    ious=np.concatenate(ious)
np.savez(args.out,T=Tall,iou=ious,pid=pid)
ho=~use if args.holdout else np.ones(F,bool)
print('heldout IoU mean %.4f  static1 %.4f carried %.4f static2 %.4f'%(ious[ho].mean(),ious[ho&(pid==0)].mean(),ious[ho&(pid>=2)].mean(),ious[ho&(pid==1)].mean()))
