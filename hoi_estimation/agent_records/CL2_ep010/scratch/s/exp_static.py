import sys; sys.path.insert(0,'/work/scratch/s')
from silopt import *
from scipy.spatial.transform import Rotation as Rot
torch.manual_seed(0)
r=Renderer(); K,Kp,Twc=load_cam(); sf=SilFit(r,K)
Tb=np.linalg.inv(Twc)@np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(np.float64)
a,b=int(sys.argv[1]),int(sys.argv[2])
fr=np.arange(a,b+1); tr=fr[fr%2==0]; te=fr[fr%2==1]
def mean_pose(Ts):
    T=np.eye(4); T[:3,:3]=Rot.from_matrix(Ts[:,:3,:3]).mean().as_matrix(); T[:3,3]=np.median(Ts[:,:3,3],0); return T
Tm=mean_pose(Tb[fr])
def iou(Tc, frames):
    M,V=load_masks(frames)
    out=[]
    for k in range(0,len(frames),16):
        T=torch.tensor(Tc[k:k+16],dtype=torch.float32,device='cuda')
        A=sf.sil(T[:,:3,:3],T[:,:3,3])>0.5
        m=M[k:k+16]>0.5; v=V[k:k+16]>0
        out+=list(((A&m&v).sum((1,2))/((A|m)&v).sum((1,2)).clamp_min(1)).cpu().numpy())
    return np.array(out)
def fit(T0, frames, iters=150, lr=2e-3):
    M,V=load_masks(frames)
    T0t=torch.tensor(T0[None],dtype=torch.float32,device='cuda')
    w=torch.zeros(1,3,device='cuda',requires_grad=True); t=torch.zeros(1,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=lr)
    for it in range(iters):
        opt.zero_grad(); L=0
        for k in range(0,len(frames),24):
            Rn,tn=sf.poses(T0t,w,t)
            A=sf.sil(Rn.expand(len(frames[k:k+24]),3,3),tn.expand(len(frames[k:k+24]),3))
            l=(((A-M[k:k+24])**2)*V[k:k+24]).sum()/M.sum()
            l.backward(); L+=l.item()
        opt.step()
        if it%50==0 or it==iters-1: print(it,L,w.detach().cpu().numpy(),t.detach().cpu().numpy())
    Rn,tn=sf.poses(T0t,w,t); T=np.eye(4); T[:3,:3]=Rn[0].detach().cpu().numpy(); T[:3,3]=tn[0].detach().cpu().numpy(); return T
Tf=fit(Tm,tr)
for name,P in [('baseline',Tb[te]),('mean',np.repeat(Tm[None],len(te),0)),('fit',np.repeat(Tf[None],len(te),0))]:
    v=iou(P,te); print(name, 'heldout IoU mean %.4f med %.4f min %.4f'%(v.mean(),np.median(v),v.min()))
d=np.linalg.inv(Tm)@Tf; print('fit vs mean: rot %.2f deg, trans'%np.degrees(Rot.from_matrix(d[:3,:3]).magnitude()), Tf[:3,3]-Tm[:3,3])
np.save(f'/work/scratch/static_{a}_{b}.npy',np.stack([Tm,Tf]))
