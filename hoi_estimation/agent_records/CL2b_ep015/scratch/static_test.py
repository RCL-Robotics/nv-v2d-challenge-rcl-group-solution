import sys; sys.path.insert(0,'/work/scratch'); sys.path.insert(0,'/work/outputs')
import refine2 as R2
from refine2 import *
def refine_fz(idx, Tc, iters=200, lam=0.3, shared=False, fixz=True, lr=(3e-3,2e-3)):
    crops,tg,vd,img=prep(idx,Tc)
    Ps=torch.tensor(np.array([proj_from_K(crop_K(K,c,S),S,S) for c in crops]),dtype=torch.float32).cuda()
    T0=torch.tensor(Tc[idx],dtype=torch.float32).cuda()
    B=len(idx); nb=1 if shared else B
    w=torch.zeros(nb,3,device='cuda',requires_grad=True); t=torch.zeros(nb,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':lr[0]},{'params':[t],'lr':lr[1]}])
    c0=T0[:,:3,3].clone(); msk=torch.tensor([1.,1.,0. if fixz else 1.],device='cuda')
    for it in range(iters):
        Rd=rotvec_to_mat(w).expand(B,3,3)
        T=torch.cat([torch.cat([Rd@T0[:,:3,:3], (c0+t*msk)[...,None]],2), T0[:,3:]],1)
        sil,rgb,_=render_batch(r,T,Ps,S,with_tex=True)
        ls=(((sil-tg)**2)*vd).sum((1,2))/tg.sum((1,2)).clamp(min=50)
        lp=photo_loss(rgb,sil,img,vd) if lam>0 else torch.zeros_like(ls)
        l=ls+lam*lp
        opt.zero_grad(); l.sum().backward(); opt.step()
    with torch.no_grad():
        inter=((sil>0.5)&(tg>0.5)&(vd>0)).sum((1,2)).float(); uni=(((sil>0.5)|(tg>0.5))&(vd>0)).sum((1,2)).float()
    return T.detach().cpu().numpy().astype(np.float64), (inter/uni.clamp(min=1)).cpu().numpy(), ls.detach().cpu().numpy(), lp.detach().cpu().numpy()
if __name__=='__main__':
    lam=float(sys.argv[1]); fixz=sys.argv[2]=='1'
    for rng in [range(0,160,8), range(630,715,6)]:
        idx=list(rng)
        Tn,iou,ls,lp=refine_fz(idx,Tc0,lam=lam,fixz=fixz)
        _,iou0,_,_=refine_fz(idx,Tc0,iters=1,lam=lam)
        # rotation deltas in world frame
        Rw=np.array([Twc[:3,:3]@Tn[k,:3,:3] for k in range(len(idx))]); Rb=Tw[idx[0],:3,:3]
        rv=np.array([R.from_matrix(Rw[k]@Rb.T).as_rotvec() for k in range(len(idx))])
        print('range',idx[0],idx[-1],'iou %.3f -> %.3f'%(iou0.mean(),iou.mean()))
        print(' delta rotvec(deg, world) mean',np.round(np.degrees(rv.mean(0)),2),'std',np.round(np.degrees(rv.std(0)),2))
        # euler decomposition of mean rotation wrt world y axis (vertical)
        Rm=R.from_rotvec(rv.mean(0)); print(' yaw(about world y) of mean delta %.2f'%np.degrees(Rm.as_rotvec()[1]))
        ts=Tn[:,:3,3]-Tc0[idx,:3,3]; print(' dt cam mean',np.round(ts.mean(0),3),'std',np.round(ts.std(0),3))
