# per-frame silhouette + photometric refinement test
import sys; sys.path.insert(0,'/work/outputs')
from common import *
import time, torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
torch.manual_seed(0)
K,Kp,Twc=load_camera('/work/inputs')
m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
f=h5py.File('/work/inputs/masks.h5')
rr=np.load('/work/inputs/reconstruction.npz'); Tw=rr['T_world_mesh'].astype(np.float64)
Tc0=np.linalg.inv(Twc)@Tw
S=384
def gblur(x,s):
    k=int(3*s)*2+1; g=torch.exp(-(torch.arange(k,device=x.device)-k//2)**2/(2*s*s)); g=g/g.sum()
    C=x.shape[1]
    x=F.conv2d(x,g.view(1,1,1,k).repeat(C,1,1,1),padding=(0,k//2),groups=C)
    return F.conv2d(x,g.view(1,1,k,1).repeat(C,1,1,1),padding=(k//2,0),groups=C)
def prep(idx, Tc):
    crops=[];tg=[];vd=[];ims=[]
    for i in idx:
        om=f['object'][i]; pm=f['person'][i]
        img=cv2.imread(f'/work/scratch/frames/{i:04d}.jpg')[:,:,::-1].astype(np.float32)/255
        mk=r.render(Tc[i][None],want_tex=False)[0][0].cpu().numpy()
        ys,xs=np.nonzero(om|mk)
        cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2; s=max(xs.max()-xs.min(),ys.max()-ys.min())*1.25+20
        crop=(cx-s/2,cy-s/2,s,s); crops.append(crop)
        M=np.array([[S/s,0,-(cx-s/2)*S/s],[0,S/s,-(cy-s/2)*S/s]])
        fl=cv2.INTER_AREA if s>S else cv2.INTER_LINEAR
        o=cv2.warpAffine(om.astype(np.float32),M,(S,S),flags=fl)
        p=cv2.warpAffine(pm.astype(np.float32),M,(S,S))
        ims.append(cv2.warpAffine(img,M,(S,S),flags=fl))
        tg.append(o); vd.append(((p<0.5)|(o>0.5)).astype(np.float32))
    return crops, torch.tensor(np.array(tg)).cuda(), torch.tensor(np.array(vd)).cuda(), torch.tensor(np.array(ims)).cuda().permute(0,3,1,2)
def photo_loss(rgb, sil, img, vd, s=2.0):
    # rgb (B,S,S,3) render, img (B,3,S,S). local-normalised comparison inside rendered & visible region
    x=rgb.permute(0,3,1,2); w=(sil>0.5).float()*vd; w=w[:,None]
    def norm(a):
        a=gblur(a,s)
        mu=(a*w).sum((2,3),keepdim=True)/w.sum((2,3),keepdim=True).clamp(min=1)
        sd=torch.sqrt((((a-mu)**2)*w).sum((2,3),keepdim=True)/w.sum((2,3),keepdim=True).clamp(min=1)+1e-6)
        return (a-mu)/sd
    d=((norm(x)-norm(img))**2*w).sum((1,2,3))/w.sum((1,2,3)).clamp(min=1)/3
    return d
def refine(idx, Tc, iters=150, lr=(3e-3,3e-3), lam=0.3, shared=False):
    crops,tg,vd,img=prep(idx,Tc)
    Ps=torch.tensor(np.array([proj_from_K(crop_K(K,c,S),S,S) for c in crops]),dtype=torch.float32).cuda()
    T0=torch.tensor(Tc[idx],dtype=torch.float32).cuda()
    B=len(idx); nb=1 if shared else B
    w=torch.zeros(nb,3,device='cuda',requires_grad=True); t=torch.zeros(nb,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':lr[0]},{'params':[t],'lr':lr[1]}])
    c0=T0[:,:3,3].clone()
    for it in range(iters):
        Rd=rotvec_to_mat(w).expand(B,3,3)
        T=torch.cat([torch.cat([Rd@T0[:,:3,:3], (c0+t)[...,None]],2), T0[:,3:]],1)
        sil,rgb,_=render_batch(r,T,Ps,S,with_tex=True)
        ls=(((sil-tg)**2)*vd).sum((1,2))/tg.sum((1,2)).clamp(min=50)
        lp=photo_loss(rgb,sil,img,vd)
        l=ls+lam*lp
        opt.zero_grad(); l.sum().backward(); opt.step()
    with torch.no_grad():
        inter=((sil>0.5)&(tg>0.5)&(vd>0)).sum((1,2)).float(); uni=(((sil>0.5)|(tg>0.5))&(vd>0)).sum((1,2)).float()
    return T.detach().cpu().numpy().astype(np.float64), (inter/uni.clamp(min=1)).cpu().numpy(), ls.detach().cpu().numpy(), lp.detach().cpu().numpy()
if __name__=='__main__':
    idx=[0,100,200,300,305,350,380,390,400,450,560,580,650,700]
    lam=float(sys.argv[1]) if len(sys.argv)>1 else 0.3
    _,iou0,ls0,lp0=refine(idx,Tc0,iters=1,lam=lam)
    Tn,iou1,ls1,lp1=refine(idx,Tc0,lam=lam)
    for k,i in enumerate(idx):
        dang=np.degrees(np.linalg.norm(R.from_matrix(Tn[k,:3,:3]@Tc0[i,:3,:3].T).as_rotvec()))
        print(i,'iou %.3f -> %.3f'%(iou0[k],iou1[k]),'photo %.3f -> %.3f'%(lp0[k],lp1[k]),'dR %.1f'%dang,'dt',np.round(Tn[k,:3,3]-Tc0[i,:3,3],3))
    T=Tw.copy()
    for k,i in enumerate(idx): T[i]=Twc@Tn[k]
    np.savez(f'ref2_world_{lam}.npz',T_world_mesh=T)
