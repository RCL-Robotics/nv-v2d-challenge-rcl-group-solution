# per-frame silhouette refinement test
import sys; sys.path.insert(0,'/work/outputs')
from common import *
import time
torch.manual_seed(0)
K,Kp,Twc=load_camera('/work/inputs')
m=load_mesh('/work/inputs/object_mesh.glb'); r=Renderer(m,K)
f=h5py.File('/work/inputs/masks.h5')
rr=np.load('/work/inputs/reconstruction.npz'); Tw=rr['T_world_mesh'].astype(np.float64)
Tc0=np.linalg.inv(Twc)@Tw
S=384
def prep(idx, Tc):
    crops=[];tg=[];vd=[]
    for i in idx:
        om=f['object'][i]; pm=f['person'][i]
        mk=r.render(Tc[i][None],want_tex=False)[0][0].cpu().numpy()
        ys,xs=np.nonzero(om|mk)
        cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2; s=max(xs.max()-xs.min(),ys.max()-ys.min())*1.25+20
        crop=(cx-s/2,cy-s/2,s,s); crops.append(crop)
        M=np.array([[S/s,0,-(cx-s/2)*S/s],[0,S/s,-(cy-s/2)*S/s]])
        o=cv2.warpAffine(om.astype(np.float32),M,(S,S),flags=cv2.INTER_AREA if s>S else cv2.INTER_LINEAR)
        p=cv2.warpAffine(pm.astype(np.float32),M,(S,S))
        tg.append(o); vd.append(((p<0.5)|(o>0.5)).astype(np.float32))
    return crops, torch.tensor(np.array(tg)).cuda(), torch.tensor(np.array(vd)).cuda()
def refine(idx, Tc, iters=150, lr=(3e-3,3e-3)):
    crops,tg,vd=prep(idx,Tc)
    Ps=torch.tensor(np.array([proj_from_K(crop_K(K,c,S),S,S) for c in crops]),dtype=torch.float32).cuda()
    T0=torch.tensor(Tc[idx],dtype=torch.float32).cuda()
    B=len(idx)
    w=torch.zeros(B,3,device='cuda',requires_grad=True); t=torch.zeros(B,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':lr[0]},{'params':[t],'lr':lr[1]}])
    c0=T0[:,:3,3].clone()
    for it in range(iters):
        Rd=rotvec_to_mat(w)
        T=T0.clone()
        T=torch.cat([torch.cat([Rd@T0[:,:3,:3], (c0+t)[...,None]],2), T0[:,3:]],1)
        sil,_,_=render_batch(r,T,Ps,S)
        l=(((sil-tg)**2)*vd).sum((1,2))/tg.sum((1,2)).clamp(min=50)
        opt.zero_grad(); l.sum().backward(); opt.step()
    with torch.no_grad():
        inter=((sil>0.5)&(tg>0.5)&(vd>0)).sum((1,2)).float(); uni=(((sil>0.5)|(tg>0.5))&(vd>0)).sum((1,2)).float()
    return T.detach().cpu().numpy().astype(np.float64), (inter/uni.clamp(min=1)).cpu().numpy(), l.detach().cpu().numpy()
if __name__=='__main__':
    idx=[0,100,200,300,305,350,380,390,400,450,560,580,650,700]
    t0=time.time()
    _,iou0,_=refine(idx,Tc0,iters=1)
    Tn,iou1,l=refine(idx,Tc0)
    print(time.time()-t0)
    from scipy.spatial.transform import Rotation as R
    for k,i in enumerate(idx):
        dang=np.degrees(np.linalg.norm(R.from_matrix(Tn[k,:3,:3]@Tc0[i,:3,:3].T).as_rotvec()))
        print(i,'iou %.3f -> %.3f'%(iou0[k],iou1[k]),'dR %.1f'%dang,'dt',np.round(Tn[k,:3,3]-Tc0[i,:3,3],3))
    np.savez('ref_test.npz',T=Tn,idx=idx)
