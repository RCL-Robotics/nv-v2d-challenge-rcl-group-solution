from fitlib import *
import fitlib
cam,K,Twc,rec=load_inputs(); M=Mesh()
d=np.load('an/cropmasks.npz'); om,pm=d['om'],d['pm']
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
def rdeg(A,B): return np.degrees(np.linalg.norm(Rot.from_matrix(A[:3,:3].T@B[:3,:3]).as_rotvec()))
def fit_batch(frames,T0,sth,iters=300,shared=True):
    tg=torch.tensor(np.stack([greenmask_img(read_frame(i)[Y0:Y0+H,X0:X0+W],om[i],sth) for i in frames]),device=dev).float()
    valid=torch.tensor(~(pm[frames]&~om[frames]),device=dev).float()
    n=1 if shared else len(frames)
    T0t=torch.tensor(np.broadcast_to(T0,(n,4,4)).copy(),device=dev,dtype=torch.float32)
    w=torch.zeros(n,3,device=dev,requires_grad=True); t=torch.zeros(n,3,device=dev,requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=0.003)
    for it in range(iters):
        T=pose_from(T0t,w,t*torch.tensor([1,1,5.],device=dev))
        sil,_=render_soft(M,T.expand(len(frames),4,4).contiguous(),K,X0,Y0,W,H)
        iou=soft_iou(sil,tg,valid); loss=(1-iou).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return T.detach().cpu().numpy().astype(float), iou.detach().cpu().numpy()
for name,(a,b) in {'S1':(0,134),'S2':(598,748)}.items():
    fr=np.arange(a,b); Tm=mean_pose(Tc[fr])
    for sth in [90,125,160]:
        T,iou=fit_batch(list(fr[::4]),Tm,sth); T=T[0]
        print(name,sth,'iou %.3f'%iou.mean(),'rot vs mean %.1f'%rdeg(T,Tm),'dz %.1f cm'%((T[2,3]-Tm[2,3])*100), 'rotvec(cam) deg',np.degrees(Rot.from_matrix(T[:3,:3]@Tm[:3,:3].T).as_rotvec()).round(1))
        np.save(f'an/staticfree_{name}_{sth}.npy',T)
    Ts,iou=fit_batch(list(fr[::6]),Tm,125,shared=False)
    rv=np.array([np.degrees(Rot.from_matrix(T[:3,:3]@Tm[:3,:3].T).as_rotvec()) for T in Ts])
    print(name,'per-frame fits: rotvec mean',rv.mean(0).round(1),'std',rv.std(0).round(1),'dz mean/std cm',((Ts[:,2,3]-Tm[2,3])*100).mean().round(1),((Ts[:,2,3]-Tm[2,3])*100).std().round(1))
    rb=np.array([np.degrees(Rot.from_matrix(T[:3,:3]@Tm[:3,:3].T).as_rotvec()) for T in Tc[fr]])
    print(name,'baseline per-frame rotvec vs mean std',rb.std(0).round(1),'z std cm',(Tc[fr,2,3].std()*100).round(1))
