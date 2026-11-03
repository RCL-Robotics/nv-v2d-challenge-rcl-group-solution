from fitlib import *
cam,K,Twc,rec=load_inputs(); M=Mesh()
d=np.load('an/cropmasks.npz'); om,pm=d['om'],d['pm']
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n_w=np.array(pl[:3]); dpl=pl[3]
# plane in cam: n_w.(Twc p_c)+d = n_w.R p + n_w.t + d
n_c=Twc[:3,:3].T@n_w; d_c=n_w@Twc[:3,3]+dpl
V=M.tm.vertices
def floor_check(T):
    p=(T[:3,:3]@V.T).T+T[:3,3]; sd=p@n_c+d_c
    ax=T[:3,:3][:,0]  # mesh +x (down)
    return np.degrees(np.arccos(abs(ax@n_c))), np.percentile(sd,0.5)
def fit(frames, T0, fix_depth, iters=300):
    tg=torch.tensor(np.stack([greenmask_img(read_frame(i)[Y0:Y0+H,X0:X0+W],om[i]) for i in frames]),device=dev).float()
    valid=torch.tensor(~(pm[frames]&~om[frames]),device=dev).float()
    T0t=torch.tensor(T0,device=dev,dtype=torch.float32)
    w=torch.zeros(3,device=dev,requires_grad=True); t=torch.zeros(3,device=dev,requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=0.003)
    for it in range(iters):
        tt=t*1.0
        if fix_depth:
            # move only perpendicular to the ray through the object centre: scale lateral, keep z
            tt=torch.stack([t[0],t[1],torch.zeros_like(t[2])])
        T=pose_from(T0t,w,tt)
        sil,_=render_soft(M,T[None].expand(len(frames),4,4).contiguous(),K,X0,Y0,W,H)
        iou=soft_iou(sil,tg,valid); loss=(1-iou).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return pose_from(T0t,w,tt).detach().cpu().numpy().astype(float), iou.detach().cpu().numpy()
def eval_iou(frames,T):
    tg=torch.tensor(np.stack([greenmask_img(read_frame(i)[Y0:Y0+H,X0:X0+W],om[i]) for i in frames]),device=dev).float()
    valid=torch.tensor(~(pm[frames]&~om[frames]),device=dev).float()
    Tt=torch.tensor(T,device=dev,dtype=torch.float32)
    if Tt.ndim==2: Tt=Tt[None].expand(len(frames),4,4)
    sil,_=render_soft(M,Tt.contiguous(),K,X0,Y0,W,H)
    return soft_iou(sil,tg,valid).cpu().numpy()
def rdeg(A,B): return np.degrees(np.linalg.norm(Rot.from_matrix(A[:3,:3].T@B[:3,:3]).as_rotvec()))
for name,(a,b) in {'S1':(0,134),'S2':(598,748)}.items():
    fr=np.arange(a,b); trn=fr[::4]; hold=fr[2::4]
    Tm=mean_pose(Tc[fr])
    print(name,'baseline per-frame IoU(hold) %.3f'%eval_iou(hold,Tc[hold]).mean(),'mean-pose IoU %.3f'%eval_iou(hold,Tm).mean(),'floor',np.round(floor_check(Tm),3))
    for fd in [True,False]:
        T,_=fit(list(trn),Tm,fd)
        zs=np.median(Tc[fr,2,3])/T[2,3]; Ta=T.copy(); Ta[:3,3]*=zs
        print(' fix_depth',fd,'IoU hold %.3f'%eval_iou(hold,T).mean(),'anchored %.3f'%eval_iou(hold,Ta).mean(),'rot vs mean %.1f'%rdeg(T,Tm),'dz cm %.1f'%((T[2,3]-Tm[2,3])*100),'floor',np.round(floor_check(Ta),3))
        np.save(f'an/static_{name}_{int(fd)}.npy',Ta)
