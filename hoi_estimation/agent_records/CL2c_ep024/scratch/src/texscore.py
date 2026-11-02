import sys; sys.path.insert(0,'/work/scratch/src')
from fitlib import *
cam,K,Twc,rec=load_inputs(); M=Mesh()
d=np.load('an/targets.npz'); tg_np,valid_np=d['tg'],d['valid']
S=3
def hp(g):  # high-pass grayscale
    g=g.astype(np.float32)
    return cv2.GaussianBlur(g,(0,0),1.0)-cv2.GaussianBlur(g,(0,0),4.0)
_imcache={}
def img_hp(i):
    if i not in _imcache:
        im=read_frame(i)[Y0:Y0+H,X0:X0+W]; im=cv2.resize(im,None,fx=S,fy=S,interpolation=cv2.INTER_CUBIC)
        _imcache[i]=hp(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY))
    return _imcache[i]
def fit_t(R,T0,frames,iters=150):
    T0=T0.copy(); T0[:3,:3]=R
    T0t=torch.tensor(T0,device=dev,dtype=torch.float32)
    t=torch.zeros(3,device=dev,requires_grad=True); opt=torch.optim.Adam([t],lr=0.003)
    tg=torch.tensor(tg_np[frames],device=dev).float(); va=torch.tensor(valid_np[frames],device=dev).float()
    for it in range(iters):
        T=pose_from(T0t,torch.zeros(3,device=dev),t*torch.tensor([1,1,5.],device=dev))
        sil,_=render_soft(M,T[None].expand(len(frames),4,4).contiguous(),K,X0,Y0,W,H)
        iou=soft_iou(sil,tg,va); l=(1-iou).mean(); opt.zero_grad(); l.backward(); opt.step()
    return T.detach().cpu().numpy().astype(float), iou.mean().item()
def tex_score(T,frames):
    Tt=torch.tensor(np.broadcast_to(T,(len(frames),4,4)).copy() if T.ndim==2 else T,device=dev)
    m,rgb,_=M.render(Tt,K,X0,Y0,W,H,S)
    sc=[]
    for k,i in enumerate(frames):
        mm=m[k].cpu().numpy(); r=(rgb[k].cpu().numpy()[...,::-1]*255).astype(np.uint8)
        rh=hp(cv2.cvtColor(r,cv2.COLOR_BGR2GRAY))
        ih=img_hp(i)
        vv=cv2.resize(valid_np[i].astype(np.uint8),None,fx=S,fy=S,interpolation=cv2.INTER_NEAREST)>0
        reg=cv2.erode(mm.astype(np.uint8),np.ones((7,7),np.uint8))>0
        reg&=vv
        a=rh[reg]; b=ih[reg]
        a=a-a.mean(); b=b-b.mean()
        sc.append((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-6))
    return np.array(sc)
if __name__=='__main__':
    Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
    J=np.load('an/joint_zs0.92.npz')['T']; J2=np.load('an/joint_0.0001_0.001.npz')['T']
    for name,fr,Tm,cands in [('S1',list(range(0,100,6)),mean_pose(Tc[:134]),{'s125':'an/staticfree_S1_125.npy','s90':'an/staticfree_S1_90.npy'}),
                             ('S2',list(range(670,748,5)),mean_pose(Tc[598:]),{'s125':'an/staticfree_S2_125.npy','s90':'an/staticfree_S2_90.npy'})]:
        C={'basemean':Tm[:3,:3],'joint92':J[fr[0]][:3,:3],'joint_b':J2[fr[0]][:3,:3]}
        for k,v in cands.items(): C[k]=np.load(v)[:3,:3]
        for k,Rm in C.items():
            T,iou=fit_t(Rm,Tm*np.array([1,1,1,1.])[None]*1,fr)
            print(name,k,'iou %.3f'%iou,'tex %.4f'%tex_score(T,fr).mean())
        # also per-frame baseline
        print(name,'baseline per-frame tex %.4f'%tex_score(Tc[fr],fr).mean())
