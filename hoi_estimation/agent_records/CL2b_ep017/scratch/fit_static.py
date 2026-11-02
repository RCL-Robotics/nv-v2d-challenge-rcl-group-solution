import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
a,b=int(sys.argv[1]),int(sys.argv[2])
frames=list(range(a,b,4))
Tmed=ep.Tc_base[frames[len(frames)//2]]
Tm=np.median(ep.Tc_base[frames][:,:3,3],0); Tmed=Tmed.copy(); Tmed[:3,3]=Tm
tg=MaskTargets(ep,frames,np.repeat(Tmed[None],len(frames),0),size=160)
res=[]
for th0 in range(-45,46,15):
    T0=torch.tensor(axis_rot(Tmed,th0),device='cuda',dtype=torch.float32)
    w=torch.zeros(3,device='cuda',requires_grad=True); t=torch.zeros(3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=0.01)
    for it in range(150):
        R=so3_exp(w[None])[0]
        T=T0.clone(); 
        # rotate about the object's centre in camera frame
        cc=T0[:3,:3]@torch.tensor(AXIS_C+[0,0,0.09],device='cuda',dtype=torch.float32)+T0[:3,3]
        Rn=R@T0[:3,:3]; tn=R@(T0[:3,3]-cc)+cc+t*torch.tensor([1,1,float(sys.argv[3])],device='cuda')
        T=torch.cat([torch.cat([Rn,tn[:,None]],1),T0[3:]],0)
        m=render_masks(rd,T[None].expand(len(frames),4,4),ep.K,tg.crops,tg.size)
        iou=soft_iou(m,tg); loss=1-iou.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it==100:
            for g in opt.param_groups: g['lr']=0.003
    Tf=T.detach().cpu().numpy().astype(np.float64)
    tw,sw=twist_swing(Tmed[:3,:3].T@Tf[:3,:3])
    print('init',th0,'iou %.4f'%iou.mean().item(),'twist %.1f swing %.1f'%(tw,sw),'dt(cm)',np.round((Tf[:3,3]-Tmed[:3,3])*100,2))
    res.append((iou.mean().item(),Tf))
np.save(f'static_{a}_{b}.npy',max(res,key=lambda x:x[0])[1])
