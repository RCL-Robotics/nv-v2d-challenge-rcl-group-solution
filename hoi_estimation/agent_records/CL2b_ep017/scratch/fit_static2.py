import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
a,b,zs=int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3])
frames=list(range(a,b,6))
Tm=ep.Tc_base[frames[len(frames)//2]].copy(); Tm[:3,3]=np.median(ep.Tc_base[frames][:,:3,3],0)
tg=MaskTargets(ep,frames,np.repeat(Tm[None],len(frames),0),size=192)
offs=np.arange(-45,45,5)
T0=np.array([axis_rot(Tm,o) for o in offs])
# shared pose per hypothesis: optimise each hypothesis jointly over all frames
res=[]
for h in range(len(offs)):
    T0c=torch.tensor(T0[h:h+1],device='cuda',dtype=torch.float32)
    w=torch.zeros(1,3,device='cuda',requires_grad=True); t=torch.zeros(1,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=0.01)
    for it in range(120):
        T=compose(T0c,w,t,zs).expand(len(frames),4,4)
        iou=soft_iou(render_masks(rd,T,ep.K,tg.crops,tg.size),tg)
        loss=1-iou.mean(); opt.zero_grad(); loss.backward(); opt.step()
        if it==80:
            for g in opt.param_groups: g['lr']=0.003
    Tf=T[0].detach().cpu().numpy().astype(np.float64)
    tw,sw=twist_swing(ep.T_base[0,:3,:3].T@(ep.T_world_cam@Tf)[:3,:3])
    print('off %4d -> twist %6.1f swing %4.1f iou %.4f dt(cm) %s'%(offs[h],tw,sw,iou.mean().item(),np.round((Tf[:3,3]-Tm[:3,3])*100,2)))
