import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch, cv2
from hoilib import *
ep=Episode('/work/inputs'); rd=Renderer(ep.mesh)
a,b=int(sys.argv[1]),int(sys.argv[2])
frames=list(range(a,b,6))
Tm=np.load(f'Tm_{a}.npy')
tg=MaskTargets(ep,frames,np.repeat(Tm[None],len(frames),0),size=192)
offs=np.arange(0,90,10)
R0=ep.T_base[0,:3,:3]
tw0=twist_swing(R0.T@(ep.T_world_cam@Tm)[:3,:3])[0]
outs=[]
for o in offs:
    T0c=torch.tensor(axis_rot(Tm,o-tw0+(-90 if a>300 else 0))[None],device='cuda',dtype=torch.float32)
    w=torch.zeros(1,3,device='cuda',requires_grad=True); t=torch.zeros(1,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([w,t],lr=0.01)
    for it in range(120):
        T=compose(T0c,w,t,0.0,True).expand(len(frames),4,4)
        iou=soft_iou(render_masks(rd,T,ep.K,tg.crops,tg.size),tg)
        loss=1-iou.mean(); opt.zero_grad(); loss.backward(); opt.step()
    Tf=T[0].detach().cpu().numpy().astype(np.float64)
    tw,sw=twist_swing(R0.T@(ep.T_world_cam@Tf)[:3,:3])
    print('twist %.1f swing %.1f iou %.4f'%(tw,sw,iou.mean().item()))
    outs.append((iou.mean().item(),Tf,tw))
best=max(outs,key=lambda x:x[0])
# visual of median image + mask contours of best and of best+45
fr=ep.frames()[a:b:6]
crop=tg.crops[0]; x0,y0,s=crop
med=np.median(np.stack([cut(f,crop,384,cv2.INTER_CUBIC) for f in fr]),0).astype(np.uint8)[:,:,::-1].copy()
for (T,col) in [(best[1],(0,255,0)),(axis_rot(best[1],45),(0,0,255))]:
    m,_=rd.render(T[None],ep.K,crop,384); m=(m[0].cpu().numpy()>0.5).astype(np.uint8)
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(med,cs,-1,col,1)
om=cut(ep.obj_mask[frames[len(frames)//2]].astype(np.uint8),crop,384,cv2.INTER_NEAREST)
cs,_=cv2.findContours(om,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE); cv2.drawContours(med,cs,-1,(255,0,0),1)
cv2.imwrite(f'fitvis_{a}.png',med)
