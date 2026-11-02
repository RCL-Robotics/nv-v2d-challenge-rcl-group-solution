from fitlib import *
import torch.nn.functional as Fn
cam,K,Twc,rec=load_inputs(); SM=SilMesh(Mesh().tm)
d=np.load('an/targets.npz'); tg_np,valid_np=d['tg'],d['valid']
c=np.load('an/cropmasks.npz'); om=c['om']
Tc=np.linalg.inv(Twc)@rec['T_world_mesh'].astype(float)
def dil(sil,r):
    if r==0: return sil
    return Fn.max_pool2d(sil[:,None],2*r+1,1,r)[:,0]
for tname,TG in [('green',tg_np),('sam',om)]:
  for name,fr in [('S1',list(range(0,100,5))),('S2',list(range(640,748,5)))]:
    Tm=mean_pose(Tc[fr])
    for r in [0,1,2,3]:
        T0t=torch.tensor(Tm,device=dev,dtype=torch.float32)
        w=torch.zeros(3,device=dev,requires_grad=True); t=torch.zeros(3,device=dev,requires_grad=True)
        opt=torch.optim.Adam([w,t],lr=0.003)
        tg=torch.tensor(TG[fr],device=dev).float(); va=torch.tensor(valid_np[fr],device=dev).float()
        for it in range(300):
            T=pose_from(T0t,w,t*torch.tensor([1,1,5.],device=dev))
            sil,_=render_soft(SM,T[None].expand(len(fr),4,4).contiguous(),K,X0,Y0,W,H)
            iou=soft_iou(dil(sil,r),tg,va); l=(1-iou).mean(); opt.zero_grad(); l.backward(); opt.step()
        T=T.detach().cpu().numpy()
        print(tname,name,'dil',r,'iou %.3f'%iou.mean().item(),'z ratio %.3f'%(T[2,3]/Tm[2,3]),'rot %.1f'%np.degrees(np.linalg.norm(Rot.from_matrix(T[:3,:3]@Tm[:3,:3].T).as_rotvec())))
