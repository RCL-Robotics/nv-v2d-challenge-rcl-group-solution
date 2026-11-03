from common import *
# IoU-only pooled fit of (theta,x,y,z) per segment
def fit(a,b,step,w_ncc=0.0,iters=200):
    fr=list(range(a,b+1,step))
    D=DiffFitter(Rd,K); D.load_frames([frame(i) for i in fr],[OM[i] for i in fr],[PM[i] for i in fr])
    th0=np.median(yaw_base[a:b+1]); tc0=np.median(Tcm[a:b+1,:3,3],0)
    dev='cuda'; Bc=torch.tensor(Rcw@B,device=dev,dtype=torch.float32)
    th=torch.tensor(th0,device=dev,dtype=torch.float32,requires_grad=True)
    dt=torch.zeros(3,device=dev,requires_grad=True)
    opt=torch.optim.Adam([{'params':[th],'lr':0.005},{'params':[dt],'lr':0.001}])
    tc0t=torch.tensor(tc0,device=dev,dtype=torch.float32)
    for it in range(iters):
        Rcm=(Bc@rotz_t(th))[None].expand(len(fr),3,3); tcm=(tc0t+dt)[None].expand(len(fr),3)
        iou,ncc=D.losses(Rcm,tcm,torch.arange(len(fr)))
        loss=-(iou.mean()+w_ncc*ncc.mean()); opt.zero_grad(); loss.backward(); opt.step()
    c=(tc0t+dt).detach().cpu().numpy()
    print(a,b,'w_ncc',w_ncc,'iou %.4f'%iou.mean().item(),'dth %.1f'%np.degrees(th.item()-th0),'z %.4f (base med %.4f) diff %.1fcm'%(c[2],tc0[2],100*(c[2]-tc0[2])))
for seg in [(0,130,3),(140,208,3),(276,313,2),(465,552,3),(421,460,2)]:
    fit(*seg)
