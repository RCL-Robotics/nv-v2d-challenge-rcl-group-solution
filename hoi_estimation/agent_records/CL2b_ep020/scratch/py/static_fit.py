from common import *
import time
a,b,step=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]); fixz=len(sys.argv)>4 and sys.argv[4]=='fixz'
fr=list(range(a,b+1,step))
D=DiffFitter(Rd,K)
D.load_frames([frame(i) for i in fr],[OM[i] for i in fr],[PM[i] for i in fr])
th0=np.median(yaw_base[a:b+1])
tc0=np.median(Tcm[a:b+1,:3,3],0)
dev='cuda'
Bc=torch.tensor(Rcw@B,device=dev,dtype=torch.float32)
th=torch.tensor(th0,device=dev,dtype=torch.float32,requires_grad=True)
dt=torch.zeros(3,device=dev,requires_grad=True)
opt=torch.optim.Adam([{'params':[th],'lr':0.01},{'params':[dt],'lr':0.002}])
tc0t=torch.tensor(tc0,device=dev,dtype=torch.float32)
t0=time.time()
for it in range(150):
    Rcm=(Bc@rotz_t(th))[None].expand(len(fr),3,3)
    d=dt*torch.tensor([1,1,0. if fixz else 1.],device=dev)
    tcm=(tc0t+d)[None].expand(len(fr),3)
    iou,ncc=D.losses(Rcm,tcm,torch.arange(len(fr)))
    loss=-(iou.mean()+0.5*ncc.mean())
    opt.zero_grad(); loss.backward(); opt.step()
    if it%30==0 or it==149: print(it,'iou %.4f ncc %.4f dth %.2fdeg dt(mm) %s'%(iou.mean().item(),ncc.mean().item(),np.degrees(th.item()-th0),np.round(d.detach().cpu().numpy()*1000,1)))
print('time',time.time()-t0)
# baseline score with same metric (per-frame baseline poses)
with torch.no_grad():
    Rb=torch.tensor(Tcm[fr,:3,:3],device=dev,dtype=torch.float32); tb=torch.tensor(Tcm[fr,:3,3],device=dev,dtype=torch.float32)
    iou,ncc=D.losses(Rb,tb,torch.arange(len(fr)))
    print('baseline per-frame: iou %.4f ncc %.4f'%(iou.mean().item(),ncc.mean().item()))
c=tc0+d.detach().cpu().numpy()
print('fitted cam t',np.round(c,4),' baseline median',np.round(tc0,4),' baseline mean z %.4f'%Tcm[a:b+1,2,3].mean())
