import sys, time; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
torch.manual_seed(0); dev='cuda'
S=setup(); pm=S['pm']; par=S['par_base']
t0=time.time()
E=Evidence(S); print('evidence',time.time()-t0, flush=True)
np.save('occ.npy',E.OCC)
sr=SilRenderer(decimate(S['mesh']),S['K'],S['Twc'],scale=0.5)
frames=list(range(len(par))) if sys.argv[1]=="all" else [int(x) for x in sys.argv[1].split(",")]
NG=120; ths=np.arange(NG)*2*np.pi/NG
out_cost=np.zeros((len(par),NG,5),np.float32)
t0=time.time()
for fi in frames:
    M,V,I=E.frame(fi)
    th=torch.tensor(ths,dtype=torch.float32,device=dev)
    a=torch.full((NG,),par[fi,1],device=dev,requires_grad=True); b=torch.full((NG,),par[fi,2],device=dev,requires_grad=True)
    opt=torch.optim.Adam([a,b],lr=0.004)
    for it in range(40):
        o=sr.render(pm.torch_pose(th,a,b,dev))
        L,_,_=Evidence.cost(o,M,V,I,blur=max(3.0*(1-it/25),0))
        opt.zero_grad(); L.sum().backward(); opt.step()
    with torch.no_grad():
        o=sr.render(pm.torch_pose(th,a,b,dev)); L,sil,fb=Evidence.cost(o,M,V,I)
    out_cost[fi]=np.stack([L.cpu().numpy(),sil.cpu().numpy(),fb.cpu().numpy(),a.detach().cpu().numpy(),b.detach().cpu().numpy()],1)
    if fi%25==0: print(fi,time.time()-t0,flush=True)
np.save(sys.argv[2] if len(sys.argv)>2 else 'grid_cost.npy',out_cost)
print('time',time.time()-t0)
