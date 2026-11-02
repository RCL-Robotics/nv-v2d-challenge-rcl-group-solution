import sys, time; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
S=setup(); pm=S['pm']; dev='cuda'
E=Evidence(S); sr=SilRenderer(decimate(S['mesh']),S['K'],S['Twc'],scale=0.5)
r=np.load('exp4.npz'); th4,a4,b4,dh4=r['th'],r['a'],r['b'],float(r['dh'])
up=torch.tensor(pm.up,dtype=torch.float32,device=dev)
def pooled(frames, th0, a0, b0, free_dh, iters=120, ths=None):
    fr=E_frames=[E.frame(i) for i in frames]
    NH=len(ths) if ths is not None else 1
    th=torch.tensor(ths if ths is not None else [th0],dtype=torch.float32,device=dev).requires_grad_()
    a=torch.full((NH,),a0,device=dev,requires_grad=True); b=torch.full((NH,),b0,device=dev,requires_grad=True)
    dh=torch.full((NH,),dh4,device=dev,requires_grad=free_dh)
    ps=[{'params':[th],'lr':0.004},{'params':[a,b],'lr':0.002}]+([{'params':[dh],'lr':0.002}] if free_dh else [])
    opt=torch.optim.Adam(ps)
    for it in range(iters):
        T=pm.torch_pose(th,a,b,dev).clone(); T[:,:3,3]=T[:,:3,3]-dh[:,None]*up
        o=sr.render(T)  # NH renders, compare to every frame
        L=0
        for (M,V,I) in fr:
            l,_,_=Evidence.cost(o,M,V,I,blur=max(2*(1-it/60),0)); L=L+l
        L=L/len(fr); opt.zero_grad(); L.sum().backward(); opt.step()
    return th.detach().cpu().numpy(),a.detach().cpu().numpy(),b.detach().cpu().numpy(),dh.detach().cpu().numpy(),L.detach().cpu().numpy()
for nm,frames in [('S1',list(range(0,164,2))),('S2',list(range(469,546,2)))]:
    i0=frames[0]
    for free in [False,True]:
        th,a,b,dh,L=pooled(frames,th4[i0],a4[i0],b4[i0],free)
        print(nm,'free_dh',free,'yaw',np.degrees(th).round(2),'(exp4',np.degrees(th4[i0]).round(2),') a,b',a.round(3),b.round(3),'dh cm',(dh*100).round(1),'L',L.round(4))
