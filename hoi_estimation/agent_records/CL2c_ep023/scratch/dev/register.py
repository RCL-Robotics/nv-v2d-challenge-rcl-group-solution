import sys, time; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, h5py, torch
from hoi_lib import *
torch.manual_seed(0)
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp); mesh=load_mesh(inp)
Tcw=np.linalg.inv(Twc); o=Twc[:3,3]
fm=FloorModel(mesh,plane,np.array([-0.28,-0.02,-0.96]),h0=0.014)
R=Renderer(mesh,K,1536,1152)
T=rec['T_world_mesh'].astype(np.float64); F=len(T)
a_,b_,c_=map(int,sys.argv[1].split(":")); ids=list(range(a_,b_,c_))
RES=256
init=np.array([fm.from_pose(T[i],o) for i in ids])
mf=h5py.File(inp+'/masks.h5')
frames=read_video(inp+'/video.mp4',ids)
v=np.asarray(mesh.vertices)
crops=[];Kcs=[];M=[];V=[];IM=[];BG=[]
bgp=np.load('/work/scratch/bg.npy')
for j,i in enumerate(ids):
    om=mf['object'][i]; pm=mf['person'][i]
    Pc=Tcw@fm.pose(*init[j]); pv=(v[::50]@Pc[:3,:3].T+Pc[:3,3]); uv=pv[:,:2]/pv[:,2:]*K[[0,1],[0,1]]+K[:2,2]
    lo,hi=uv.min(0),uv.max(0)
    ys,xs=np.nonzero(om)
    if len(xs)>50: lo=np.minimum(lo,[xs.min(),ys.min()]); hi=np.maximum(hi,[xs.max(),ys.max()])
    c=(lo+hi)/2; S=int(max(RES,1.4*(hi-lo).max())); x0,y0=int(c[0]-S/2),int(c[1]-S/2)
    r=RES/S
    Kc=np.array([[K[0,0]*r,0,(K[0,2]-x0+0.5)*r-0.5],[0,K[1,1]*r,(K[1,2]-y0+0.5)*r-0.5],[0,0,1]])
    def crop(a):
        pad=np.zeros((S,S)+a.shape[2:],a.dtype)
        xa,ya=max(0,x0),max(0,y0); xb,yb=min(a.shape[1],x0+S),min(a.shape[0],y0+S)
        if xb>xa and yb>ya: pad[ya-y0:yb-y0,xa-x0:xb-x0]=a[ya:yb,xa:xb]
        return cv2.resize(pad,(RES,RES),interpolation=cv2.INTER_AREA)
    inb=np.ones((1152,1536),np.float32)
    M.append(crop(om.astype(np.float32))); V.append(crop(((~pm)|om).astype(np.float32))*crop(inb)); IM.append(crop(frames[j].astype(np.float32)/255)); BG.append(crop(bgp.astype(np.float32)/255))
    Kcs.append(Kc)
M=torch.tensor(np.stack(M),device='cuda'); V=torch.tensor(np.stack(V),device='cuda'); IM=torch.tensor(np.stack(IM),device='cuda'); BG=torch.tensor(np.stack(BG),device='cuda'); Kcs=torch.tensor(np.stack(Kcs),dtype=torch.float32,device='cuda')
Tcw_t=torch.tensor(Tcw,dtype=torch.float32,device='cuda')
import os
FREEH=os.environ.get('FREEH')=='1'
MODE=sys.argv[2] if len(sys.argv)>2 else 'photo'
SIG=[1.2]
def blur(x):
    sg=SIG[0]; r=int(3*sg); gk=torch.tensor(cv2.getGaussianKernel(2*r+1,sg)[:,0],dtype=torch.float32,device='cuda')
    B,H,W,C=x.shape; y=x.permute(0,3,1,2).reshape(B*C,1,H,W)
    y=torch.nn.functional.conv2d(y,gk.view(1,1,-1,1),padding=(r,0)); y=torch.nn.functional.conv2d(y,gk.view(1,1,1,-1),padding=(0,r))
    return y.reshape(B,C,H,W).permute(0,2,3,1)
def loss_fn(th,a,b,sl,h=None):
    Tw=fm.pose_torch(th,a,b) if h is None else fm.pose_torch(th,a,b,0)+torch.nn.functional.pad((h[:,None]*torch.tensor(fm.n,dtype=torch.float32,device='cuda'))[:,:,None],(3,0,0,1)); Tc=Tcw_t@Tw
    al,rgb,_=render_crops(R,Tc,Kcs[sl],RES,tex=True)
    m,vv=M[sl],V[sl]
    if MODE.startswith('sil'):
        dl=int(MODE[3:] or 0)
        if dl>0: al=torch.nn.functional.max_pool2d(al[:,None],2*dl+1,1,dl)[:,0]
        inter=(al*m*vv).sum((1,2)); uni=((al+m-al*m)*vv).sum((1,2))
        return 1-inter/uni.clamp(min=1)
    I=IM[sl]; bg=BG[sl]
    with torch.no_grad():  # per-frame per-channel affine colour model of the rendered texture
        w=(al>0.5).float()*vv; n=w.sum((1,2)).clamp(min=1)[:,None]
        x=rgb; y=I
        mx=(x*w[...,None]).sum((1,2))/n; my=(y*w[...,None]).sum((1,2))/n
        vx=(((x-mx[:,None,None])**2)*w[...,None]).sum((1,2))/n; cxy=(((x-mx[:,None,None])*(y-my[:,None,None]))*w[...,None]).sum((1,2))/n
        gain=(cxy/(vx+1e-4)).clamp(0.3,3); bias=my-gain*mx
    comp=al[...,None]*(rgb*gain[:,None,None]+bias[:,None,None])+(1-al[...,None])*bg
    d=(blur(comp)-blur(I)).abs().sum(-1)
    return (d*vv).sum((1,2))/vv.sum((1,2)).clamp(min=1)
def optimize(th0,a0,b0,iters=120):
    out=[]
    for s0 in range(0,len(ids),64):
        sl=slice(s0,min(len(ids),s0+64))
        th=torch.tensor(th0[sl],dtype=torch.float32,device='cuda',requires_grad=True)
        a=torch.tensor(a0[sl],dtype=torch.float32,device='cuda',requires_grad=True)
        b=torch.tensor(b0[sl],dtype=torch.float32,device='cuda',requires_grad=True)
        h=torch.full_like(a,0.014,requires_grad=True) if FREEH else None
        opt=torch.optim.Adam([{'params':[th],'lr':0.02},{'params':[a,b],'lr':0.005}]+([{'params':[h],'lr':0.002}] if FREEH else []))
        for it in range(iters):
            SIG[0]=3.0 if it<iters//2 else 1.2
            L=loss_fn(th,a,b,sl,h); opt.zero_grad(); L.sum().backward(); opt.step()
        with torch.no_grad(): L=loss_fn(th,a,b,sl,h)
        out.append(np.stack([th.detach().cpu().numpy(),a.detach().cpu().numpy(),b.detach().cpu().numpy(),L.cpu().numpy()]+([h.detach().cpu().numpy()] if FREEH else []),1))
    return np.concatenate(out)
t=time.time()
with torch.no_grad():
    L0=torch.cat([loss_fn(*[torch.tensor(init[s:s+64,k],dtype=torch.float32,device='cuda') for k in range(3)],slice(s,s+64)) for s in range(0,len(ids),64)]).cpu().numpy()
res={}
OFFS=[0] if len(sys.argv)>3 else [0,np.pi/2,np.pi,-np.pi/2]
for off in OFFS:
    res[off]=optimize(init[:,0]+off,init[:,1],init[:,2])
print('time',time.time()-t)
np.savez(f'/work/scratch/reg_{MODE}_{a_}_{b_}.npz',ids=ids,init=init,L0=L0,**{f'r{k}':res[off] for k,off in enumerate(res)})
for j,i in enumerate(ids):
    print(i,'L0 %.3f'%L0[j],' '.join('%.3f'%res[off][j,3] for off in res),'dth %.1f'%np.degrees(res[0][j,0]-init[j,0]),'da %.1f db %.1f cm'%(100*(res[0][j,1]-init[j,1]),100*(res[0][j,2]-init[j,2])))
