import sys, numpy as np, torch, cv2, json, h5py
sys.path.insert(0,'/work/outputs')
from hoilib import *
torch.manual_seed(0)
fr=np.load('/work/scratch/frames_all.npy',mmap_mode='r')
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
mesh=TexMesh('/work/inputs/object_mesh.glb')
H5=h5py.File('/work/inputs/masks.h5')
Tb=np.linalg.inv(Twc)[None]@np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
FY=np.diag([-1,1,-1,1.])

def crop_for(i, Tc):
    om=H5['object'][i]; ys,xs=np.nonzero(om)
    if len(xs)>20:
        cx,cy=(xs.min()+xs.max())/2,(ys.min()+ys.max())/2; s=max(xs.max()-xs.min(),ys.max()-ys.min())
    else:
        p=K@Tc[:3,3]; cx,cy=p[:2]/p[2]; s=80
    S=int(max(96,2.0*s)); x0=int(cx-S/2); y0=int(cy-S/2)
    return (x0,y0,S,S)

def getcrop(arr,crop):
    x0,y0,W,H=crop; out=np.zeros((H,W)+arr.shape[2:],arr.dtype)
    xa,ya=max(x0,0),max(y0,0); xb,yb=min(x0+W,arr.shape[1]),min(y0+H,arr.shape[0])
    if xb>xa and yb>ya: out[ya-y0:yb-y0,xa-x0:xb-x0]=arr[ya:yb,xa:xb]
    return out

def refine(i, Tinit, iters=150, lr_r=0.01, lr_t=0.005, fix_depth=False):
    crop=crop_for(i,Tinit[0])
    obs=torch.tensor(getcrop(H5['object'][i],crop),dtype=torch.float32,device='cuda')
    per=torch.tensor(getcrop(H5['person'][i],crop),dtype=torch.float32,device='cuda')
    valid=1-per*(1-obs)
    B=len(Tinit); T0=torch.tensor(Tinit,dtype=torch.float32,device='cuda')
    w=torch.zeros(B,3,device='cuda',requires_grad=True); dt=torch.zeros(B,3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':lr_r},{'params':[dt],'lr':lr_t}])
    norm=obs.sum().clamp(min=50)
    for it in range(iters):
        R=so3_exp(w)@T0[:,:3,:3]
        # rotate about object centre (translation part unchanged), translation delta in camera frame
        T=torch.zeros_like(T0); T[:,:3,:3]=R; T[:,:3,3]=T0[:,:3,3]+dt*torch.tensor([1,1,0. if fix_depth else 1],device='cuda'); T[:,3,3]=1
        sil,_=render_soft(mesh,T,K,crop)
        loss=((sil-obs)**2*valid).sum((1,2))/norm
        opt.zero_grad(); loss.sum().backward(); opt.step()
    with torch.no_grad():
        sil,rgb=render_soft(mesh,T,K,crop,need_rgb=True)
        hard=(sil>0.5).float()
        inter=(hard*obs*valid).sum((1,2)); uni=((hard+obs-hard*obs)*valid).sum((1,2)); iou=inter/uni.clamp(min=1)
    return T.detach().cpu().numpy().astype(np.float64), iou.cpu().numpy(), crop, rgb, hard, obs, valid

def tex_score(i, crop, rgb, hard, obs, valid, sigma=2.0):
    img=getcrop(np.asarray(fr[i]),crop)[...,::-1].astype(np.float32)/255
    g=img.mean(-1)
    def hpf(x,mk):
        mk=mk.astype(np.float32); lo=cv2.GaussianBlur(x*mk,(0,0),sigma*2)/np.maximum(cv2.GaussianBlur(mk,(0,0),sigma*2),1e-3)
        return cv2.GaussianBlur((x-lo)*mk,(0,0),0.7)/np.maximum(cv2.GaussianBlur(mk,(0,0),0.7),1e-3)
    out=[]
    for b in range(rgb.shape[0]):
        m=(hard[b]*obs*valid).cpu().numpy()>0.5
        r=rgb[b].cpu().numpy().mean(-1)
        hp=hpf(g,m); rh=hpf(r,m)
        m=cv2.erode(m.astype(np.uint8),np.ones((3,3)))>0
        a=hp[m]-hp[m].mean(); c=rh[m]-rh[m].mean()
        out.append((a*c).sum()/np.sqrt((a*a).sum()*(c*c).sum()+1e-9))
    return np.array(out)

if __name__=='__main__':
    frames=[int(x) for x in sys.argv[1].split(',')] if ',' in sys.argv[1] else list(range(*map(int,sys.argv[1].split(':'))))
    res=[]
    for i in frames:
        hyps=np.stack([Tb[i],Tb[i]@FY])
        T,iou,crop,rgb,hard,obs,valid=refine(i,hyps)
        ts=tex_score(i,crop,rgb,hard,obs,valid)
        res.append((i,iou,ts))
        print(i,'iou base %.3f flip %.3f | tex base %.3f flip %.3f'%(iou[0],iou[1],ts[0],ts[1]),flush=True)
    a=np.array([r[2] for r in res]); b=np.array([r[1] for r in res])
    print('mean tex',a.mean(0),'wins base',(a[:,0]>a[:,1]).mean(),'mean iou',b.mean(0))
