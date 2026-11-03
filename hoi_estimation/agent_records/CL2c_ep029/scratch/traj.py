import sys; sys.path.insert(0,'/work/scratch')
from fit import *   # r,K,Twc,Tcw,Tc,NRM,DPL,U_M,C_M,E1,E2,rot_a2b,rot_mean,upright_init
from render import SilRenderer
from scipy.ndimage import median_filter, binary_closing, binary_opening
import time
F=len(Tc)
Tw=r['T_world_mesh'].astype(float)
def detect_phases(OMall):
    feat=[]
    for i in range(F):
        ys,xs=np.nonzero(OMall[i]); feat.append([xs.mean(),ys.mean(),xs.min(),xs.max(),ys.min(),ys.max()])
    feat=np.array(feat); fm=median_filter(feat,size=(5,1),mode='nearest')
    mv=np.array([np.abs(fm[min(F-1,t+3)]-fm[max(0,t-3)]).max() for t in range(F)])
    moving=binary_opening(binary_closing(mv>2.5,np.ones(9)),np.ones(5))
    return moving, feat
dec=np.load('/work/scratch/dec_body.npz')
bodyR=SilRenderer(dec['V'],dec['F'])
def tt(x): return torch.tensor(np.asarray(x),dtype=torch.float32,device='cuda')
class Traj:
    def __init__(self, seg, nseg, yaw0, p0, Rb, cw0):
        self.seg=torch.tensor(seg,device='cuda',dtype=torch.long)
        self.yaw=torch.tensor(yaw0,dtype=torch.float32,device='cuda',requires_grad=True)
        self.p=torch.tensor(p0,dtype=torch.float32,device='cuda',requires_grad=True)
        self.ls=torch.zeros((),device='cuda',requires_grad=True)
        self.tilt=torch.zeros(2,device='cuda',requires_grad=True)
        self.Rb,self.cw0=tt(Rb),tt(cw0)
        self.n,self.e1,self.e2,self.cm=tt(NRM),tt(E1),tt(E2),tt(C_M)
        self.Rcw,self.tcw=tt(Tcw[:3,:3]),tt(Tcw[:3,3])
    def frames(self, idx, scale=True):
        idx=torch.as_tensor(idx,device='cuda'); sg=self.seg[idx]
        yaw=self.yaw[sg]; p=self.p[sg]
        Rt=roma.rotvec_to_rotmat(torch.stack([self.tilt[0],self.tilt[1],torch.zeros_like(self.tilt[0])]))
        Ry=roma.rotvec_to_rotmat(self.n[None]*yaw[:,None])
        Rw=Ry@(self.Rb@Rt)[None]
        cw=self.cw0[None]+p[:,0:1]*self.e1+p[:,1:2]*self.e2
        tw=cw-(Rw@self.cm)
        s=torch.exp(self.ls) if scale else torch.ones((),device='cuda')
        Rc=self.Rcw@Rw; tc=(Rc@self.cm)+torch.einsum('ij,bj->bi',self.Rcw,cw-Rw@self.cm)+self.tcw  # placeholder, recomputed below
        # scale about contact centre: x_w = Rw s (v - cm) + cw
        Rc=self.Rcw@Rw*s
        tc=torch.einsum('ij,bj->bi',self.Rcw,cw-(Rw@self.cm)*s)+self.tcw
        return Rc,tc,Rw,tw
def init_from_baseline(moving):
    # static segments: maximal runs of non-moving frames shrunk by a 3-frame margin; others per-frame
    from scipy.ndimage import label, binary_dilation
    mv=binary_dilation(moving,np.ones(7))
    lab,ns=label(~mv)
    seg=np.zeros(F,int); k=0; static_ids=[]
    for i in range(F):
        pass
    segmap={}
    for i in range(F):
        if lab[i]>0:
            if lab[i] not in segmap: segmap[lab[i]]=k; static_ids.append(k); k+=1
            seg[i]=segmap[lab[i]]
        else: seg[i]=k; k+=1
    a,b=0,np.flatnonzero(lab==1).max()+1
    Tm=np.eye(4); Tm[:3,:3]=rot_mean(Tw[a:b,:3,:3]); Tm[:3,3]=np.median(Tw[a:b,:3,3],0)
    Rb,cw0,_=upright_init(Tm)
    yaw=np.zeros(F); p=np.zeros((F,2))
    for i in range(F):
        Rbi,cwi,_=upright_init(Tw[i])
        D=Rbi@Rb.T; rv=R.from_matrix(D).as_rotvec(); yaw[i]=rv@NRM
        dc=cwi-cw0; p[i]=[dc@E1,dc@E2]
    yaw=np.unwrap(yaw)
    yaw0=np.zeros(k); p0=np.zeros((k,2))
    for s in range(k):
        yaw0[s]=np.median(yaw[seg==s]); p0[s]=np.median(p[seg==s],0)
    return seg,k,yaw0,p0,Rb,cw0,static_ids
def make_crop(OMall, margin=40):
    u=OMall.any(0); ys,xs=np.nonzero(u)
    x0=max(0,xs.min()-margin); y0=max(0,ys.min()-margin); x1=min(OMall.shape[2],xs.max()+margin); y1=min(OMall.shape[1],ys.max()+margin)
    return int(x0),int(y0),int(x1-x0),int(y1-y0)
def optimize(tr, Ogpu, Vgpu, crop, data_frames, iters=400, lr=2e-3, w_acc=1e3, w_vel_static=0.0, fit_scale=True, fit_tilt=True, chunk=120, verbose=True):
    x0,y0,W,H=crop
    Pm=bodyR.proj(K,x0,y0,W,H)
    params=[{'params':[tr.yaw,tr.p],'lr':lr}]
    if fit_scale: params.append({'params':[tr.ls],'lr':lr})
    if fit_tilt: params.append({'params':[tr.tilt],'lr':lr*0.5})
    opt=torch.optim.Adam(params)
    data_frames=np.asarray(data_frames); allf=np.arange(F)
    t0=time.time()
    for it in range(iters):
        opt.zero_grad(); tot=0.
        for c in range(0,len(data_frames),chunk):
            idx=data_frames[c:c+chunk]
            Rc,tc,_,_=tr.frames(idx,scale=True)
            m,_=bodyR.render(Rc,tc,Pm[None].expand(len(idx),-1,-1),H,W)
            o=Ogpu[idx]; v=Vgpu[idx]
            inter=(m*o*v).sum((1,2)); uni=((m+o-m*o)*v).sum((1,2))
            loss=(1-inter/uni).sum()/len(data_frames)
            loss.backward(); tot+=loss.item()
        # smoothness on the per-frame sequence
        sg=tr.seg; yaw=tr.yaw[sg]; p=tr.p[sg]
        acc=(yaw[2:]-2*yaw[1:-1]+yaw[:-2])**2*0.25+((p[2:]-2*p[1:-1]+p[:-2])**2).sum(1)   # yaw (rad) weighted ~ 0.5m lever
        ls=w_acc*acc.mean()
        ls.backward(); opt.step()
        if verbose and (it%100==0 or it==iters-1): print(it,'data %.4f smooth %.5f'%(tot,ls.item()),'s %.3f tilt'%torch.exp(tr.ls).item(),np.degrees(tr.tilt.detach().cpu().numpy()).round(2),'%.0fs'%(time.time()-t0),flush=True)
    return tr
def eval_iou(tr, Ogpu, Vgpu, crop, scale=True, Tcams=None, chunk=120):
    x0,y0,W,H=crop; Pm=bodyR.proj(K,x0,y0,W,H); out=[]
    with torch.no_grad():
        for c in range(0,F,chunk):
            idx=np.arange(c,min(F,c+chunk))
            if Tcams is None: Rc,tc,_,_=tr.frames(idx,scale=scale)
            else: T=tt(Tcams[idx]); Rc,tc=T[:,:3,:3],T[:,:3,3]
            m,_=bodyR.render(Rc,tc,Pm[None].expand(len(idx),-1,-1),H,W); m=(m>0.5).float()
            o=Ogpu[idx]; v=Vgpu[idx]
            out.append(((m*o*v).sum((1,2))/(((m+o)>0).float()*v).sum((1,2))).cpu().numpy())
    return np.concatenate(out)
def world_poses(tr):
    with torch.no_grad():
        _,_,Rw,tw=tr.frames(np.arange(F),scale=False)
    T=np.tile(np.eye(4),(F,1,1)); T[:,:3,:3]=Rw.cpu().numpy(); T[:,:3,3]=tw.cpu().numpy(); return T
