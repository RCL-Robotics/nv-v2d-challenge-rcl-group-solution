import sys; sys.path.insert(0,'/work/scratch')
from common import *
from render import SilRenderer
import torch, roma
torch.manual_seed(0)
r,K,Twc,Kp=load()
Tw=r['T_world_mesh'].astype(float)
Tc=np.linalg.inv(Twc)@Tw
sm=np.load('/work/scratch/silmesh.npz')
body=SilRenderer(sm['V'],sm['body'])
hf=h5py.File(INP+'/masks.h5')
OM=hf['object']; PM=hf['person']
CROP=(520,600,320,448)  # x0,y0,W,H covers object in all frames
x0,y0,W,H=CROP
P=body.proj(K,x0,y0,W,H)
def get_masks(idx):
    o=torch.tensor(np.stack([OM[i][y0:y0+H,x0:x0+W] for i in idx]),device='cuda').float()
    p=torch.tensor(np.stack([PM[i][y0:y0+H,x0:x0+W] for i in idx]),device='cuda').float()
    return o,p
def iou(Tcs, idx, o=None,p=None):
    if o is None: o,p=get_masks(idx)
    T=torch.tensor(Tcs,dtype=torch.float32,device='cuda')
    with torch.no_grad():
        m,_=body.render(T[:,:3,:3],T[:,:3,3],P[None].expand(len(idx),-1,-1),H,W)
    m=(m>0.5).float(); v=1-p
    inter=(m*o*v).sum((1,2)); uni=(((m+o)>0).float()*v).sum((1,2))
    return (inter/uni).cpu().numpy()
def rot_mean(Rs):
    M=Rs.sum(0); U,S,Vt=np.linalg.svd(M); D=np.diag([1,1,np.linalg.det(U@Vt)]); return U@D@Vt
def fit_static(idx, T0, iters=300, lr_r=3e-3, lr_t=3e-3, fix_depth=False, verbose=False):
    o,p=get_masks(idx); v=1-p; B=len(idx)
    R0=torch.tensor(T0[:3,:3],dtype=torch.float32,device='cuda'); t0=torch.tensor(T0[:3,3],dtype=torch.float32,device='cuda')
    w=torch.zeros(3,device='cuda',requires_grad=True); dt=torch.zeros(3,device='cuda',requires_grad=True)
    opt=torch.optim.Adam([{'params':[w],'lr':lr_r},{'params':[dt],'lr':lr_t}])
    Pb=P[None].expand(B,-1,-1)
    for it in range(iters):
        Rr=roma.rotvec_to_rotmat(w)@R0
        t=t0+dt
        if fix_depth: t=t0*(1+dt[0]*0)+torch.stack([dt[0],dt[1],torch.zeros_like(dt[2])])  # placeholder
        m,_=body.render(Rr[None].expand(B,-1,-1),t[None].expand(B,-1),Pb,H,W)
        inter=(m*o*v).sum((1,2)); uni=((m+o-m*o)*v).sum((1,2))
        loss=(1-inter/uni).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and it%50==0: print(it,loss.item())
    T=np.eye(4); T[:3,:3]=(roma.rotvec_to_rotmat(w)@R0).detach().cpu().numpy(); T[:3,3]=(t0+dt).detach().cpu().numpy()
    return T, 1-loss.item()
pl=json.load(open(INP+'/ground_plane.json'))['plane']; NRM=np.array(pl[:3]); DPL=pl[3]
Tcw=np.linalg.inv(Twc)
VM=sm['V']
def wheel_points(V=VM):
    low=V[V[:,2]<-0.29]; pts=[]
    for sx in [-1,1]:
        for sy in [-1,1]:
            s=low[(np.sign(low[:,0])==sx)&(np.sign(low[:,1])==sy)]; pts.append(s[np.argmin(s[:,2])])
    return np.array(pts)
WP=wheel_points()
C_M=WP.mean(0)
_u=np.linalg.svd(WP-C_M)[2][2]; U_M=_u if _u[2]>0 else -_u   # wheel-plane normal in mesh coords (up)
def rot_a2b(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=a@b
    if np.linalg.norm(v)<1e-9: return np.eye(3)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return np.eye(3)+vx+vx@vx/(1+c)
def rodrigues_t(axis,ang):
    axis=torch.as_tensor(axis,dtype=torch.float32,device='cuda'); return roma.rotvec_to_rotmat(axis*ang)
# in-plane basis
E1=np.cross(NRM,[0,0,1.]); E1/=np.linalg.norm(E1); E2=np.cross(NRM,E1)
def upright_init(Tw_mean):
    """from a world pose, get R_base (upright), yaw=0, contact center on plane"""
    Rw=Tw_mean[:3,:3]
    Rb=rot_a2b(Rw@U_M,NRM)@Rw     # minimal tilt correction
    cw=Rw@C_M+Tw_mean[:3,3]
    hc=cw@NRM+DPL; cw0=cw-hc*NRM   # project to plane
    return Rb,cw0,hc
def upright_pose(Rb,cw0,yaw,p,s=1.0,h0=0.0):
    """world pose: mesh -> world. returns 4x4 numpy (scale ignored in output)"""
    Ry=R.from_rotvec(NRM*yaw).as_matrix(); Rw=Ry@Rb
    cw=cw0+p[0]*E1+p[1]*E2+h0*NRM
    T=np.eye(4); T[:3,:3]=Rw; T[:3,3]=cw-Rw@(C_M)
    return T
def fit_upright(idx, Rb, cw0, iters=300, lr=3e-3, h0=0.0, fit_scale=True, yaw0=0.0, p0=(0,0), verbose=False):
    o,p=get_masks(idx); v=1-p; B=len(idx)
    tt=lambda x: torch.tensor(x,dtype=torch.float32,device='cuda')
    Rb_t,cw_t,n_t,e1,e2,cm=tt(Rb),tt(cw0),tt(NRM),tt(E1),tt(E2),tt(C_M)
    Rcw,tcw=tt(Tcw[:3,:3]),tt(Tcw[:3,3])
    yaw=torch.tensor(float(yaw0),device='cuda',requires_grad=True); pp=torch.tensor(list(p0),dtype=torch.float32,device='cuda',requires_grad=True)
    ls=torch.zeros((),device='cuda',requires_grad=True)
    params=[yaw,pp]+([ls] if fit_scale else [])
    opt=torch.optim.Adam(params,lr=lr)
    Pb=P[None].expand(B,-1,-1)
    for it in range(iters):
        Ry=roma.rotvec_to_rotmat(n_t*yaw); Rw=Ry@Rb_t; s=torch.exp(ls)
        cw=cw_t+pp[0]*e1+pp[1]*e2+h0*n_t
        Rc=Rcw@Rw*s; tc=Rcw@(cw-Rw@cm*s)+tcw
        m,_=body.render(Rc[None].expand(B,-1,-1),tc[None].expand(B,-1),Pb,H,W)
        inter=(m*o*v).sum((1,2)); uni=((m+o-m*o)*v).sum((1,2))
        loss=(1-inter/uni).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and it%50==0: print(it,loss.item(),yaw.item(),pp.tolist(),s.item())
    return float(yaw.item()), pp.detach().cpu().numpy().astype(float), float(torch.exp(ls).item()), 1-loss.item()
