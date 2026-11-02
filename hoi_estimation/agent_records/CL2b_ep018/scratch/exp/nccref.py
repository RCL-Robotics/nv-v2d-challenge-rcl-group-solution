import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, torch
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rot
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
center=(mesh.bounds[0]+mesh.bounds[1])/2
R=Renderer(mesh,D['K'],D['W'],D['H'],scale=0.5)
T=np.load('final_c.npz')['T']
VID=np.load('vid_half.npy',mmap_mode='r'); OM=np.load('om_half.npy',mmap_mode='r'); PM=np.load('pm_half.npy',mmap_mode='r'); BB=np.load('bb.npy')
K=R.K
def perturb(T0,p):
    # p: rotvec in object frame (deg) about centre, du,dv (px at half res)
    T1=T0.copy(); Rm=T0[:3,:3]@Rot.from_rotvec(np.radians(p[:3])).as_matrix()
    pc=T0[:3,:3]@center+T0[:3,3]
    u=K[0,0]*pc[0]/pc[2]+K[0,2]+p[3]; v=K[1,1]*pc[1]/pc[2]+K[1,2]+p[4]
    pc2=np.array([(u-K[0,2])/K[0,0]*pc[2],(v-K[1,2])/K[1,1]*pc[2],pc[2]])
    T1[:3,:3]=Rm; T1[:3,3]=pc2-Rm@center; return T1
def score(T0,p,d,frames):
    Tp=np.repeat(perturb(T0,p)[None],len(frames),0)
    rgb,m=render_crop(R,torch.tensor(Tp,dtype=torch.float32,device='cuda'),d['off'],192,256)
    return ncc_score(rgb,m,d['img'],d['vo']).cpu().numpy()
def prep(frames):
    bb=BB[frames]; cen=np.stack([(bb[:,0]+bb[:,1])/2,(bb[:,2]+bb[:,3])/2],1)
    off,(omc,pmc,imc)=make_crops([OM[frames],PM[frames],VID[frames]],cen)
    return dict(off=torch.tensor(off,dtype=torch.float32,device='cuda'),img=torch.tensor(imc,dtype=torch.float32,device='cuda')/255,
                vo=torch.tensor(omc&~pmc,dtype=torch.float32,device='cuda'))
def refine(T0,frames):
    d=prep(frames); p=np.zeros(5)
    steps=[1.0,1.0,1.0,0.5,0.5]
    for rnd in range(3):
        for j in range(5):
            best=score(T0,p,d,frames).mean(); 
            for sgn in [-1,1]:
                while True:
                    q=p.copy(); q[j]+=sgn*steps[j]
                    sc=score(T0,q,d,frames).mean()
                    if sc>best+1e-4: p,best=q,sc
                    else: break
        steps=[s/2 for s in steps]
    return p,best
for nm,s,e in [('S1',0,252),('S2',456,659),('S3',720,858)]:
    fa=list(range(s,e,6)); fb=list(range(s+3,e,6)); fh=[i for i in range(s,e) if (i-s)%3]
    T0=T[s+1]
    pa,_=refine(T0,fa); pb,_=refine(T0,fb)
    dh=prep(fh[::2])
    print(nm,'even-set opt',np.round(pa,2),'odd-set opt',np.round(pb,2),
          'heldout ncc: init %.3f  pa %.3f pb %.3f'%(score(T0,np.zeros(5),dh,fh[::2]).mean(),score(T0,pa,dh,fh[::2]).mean(),score(T0,pb,dh,fh[::2]).mean()))
print('--- iou check')
def iou(T0,p,frames):
    bb=BB[frames]; cen=np.stack([(bb[:,0]+bb[:,1])/2,(bb[:,2]+bb[:,3])/2],1)
    off,(omc,pmc)=make_crops([OM[frames],PM[frames]],cen)
    Tp=np.repeat(perturb(T0,p)[None],len(frames),0)
    _,m=render_crop(R,torch.tensor(Tp,dtype=torch.float32,device='cuda'),torch.tensor(off,dtype=torch.float32,device='cuda'),192,256,need_tex=False)
    tg=torch.tensor(omc,device='cuda'); val=torch.tensor(~pmc|omc,device='cuda')
    return (((m&tg)&val).sum().float()/((m|tg)&val).sum().float()).item()
for nm,s,e,p in [('S1',0,252,[0.75,-1,-0.25,0,0.5]),('S2',456,659,[0.5,1.25,-0.4,0,0]),('S3',720,858,[1.4,0.25,0,0,0.8])]:
    fh=list(range(s+1,e,3)); p=np.array(p,float)
    print(nm,'iou init %.4f ncc-opt %.4f half %.4f'%(iou(T[s+1],np.zeros(5),fh),iou(T[s+1],p,fh),iou(T[s+1],p/2,fh)))
