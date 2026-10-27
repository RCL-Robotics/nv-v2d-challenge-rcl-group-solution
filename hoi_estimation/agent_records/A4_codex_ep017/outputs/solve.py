#!/usr/bin/env python3
"""Offline cone refinement. All observations come from the supplied reconstruction/masks.
No depth, scale, floor or global-frame calibration is estimated.
"""
import argparse, json, time, os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import cv2
import h5py
import trimesh
import torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares

def rz(a):
    c,s=torch.cos(a),torch.sin(a); o=torch.zeros_like(a); u=torch.ones_like(a)
    return torch.stack([c,-s,o,s,c,o,o,o,u],-1).reshape(-1,3,3)

def smallrot(x,z):
    o=torch.zeros_like(x)
    A=torch.stack([o,-z,o,z,o,-x,o,x,o],-1).reshape(-1,3,3)
    return torch.matrix_exp(A)

class Silhouette:
    def __init__(self,mesh):
        self.v=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda')
        self.f=torch.tensor(mesh.faces.astype(np.int32),device='cuda')
        self.ctx=dr.RasterizeCudaContext()
    def render(self,R,t,K,origins):
        v=self.v[None]@R.transpose(1,2)+t[:,None]
        z=v[:,:,2]
        x=2*(K[0,0]*v[:,:,0]+(K[0,2]-origins[:,0,None])*z)/256-z
        y=2*(K[1,1]*v[:,:,1]+(K[1,2]-origins[:,1,None])*z)/256-z
        clip=torch.stack([x,y,1.002002*z-.02002002,z],-1).contiguous()
        rast,_=dr.rasterize(self.ctx,clip,self.f,(256,256))
        mask=(rast[:,:,:,3:4]>0).float()
        return dr.antialias(mask,rast,clip,self.f)[:,:,:,0]

def fit_visible(T,K,mesh,masks):
    """Bounded full-mesh silhouette fit. Person pixels permit hand/body occlusion."""
    ren=Silhouette(mesh); n=len(T); pars=np.zeros((n,5)); quality=np.zeros(n)
    origins=[]; objects=[]; valid=[]; frames=[]
    for f in range(n):
        obs=masks['object'][f]; yy,xx=np.where(obs)
        if len(xx)<1500:continue
        # Fixed 256px crop fits this cone even at its closest approach.
        p=K@T[f,:3,3]; p=p[:2]/p[2]
        xy=np.round(p-[128,128]).astype(int)
        xy=np.clip(xy,[0,0],[1536-256,1152-256]); x,y=xy
        obj=obs[y:y+256,x:x+256]
        if obj.sum()<.99*len(xx):continue
        person=masks['person'][f,y:y+256,x:x+256]
        occ=cv2.dilate((person&~obj).astype('uint8'),np.ones((3,3),np.uint8))>0
        occ[obj]=False
        origins.append(xy);objects.append(obj);valid.append(~occ);frames.append(f)
    allbase=np.full(n,np.nan);allfit=np.full(n,np.nan)
    for start in range(0,len(frames),12):
        fs=np.array(frames[start:start+12]); sl=slice(start,start+len(fs))
        R0=torch.tensor(T[fs,:3,:3],dtype=torch.float32,device='cuda')
        t0=torch.tensor(T[fs,:3,3],dtype=torch.float32,device='cuda')
        radius=t0.norm(dim=-1,keepdim=True)
        ray=t0/t0[:,2:3]
        orig=torch.tensor(np.array(origins[sl]),device='cuda',dtype=torch.float32)
        obj=torch.tensor(np.array(objects[sl]),device='cuda',dtype=torch.float32)
        val=torch.tensor(np.array(valid[sl]),device='cuda',dtype=torch.float32)
        def iou(mask):
            inter=(mask*obj).sum((1,2)); union=((mask+obj-mask*obj)*val).sum((1,2))
            return inter/union.clamp_min(1)
        # Coarse yaw search avoids a silhouette extremum at a square-base diagonal.
        with torch.no_grad():
            base=ren.render(R0,t0,K,orig); baseiou=iou(base)
            best=baseiou.clone();initial=torch.zeros(len(fs),device='cuda')
            for a in range(-42,43,6):
                angle=torch.full((len(fs),),float(np.deg2rad(a)),device='cuda')
                score=iou(ren.render(R0@rz(angle),t0,K,orig))-.015*(a/30)**2
                sel=score>best; initial[sel]=angle[sel];best[sel]=score[sel]
        par=torch.zeros((len(fs),5),device='cuda',requires_grad=True)
        with torch.no_grad():par[:,0]=initial
        optim=torch.optim.Adam([par],lr=.025)
        bounds=torch.tensor([np.deg2rad(45),np.deg2rad(4),np.deg2rad(4),4,4],device='cuda')
        bestloss=torch.full((len(fs),),float('inf'),device='cuda');bestpar=par.detach().clone()
        for step in range(110):
            yaw,tx,tz,dx,dy=par.unbind(-1)
            R=smallrot(tx,tz)@R0@rz(yaw)
            raynew=ray+torch.stack([dx/K[0,0],dy/K[1,1],torch.zeros_like(dx)],-1)
            t=raynew/raynew.norm(dim=-1,keepdim=True)*radius
            mask=ren.render(R,t,K,orig)
            regular=.015*(yaw/np.deg2rad(30))**2+.012*((tx/np.deg2rad(3))**2+(tz/np.deg2rad(3))**2)+.008*((dx/3)**2+(dy/3)**2)
            loss=1-iou(mask)+regular
            with torch.no_grad():
                sel=loss<bestloss;bestloss[sel]=loss[sel];bestpar[sel]=par.detach()[sel]
            optim.zero_grad();loss.sum().backward();optim.step()
            with torch.no_grad():par.clamp_(-bounds,bounds)
        p=bestpar.cpu().numpy();pars[fs]=p
        with torch.no_grad():
            raynew=ray+torch.stack([bestpar[:,3]/K[0,0],bestpar[:,4]/K[1,1],torch.zeros_like(bestpar[:,3])],-1)
            t=raynew/raynew.norm(dim=-1,keepdim=True)*radius
            fit=ren.render(smallrot(bestpar[:,1],bestpar[:,2])@R0@rz(bestpar[:,0]),t,K,orig)
            fi=iou(fit).cpu().numpy();bi=baseiou.cpu().numpy()
        allbase[fs]=bi;allfit[fs]=fi
        # A poor mask or an almost entirely hidden cone cannot support a correction.
        good=(fi>.65)&(fi-bi>.015)
        pars[fs[~good]]=0
        quality[fs]=np.clip((fi-.6)/.25,0,1)
        print(f'fit {start+len(fs)}/{len(frames)}, mean IoU {bi.mean():.3f} -> {fi.mean():.3f}',flush=True)
    return pars,quality,allbase,allfit

def refine(args):
    start=time.time();np.random.seed(0);torch.manual_seed(0);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    root=args.inputs; rec=np.load(os.path.join(root,'reconstruction.npz'))
    baseline=rec['T_world_mesh'].astype(np.float64);F=len(baseline)
    cam=json.load(open(os.path.join(root,'camera.json')));K=np.array(cam['K'])
    W=np.array(cam['T_world_cam']);C=np.linalg.inv(W);T=C@baseline
    mesh=trimesh.load(os.path.join(root,'object_mesh.glb'),force='scene',process=False).to_geometry()
    with h5py.File(os.path.join(root,'masks.h5'),'r') as masks:
        par,q,bi,fi=fit_visible(T,K,mesh,masks)
        area=np.array([np.count_nonzero(masks['object'][f]) for f in range(F)])
    # Episode-specific contact bounds, confirmed by video: frame 117 is the
    # first baseline symmetry jump while the cone is stationary; frame 340 is
    # already stationary on the table. Keep the initial identity of its faces.
    pickup,settled=117,340
    # Filter tiny per-frame fit noise, pooling stationary intervals. No temporal
    # filtering of the supplied centre distance is performed.
    for k in range(5):
        par[:pickup,k]=np.median(par[:pickup,k])
        par[settled:,k]=np.median(par[settled:,k])
        par[pickup:settled,k]=gaussian_filter1d(par[pickup:settled,k],1.5)
    # The near-circular shaft has a square base: geometry yields yaw modulo 90.
    # Resolve its branch using the preserved pre-pickup pose and the observed
    # full turn of the shoulders while both hands hold the shaft.
    hp=rec['human_keypoints_world'].astype(np.float64)@C[:3,:3].T
    shoulder=hp[:,5]-hp[:,6]
    h=np.unwrap(np.arctan2(shoulder[:,2],shoulder[:,0]));h=gaussian_filter1d(h,2)
    raw=np.arctan2(T[:,2,0],T[:,0,0])
    measure=raw+par[:,0]
    anchor=np.median(measure[:pickup])
    expected=h-h[120]+anchor
    expected[:pickup]=anchor;expected[settled:]=expected[settled]
    weight=q.copy();weight[area<1500]=0
    weight[:pickup]=1;weight[settled:]=1
    # A soft shoulder prior selects a face branch, not the cone's precise yaw.
    # Acceleration regularization bridges the genuinely invisible interval and
    # suppresses the baseline's 25--54 degree one-frame oscillations.
    priorweight=np.full(F,1/np.deg2rad(32));priorweight[:pickup]=1/np.deg2rad(3)
    priorweight[settled:]=0
    velocity=np.diff(expected);velocity[:pickup-1]=0;velocity[settled-1:]=0
    def residual(theta):
        obs=np.sqrt(weight)*np.sin(2*(theta-measure))/np.deg2rad(9)/2
        # During the first continuously visible carry segment, preserve the
        # baseline's face identity after removing its observed +90deg pickup
        # jump. A shoulder prior alone can select another face when the arms
        # rotate relative to the torso during lifting.
        obs[pickup:190]=np.sqrt(weight[pickup:190])*(theta[pickup:190]-measure[pickup:190]+np.pi/2)/np.deg2rad(9)
        prior=(theta-expected)*priorweight
        vel=(np.diff(theta)-velocity)/np.deg2rad(9)
        acc=np.diff(theta,2)/np.deg2rad(3)
        return np.r_[obs,prior,vel,acc]
    # Banded finite-difference sparsity makes this deterministic solve fast.
    from scipy.sparse import lil_matrix
    spars=lil_matrix((4*F-3,F),dtype=int)
    for f in range(F):spars[f,f]=1;spars[F+f,f]=1
    for f in range(F-1):spars[2*F+f,f:f+2]=1
    for f in range(F-2):spars[3*F-1+f,f:f+3]=1
    opt=least_squares(residual,expected,jac_sparsity=spars.tocsr(),max_nfev=150,ftol=1e-9,xtol=1e-9,gtol=1e-8)
    theta=opt.x
    # Keep stable endpoints' measured branch; release permits the person to
    # turn independently afterwards. Anchor pre-pickup corrections to small
    # silhouette evidence only, and use the constant final fitted orientation.
    theta[:pickup]=anchor
    final_phase=measure[settled]+np.round((theta[settled]-measure[settled])/(2*np.pi))*2*np.pi
    theta[settled:]=final_phase
    delta=theta-raw
    Rtilt=Rotation.from_rotvec(np.c_[par[:,1],np.zeros(F),par[:,2]]).as_matrix()
    Tout=T.copy();Tout[:,:3,:3]=Rtilt@T[:,:3,:3]@Rotation.from_rotvec(np.c_[np.zeros(F),np.zeros(F),delta]).as_matrix()
    radius=np.linalg.norm(T[:,:3,3],axis=1);ray=T[:,:3,3]/T[:,2,3,None]
    ray[:,:2]+=par[:,3:5]/[K[0,0],K[1,1]]
    Tout[:,:3,3]=ray/np.linalg.norm(ray,axis=1)[:,None]*radius[:,None]
    out=W@Tout;out[:,3,:]=[0,0,0,1]
    assert out.shape==(F,4,4) and np.isfinite(out).all()
    assert np.max(np.abs(np.linalg.det(out[:,:3,:3])-1))<1e-5
    assert np.max(np.abs(np.linalg.norm((C@out)[:,:3,3],axis=1)-radius))<1e-10
    np.savez_compressed(args.out,T_world_mesh=out.astype(np.float64))
    np.savez_compressed(os.path.join(os.path.dirname(args.out),'refinement_diagnostics.npz'),parameters=par,quality=q,baseline_iou=bi,fitted_iou=fi,theta=theta,raw_yaw=raw,shoulder_yaw=expected,area=area)
    metrics=dict(frames=F,seconds=time.time()-start,mean_baseline_iou=float(np.nanmean(bi)),mean_fitted_iou=float(np.nanmean(fi)),mean_rotation_change_deg=float(np.mean((Rotation.from_matrix(T[:,:3,:3]).inv()*Rotation.from_matrix(Tout[:,:3,:3])).magnitude())*180/np.pi),max_camera_distance_change_m=float(np.max(np.abs(np.linalg.norm(Tout[:,:3,3],axis=1)-radius))))
    json.dump(metrics,open(os.path.join(os.path.dirname(args.out),'run_metrics.json'),'w'),indent=2);print(metrics)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);refine(p.parse_args())
