#!/usr/bin/env python3
"""Fit the released metric mug mesh to the monocular video; no external assets.
The long local mesh axis is z. Mesh +z points toward the BOTTOM of this mug,
identified from its lid/handle geometry and texture. Thus upright hypotheses have
+z approximately opposite the released upward ground normal. This explicit
asset-specific convention prevents the observed upside-down tracking solution.
"""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, json, time, random
from pathlib import Path
import numpy as np
import cv2, h5py, trimesh, torch
import torch.nn.functional as tf
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation, Slerp
from scipy.ndimage import gaussian_filter1d

def tensor(a):
    return torch.tensor(np.array(a,copy=True), device='cuda:0',dtype=torch.float32)

def r6_matrix(x):
    a=tf.normalize(x[...,0:3],dim=-1)
    b=x[...,3:6]-(a*x[...,3:6]).sum(-1,keepdim=True)*a
    b=tf.normalize(b,dim=-1)
    return torch.stack((a,b,torch.cross(a,b,dim=-1)),dim=-1)

def matrix_r6(x):
    return np.concatenate((x[..., :,0],x[..., :,1]),axis=-1)

class Fitter:
    def __init__(self, inputs):
        self.p=Path(inputs)
        # Retain the released calibration. The reference body's unclothed
        # silhouette differs from hair, clothing and footwear, so fitting that
        # outline would not provide a reliable independent focal calibration.
        self.cam=json.loads((self.p/'camera.json').read_text())
        self.K=np.array(self.cam['K']);self.wc=np.array(self.cam['T_world_cam']);self.cw=np.linalg.inv(self.wc)
        self.old=np.load(self.p/'reconstruction.npz')
        self.T0=np.einsum('ij,fjk->fik',self.cw,self.old['T_world_mesh'].astype(float))
        self.N=len(self.T0)
        self.normal=np.array(json.loads((self.p/'ground_plane.json').read_text())['plane'][:3]);self.normal/=np.linalg.norm(self.normal)
        self.down_cam=self.cw[:3,:3]@(-self.normal)
        scene=trimesh.load(self.p/'object_mesh.glb',force='scene',process=False)
        self.mesh=scene.to_geometry()
        # Fit the nearly planar bottom face: the scanned mesh's z-axis differs
        # from its actual support normal by about three degrees.
        verts=np.asarray(self.mesh.vertices)
        bottom=verts[verts[:,2]>verts[:,2].max()-.004]
        for _ in range(5):
            design=np.c_[bottom[:,:2],np.ones(len(bottom))]
            plane=np.linalg.lstsq(design,bottom[:,2],rcond=None)[0]
            residual=bottom[:,2]-design@plane
            good=np.abs(residual)<.001
            if good.sum()<100:break
            bottom=bottom[good]
        self.bottom_normal=np.r_[-plane[:2],1.]
        self.bottom_normal/=np.linalg.norm(self.bottom_normal)
        self.upright=Rotation.align_vectors([self.down_cam],[self.bottom_normal])[0].as_matrix()
        self.v=tensor(self.mesh.vertices);self.faces=torch.as_tensor(self.mesh.faces.astype(np.int32),device='cuda:0')
        self.uv=tensor(self.mesh.visual.uv)[None]
        self.tex=tensor(np.array(self.mesh.visual.material.baseColorTexture.convert('RGB'))[::-1].copy())[None]/255
        # Full texture mesh for fitting. A welded, simplified copy for evaluation.
        sm=trimesh.Trimesh(self.mesh.vertices.copy(),self.mesh.faces.copy(),process=True)
        sm=sm.simplify_quadric_decimation(face_count=10000)
        self.sv=tensor(sm.vertices);self.sf=torch.as_tensor(sm.faces.astype(np.int32),device='cuda:0')
        self.ctx=dr.RasterizeCudaContext(device=0)
        self.size=192;self.scale=2.0;self.span=self.size/self.scale
        self.images=[];self.masks=[];self.person=[];self.xy=[];self.bbox=[];self.area=[]
        cap=cv2.VideoCapture(str(self.p/'video.mp4'))
        with h5py.File(self.p/'masks.h5','r') as f:
            for i in range(self.N):
                ok,im=cap.read()
                if not ok:raise RuntimeError(f'Missing frame {i}')
                m=f['object'][i].astype(np.uint8);p=f['person'][i].astype(np.uint8)
                ys,xs=np.nonzero(m)
                if len(xs): bb=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1]);center=(bb[:2]+bb[2:])/2
                else:
                    t=self.T0[i,:3,3];center=self.K[:2,:2]@(t[:2]/t[2])+self.K[:2,2];bb=np.r_[center-15,center+15]
                xy=np.round(center-self.span/2).astype(int);x,y=xy;sz=int(self.span)
                self.xy.append(xy);self.bbox.append(bb);self.area.append(m.sum())
                # The object is far from the image boundary in this sequence.
                self.images.append(cv2.resize(cv2.cvtColor(im[y:y+sz,x:x+sz],cv2.COLOR_BGR2RGB),(self.size,self.size),interpolation=cv2.INTER_LINEAR))
                self.masks.append(cv2.resize(m[y:y+sz,x:x+sz],(self.size,self.size),interpolation=cv2.INTER_NEAREST))
                self.person.append(cv2.resize(p[y:y+sz,x:x+sz],(self.size,self.size),interpolation=cv2.INTER_NEAREST))
        cap.release()
        self.images=np.array(self.images);self.masks=np.array(self.masks);self.person=np.array(self.person)
        self.xy=np.array(self.xy);self.bbox=np.array(self.bbox);self.area=np.array(self.area)
        angles=[];image_axes=[]
        for m in self.masks:
            yy,xx=np.nonzero(m)
            axis2=np.linalg.eigh(np.cov(np.stack((xx,yy))))[1][:,-1]
            if axis2[1]<0:axis2=-axis2
            angles.append(abs(np.degrees(np.arctan2(axis2[0],axis2[1]))));image_axes.append(axis2)
        # Explicit image observation: in the appreciably tilted lifts the dark
        # base and its pale barcode face the camera, while the lid faces away.
        # Use a soft 10--16 degree image-axis gate, only above the support height,
        # to resolve the otherwise ambiguous forward/backward tilt hypotheses.
        self.tilt_weight=np.clip((np.array(angles)-10)/6,0,1)
        self.tilt_weight*=self.bbox[:,3]<np.quantile(self.bbox[:,3],.8)-8
        # A conditional inclination prior from the released intermediates. It
        # never selects yaw or metric depth. Require agreement of both axes,
        # agreement with the 2D mask axis, and the observed visible-base side.
        axes=[]
        for name in ['T_cam_mesh_foundationpose','T_cam_mesh_coconet']:
            a=-self.old[name][:,:3,2].astype(float)
            a[a[:,1]<0]*=-1
            a/=np.linalg.norm(a,axis=1,keepdims=True)
            axes.append(a)
        ray=self.T0[:,:3,3].copy();ray/=np.linalg.norm(ray,axis=1,keepdims=True)
        projected=[]
        for a in axes:
            a2=a[:,:2]-self.T0[:,:2,3]/self.T0[:,2:3,3]*a[:,2:3]
            a2/=np.linalg.norm(a2,axis=1,keepdims=True)
            projected.append((a2*np.array(image_axes)).sum(1))
        agreement=(axes[0]*axes[1]).sum(1)
        facing1=(axes[0]*ray).sum(1);facing2=(axes[1]*ray).sum(1)
        meanaxis=axes[0]+axes[1];meanaxis/=np.linalg.norm(meanaxis,axis=1,keepdims=True)
        self.inclination_target=(meanaxis*ray).sum(1)
        self.inclination_confidence=np.clip((agreement-.96)/.04,0,1)
        self.inclination_confidence*=((np.array(angles)>14)&(facing1<-.08)&(facing2<-.08)&
                                      (projected[0]>np.cos(np.deg2rad(15)))&(projected[1]>np.cos(np.deg2rad(15)))&
                                      (self.tilt_weight>0))
        self.xy_t=tensor(self.xy)
        self.im_t=tensor(self.images)/255; self.m_t=tensor(self.masks)
        self.p_t=tensor(self.person)
        self.er_t=1-tf.max_pool2d(1-self.m_t[:,None],5,1,2)[:,0]
        self.vis_t=(1-self.p_t)*self.m_t
        # Robust silhouette boundary alignment also guides initially disjoint seeds.
        self.dt_t=tensor(np.array([cv2.distanceTransform(1-m.astype(np.uint8),cv2.DIST_L2,5) for m in self.masks]))/self.scale

    def render(self,R,t,ids,color=False):
        v=self.v if color else self.sv; fa=self.faces if color else self.sf
        vc=torch.einsum('bij,vj->bvi',R,v)+t[:,None,:]
        xy=self.xy_t[ids]
        u=self.K[0,0]*vc[...,0]+(self.K[0,2]-xy[:,None,0]+.5)*vc[...,2]
        y=self.K[1,1]*vc[...,1]+(self.K[1,2]-xy[:,None,1]+.5)*vc[...,2]
        clip=torch.stack((u*(2/self.span)-vc[...,2],y*(2/self.span)-vc[...,2],vc[...,2]*1.002002-.2002002,vc[...,2]),dim=-1)
        rast,_=dr.rasterize(self.ctx,clip,fa,resolution=(self.size,self.size))
        mask=(rast[...,3:]>0).float()
        mask=dr.antialias(mask,rast,clip,fa)[...,0]
        col=None
        if color:
            uv,_=dr.interpolate(self.uv,rast,fa)
            col=dr.texture(self.tex,uv,filter_mode='linear',boundary_mode='clamp')
        return mask,col

    def losses(self,R,t,ids,color=False):
        mask,col=self.render(R,t,ids,color)
        target=self.m_t[ids];person=self.p_t[ids];area=target.sum((1,2)).clamp_min(1)
        missing=((1-mask)*target).sum((1,2))/area
        extra=(mask*(1-target)*(1-person)).sum((1,2))/area
        dt=(mask*self.dt_t[ids]*(1-target)*(1-person)).sum((1,2))/area
        sil=missing+extra+0.06*dt
        photo=torch.zeros_like(sil)
        if color:
            # Affine illumination compensation, fitted on visible interior pixels.
            # Detach this nuisance fit so pose cannot improve by changing contrast alone.
            weight=self.er_t[ids]*mask.detach()
            denom=weight.sum((1,2)).clamp_min(10)
            w=weight[...,None]
            mu=(col*w).sum((1,2))/denom[:,None];im=self.im_t[ids]
            mi=(im*w).sum((1,2))/denom[:,None]
            var=((col-mu[:,None,None])**2*w).sum((1,2))/denom[:,None]
            cov=((col-mu[:,None,None])*(im-mi[:,None,None])*w).sum((1,2))/denom[:,None]
            gain=(cov/(var+.002)).clamp(.4,2).detach()
            bias=(mi-gain*mu).detach()
            adj=(col*gain[:,None,None]+bias[:,None,None])
            residual=torch.sqrt((adj-im)**2+.002**2).mean(-1)
            photo=(residual*weight).sum((1,2))/denom
        return sil,photo,mask,col

    def fit(self,ids,Rs,ts,steps=150,color=False,upright=False,lr=.025,prior=None,groups=None):
        ids=np.asarray(ids,dtype=int);base=ts.copy()
        if groups is None: groups=np.arange(len(ids))
        groups=np.asarray(groups,dtype=int);gt=torch.as_tensor(groups,device="cuda:0",dtype=torch.int64)
        counts=tensor(np.bincount(groups,minlength=len(Rs))).clamp_min(1)
        Rvar=torch.nn.Parameter(tensor(matrix_r6(Rs)))
        uv=base[:,:2]/base[:,2,None]*np.diag(self.K)[:2]+self.K[:2,2]
        tr=torch.nn.Parameter(torch.zeros((len(Rs),3),device='cuda:0'))
        uv=tensor(uv);z=tensor(base[:,2])
        optim=torch.optim.Adam([Rvar,tr],lr=lr)
        best=np.full(len(Rs),np.inf);bestR=Rs.copy();bestt=ts.copy()
        for step in range(steps):
            optim.zero_grad(set_to_none=True)
            R=r6_matrix(Rvar)
            depth=z*torch.exp(tr[:,2]*.1)
            pos=uv+tr[:,:2]*10
            t=torch.cat(((pos-tensor(self.K[:2,2]))/tensor(np.diag(self.K)[:2])*depth[:,None],depth[:,None]),dim=-1)
            sil,photo,_,_=self.losses(R[gt],t[gt],ids,color)
            loss=torch.zeros(len(Rs),device="cuda:0").index_add(0,gt,sil+1.8*photo)/counts
            if np.any(upright):
                loss=loss+0.5*tensor(np.broadcast_to(upright,(len(Rs),)))*((R@tensor(self.bottom_normal)-tensor(self.down_cam))**2).sum(-1)
            # The task and visible lid/base show that all lifts keep the base below
            # the lid in the image. Reject the tracker's occasional axial flip.
            loss=loss+3*tf.relu(.25-R[:,1,2])**2
            # Explicit video observation: the handle remains to image-left of the
            # body in all six table placements and the hand grips that same side
            # during lifts. Mesh +x points toward the handle. A soft margin of .35
            # prevents a hidden, opposite-side handle from exploiting person masks.
            handle_x=R[:,0,0]-(t[:,0]/t[:,2])*R[:,2,0]
            loss=loss+2*tf.relu(handle_x+.35)**2
            tilt=tensor(np.bincount(groups,weights=self.tilt_weight[ids],minlength=len(Rs)))/counts
            facing=((R@tensor(self.bottom_normal))*tf.normalize(t,dim=-1)).sum(-1)
            loss=loss+3*tilt*tf.relu(facing+.06)**2
            conf=np.bincount(groups,weights=self.inclination_confidence[ids],minlength=len(Rs))
            target=np.bincount(groups,weights=self.inclination_confidence[ids]*self.inclination_target[ids],minlength=len(Rs))/np.maximum(conf,1e-8)
            loss=loss+.75*(tensor(conf)/counts)*(facing-tensor(target))**2
            if prior is not None:
                pR,pt,wR,wt=prior
                loss=loss+wR*((R-tensor(pR))**2).sum((1,2))+wt*((t-tensor(pt))**2).sum(1)
            if step>5:
                now=loss.detach().cpu().numpy();good=now<best
                if good.any():
                    best[good]=now[good];bestR[good]=R.detach().cpu().numpy()[good];bestt[good]=t.detach().cpu().numpy()[good]
            loss.sum().backward()
            optim.step()
            if step in [int(steps*.55),int(steps*.8)]:
                for g in optim.param_groups:g['lr']*=.4
        return bestR,bestt,best

    def initial_ts(self,ids):
        # Metric size/pose fitting determines depth, rather than the uncalibrated networks.
        return self.T0[ids,:3,3].copy()

def diagnostic(f,out):
    ids=np.zeros(24,dtype=int)
    Rs=np.array([f.upright@Rotation.from_rotvec([0,0,j*np.pi/12]).as_matrix() for j in range(24)])
    ts=f.initial_ts(ids)
    start=time.time();r,t,c=f.fit(ids,Rs,ts,steps=160,color=True,upright=True)
    print('fit time',time.time()-start,flush=True)
    order=np.argsort(c)
    for j in order:print('hyp',j,'cost',c[j], 't',t[j], 'rot',Rotation.from_matrix(r[j]).as_rotvec(),flush=True)
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(192*8,192*3))
    with torch.no_grad():
        sil,photo,mask,col=f.losses(tensor(r),tensor(t),ids,True)
    for i in range(24):
        im=f.images[0].copy(); m=mask[i].cpu().numpy();edge=cv2.morphologyEx((m>.5).astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((2,2),np.uint8))>0;im[edge]=[255,30,20]
        rc=col[i].cpu().numpy()*m[...,None]; combined=np.concatenate((im,(rc*255).clip(0,255).astype(np.uint8)),1)
        a=Image.fromarray(cv2.resize(combined,(192,192)));ImageDraw.Draw(a).text((0,0),f'{i} {c[i]:.3f}',fill='red');canvas.paste(a,((i%8)*192,(i//8)*192))
    canvas.save(out/'hypothesis_fits.jpg');np.savez(out/'hypothesis_fits.npz',R=r,t=t,cost=c)

def runs(mask):
    return np.where(np.diff(np.r_[False,np.asarray(mask,bool),False]))[0].reshape(-1,2)

def keyframes(f,out):
    rest=f.bbox[:,3]>=np.quantile(f.bbox[:,3],.8)-3
    segments=np.array([(a,b) for a,b in runs(rest) if b-a>=15])
    rest[:]=False
    for a,b in segments:rest[a:b]=True
    # Dense enough to sample each brief pickup and the changes of inclination.
    ids=np.unique(np.r_[np.arange(0,f.N,10),f.N-1,[(a+b)//2 for a,b in segments]]).astype(int)
    allR=[];allt=[];allc=[]
    yaw=np.array([Rotation.from_rotvec(f.bottom_normal*(j*2*np.pi/16)).as_matrix() for j in range(16)])
    start=time.time()
    for k,i in enumerate(ids):
        # Search all azimuths from image-derived longitudinal directions.
        # Inclination remains free during lifts.
        if rest[i]:
            Rs=f.upright[None]@yaw
        else:
            yy,xx=np.nonzero(f.masks[i])
            cov=np.cov(np.stack((xx,yy)))
            axis2=np.linalg.eigh(cov)[1][:,-1]
            if axis2[1]<0:axis2=-axis2
            seeds=[]
            # Search both an almost vertical and an appreciably tilted cup; no
            # later network pose is trusted to choose the longitudinal direction.
            for dz in [0.,-.55]:
                d=np.r_[axis2*np.sqrt(1-dz*dz),dz]
                base=Rotation.align_vectors([d],[[0,0,1]])[0].as_matrix()
                seeds.extend(base[None]@yaw)
            Rs=np.array(seeds)
        r,t,c=f.fit(np.full(len(Rs),i),Rs,f.initial_ts(np.full(len(Rs),i)),steps=180,color=True,upright=bool(rest[i]),lr=.025)
        # Equal state counts simplify the temporal dynamic program.
        if len(Rs)==16:r=np.tile(r,(2,1,1));t=np.tile(t,(2,1));c=np.tile(c,2)
        allR.append(r);allt.append(t);allc.append(c)
        if k%5==0:print(f'Keyframe {k+1}/{len(ids)}, frame {i}, best cost {c.min():.4f}, elapsed {time.time()-start:.1f}s',flush=True)
    R=np.array(allR);t=np.array(allt);cost=np.array(allc)
    # A continuous path through the image-supported alternatives. Large handle
    # flips cost more than modest frame-to-frame physical motion.
    path=np.zeros(cost.shape,dtype=int);score=cost[0].copy()
    for k in range(1,len(ids)):
        rel=np.einsum('aij,bik->abjk',R[k-1],R[k])
        ang=np.arccos(np.clip((np.trace(rel,axis1=-2,axis2=-1)-1)/2,-1,1))
        dist=((t[k-1,:,None,:]-t[k,None,:,:])**2).sum(-1)
        transition=.024*ang**2+.6*dist
        # Closely spaced extra keyframes must not enable artificial jumps.
        transition*=min(2,10/(ids[k]-ids[k-1]))
        options=score[:,None]+transition
        path[k]=options.argmin(0);score=cost[k]+options.min(0)
    states=np.zeros(len(ids),int);states[-1]=score.argmin()
    for k in range(len(ids)-1,0,-1):states[k-1]=path[k,states[k]]
    selectedR=R[np.arange(len(ids)),states];selectedt=t[np.arange(len(ids)),states]
    np.savez(out/'keyframe_fits.npz',ids=ids,R=R,t=t,cost=cost,states=states,segments=segments)
    print('Selected yaw path',states.tolist(),flush=True)
    return ids,selectedR,selectedt,segments

def shared_rest(f,ids,R,t,segments,out):
    restR=[];restt=[];restcost=[]
    for a,b in segments:
        core=np.arange(a+(8 if a>0 else 0),b-(8 if b<f.N else 0))
        sample=core[np.unique(np.linspace(0,len(core)-1,min(20,len(core))).astype(int))]
        sel=(ids>=core[0])&(ids<=core[-1])
        if not sel.any():sel=np.abs(ids-(a+b)/2)<15
        rr=Rotation.from_matrix(R[sel]).mean().as_matrix()[None]
        tt=np.median(t[sel],axis=0)[None]
        rj,tj,c=f.fit(sample,rr,tt,steps=200,color=True,upright=True,groups=np.zeros(len(sample),int),lr=.012)
        restR.append(rj[0]);restt.append(tj[0]);restcost.append(float(c[0]))
        print(f'Static interval {a}:{b}, depth {tj[0,2]:.4f}, cost {c[0]:.4f}',flush=True)
    rawR=np.array(restR);rawt=np.array(restt)
    jointR,jointt,jointc=joint_table(f,segments,rawR,rawt)
    accepted=float(np.mean(jointc))<=float(np.mean(restcost))+.015 and np.max(jointc-np.array(restcost))<.04
    f.static_joint_used=bool(accepted)
    print(f'Joint table fit accepted={accepted}; costs independent={np.mean(restcost):.4f}, joint={np.mean(jointc):.4f}',flush=True)
    if accepted:restR,restt,restcost=jointR,jointt,jointc
    np.savez(out/'rest_fits.npz',segments=segments,R=restR,t=restt,cost=restcost,independent_R=rawR,independent_t=rawt,joint_accepted=accepted)
    return np.array(restR),np.array(restt)

def joint_table(f,segments,Rs,ts):
    """All resting poses share one fitted support normal and support plane."""
    count=len(segments);ids=[];groups=[]
    for j,(a,b) in enumerate(segments):
        core=np.arange(a+(8 if a>0 else 0),b-(8 if b<f.N else 0))
        sample=core[np.unique(np.linspace(0,len(core)-1,min(16,len(core))).astype(int))]
        ids.extend(sample);groups.extend(np.full(len(sample),j))
    ids=np.array(ids);groups=np.array(groups);gt=torch.as_tensor(groups,device='cuda:0')
    counts=tensor(np.bincount(groups))
    normals=Rs@f.bottom_normal
    down=normals.mean(0);down/=np.linalg.norm(down)
    base=Rotation.align_vectors([down],[f.bottom_normal])[0].as_matrix()
    angles=Rotation.from_matrix(base.T@Rs).as_rotvec()@f.bottom_normal
    basevar=torch.nn.Parameter(tensor(matrix_r6(base)[None]))
    yawvar=torch.nn.Parameter(tensor(angles))
    uv0=ts[:,:2]/ts[:,2,None]*np.diag(f.K)[:2]+f.K[:2,2]
    uv0=tensor(uv0);uvvar=torch.nn.Parameter(torch.zeros((count,2),device='cuda:0'))
    h0=float(np.median(ts@down));hvar=torch.nn.Parameter(torch.zeros(1,device='cuda:0'))
    axis=f.bottom_normal
    skew=tensor([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    ident=torch.eye(3,device='cuda:0');bn=tensor(axis)
    opt=torch.optim.Adam([basevar,yawvar,uvvar,hvar],lr=.01)
    best=np.inf;bestR=Rs.copy();bestt=ts.copy();bestc=np.full(count,10.)
    for step in range(220):
        opt.zero_grad(set_to_none=True)
        baser=r6_matrix(basevar)[0]
        yaw=ident+torch.sin(yawvar)[:,None,None]*skew+(1-torch.cos(yawvar))[:,None,None]*(skew@skew)
        R=baser[None]@yaw
        down=baser@bn
        uv=uv0+uvvar*5
        ray=torch.cat(((uv-tensor(f.K[:2,2]))/tensor(np.diag(f.K)[:2]),torch.ones((count,1),device='cuda:0')),dim=-1)
        depth=(h0+hvar*.05)/(ray@down).clamp_min(.05)
        t=ray*depth[:,None]
        sil,photo,_,_=f.losses(R[gt],t[gt],ids,True)
        costs=torch.zeros(count,device='cuda:0').index_add(0,gt,sil+1.8*photo)/counts
        costs=costs+.5*((down-tensor(f.down_cam))**2).sum()
        handle_x=R[:,0,0]-(t[:,0]/t[:,2])*R[:,2,0]
        costs=costs+2*tf.relu(handle_x+.35)**2
        loss=costs.sum()
        if step>5 and loss.item()<best:
            best=loss.item();bestR=R.detach().cpu().numpy();bestt=t.detach().cpu().numpy();bestc=costs.detach().cpu().numpy()
        loss.backward();opt.step()
        if step in [120,180]:
            for g in opt.param_groups:g['lr']*=.35
    return bestR,bestt,bestc

def dense_fit(f,ids,R,t,segments,out):
    restR,restt=shared_rest(f,ids,R,t,segments,out)
    # Insert reliable stationary anchors into the interpolation initialization.
    for k,i in enumerate(ids):
        for j,(a,b) in enumerate(segments):
            if a+5<=i<b-5 or (a==0 and i<b-5) or (b==f.N and i>=a+5):
                R[k]=restR[j];t[k]=restt[j]
    allids=np.arange(f.N)
    ri=Slerp(ids,Rotation.from_matrix(R))(allids).as_matrix()
    ti=np.column_stack([np.interp(allids,ids,t[:,j]) for j in range(3)])
    # Refine all images, retaining a weak prior to the globally continuous branch.
    rfull=ri.copy();tfull=ti.copy();cost=np.zeros(f.N)
    for a in range(0,f.N,24):
        b=min(a+24,f.N);ix=np.arange(a,b)
        rr,tt,c=f.fit(ix,ri[ix],ti[ix],steps=120,color=True,lr=.012,
            prior=(ri[ix],ti[ix],.018,1.0))
        rfull[ix]=rr;tfull[ix]=tt;cost[ix]=c
        if a%120==0:print(f'Dense refinement {b}/{f.N}, median cost {np.median(c):.4f}',flush=True)
    np.savez(out/'dense_unsmoothed.npz',R=rfull,t=tfull,cost=cost,initial_R=ri,initial_t=ti)
    # Suppress subpixel pose jitter. Project the averaged rotations back to SO(3).
    # Translation is smoothed in image-ray/depth coordinates so axial noise does
    # not drag the projection sideways.
    uv=tfull[:,:2]/tfull[:,2,None]
    uv=gaussian_filter1d(uv,1.0,axis=0,mode='nearest')
    depth=gaussian_filter1d(tfull[:,2],2.0,mode='nearest')
    tfinal=np.column_stack((uv*depth[:,None],depth))
    ravg=gaussian_filter1d(rfull,1.3,axis=0,mode='nearest')
    u,_,vh=np.linalg.svd(ravg)
    u[:,:,-1]*=np.linalg.det(u@vh)[:,None]
    rfinal=u@vh
    # The image identifies precisely when the mug leaves/returns to the table.
    # Leave a short blend at each boundary to preserve contact and settling motion.
    for j,(a,b) in enumerate(segments):
        ix=np.arange(a,b);w=np.ones(len(ix))
        if a>0:w=np.minimum(w,(ix-a)/8)
        if b<f.N:w=np.minimum(w,(b-1-ix)/8)
        w=np.clip(w,0,1)
        # On SO(3), interpolate relative rotation rather than matrix entries.
        rel=Rotation.from_matrix(rfinal[ix].transpose(0,2,1)@restR[j]).as_rotvec()
        rfinal[ix]=rfinal[ix]@Rotation.from_rotvec(rel*w[:,None]).as_matrix()
        tfinal[ix]=(1-w[:,None])*tfinal[ix]+w[:,None]*restt[j]
    return rfinal,tfinal

def evaluate(f,R,t):
    values=[]
    with torch.no_grad():
        for a in range(0,f.N,32):
            ids=np.arange(a,min(a+32,f.N))
            mask,_=f.render(tensor(R[ids]),tensor(t[ids]),ids,False)
            target=f.m_t[ids];vis=1-f.p_t[ids]
            intersection=(mask*target).sum((1,2))
            union=target.sum((1,2))+(mask*(1-target)*vis).sum((1,2))
            values.extend((intersection/union.clamp_min(1)).cpu().numpy().tolist())
    return np.array(values)

def save_preview(f,R,t,out):
    from PIL import Image,ImageDraw
    ids=np.unique(np.r_[np.arange(0,f.N,40),f.N-1])
    canvas=Image.new('RGB',(f.size*5,f.size*2*int(np.ceil(len(ids)/5))))
    with torch.no_grad():
        mask,col=f.render(tensor(R[ids]),tensor(t[ids]),ids,True)
    for k,i in enumerate(ids):
        im=f.images[i].copy();m=mask[k].cpu().numpy();c=col[k].cpu().numpy()
        edge=cv2.morphologyEx((m>.5).astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((2,2),np.uint8))>0
        im[edge]=[255,35,20]
        x=(k%5)*f.size;y=(k//5)*2*f.size
        a=Image.fromarray(im);ImageDraw.Draw(a).text((3,3),str(i),fill='red');canvas.paste(a,(x,y))
        canvas.paste(Image.fromarray((c*m[...,None]*255).clip(0,255).astype(np.uint8)),(x,y+f.size))
    canvas.save(out/'pose_preview.jpg')

def write_report(f,metrics,segments,out):
    minutes=metrics['runtime_seconds']/60
    intervals=', '.join(f'{a}–{b-1}' for a,b in segments)
    table_text=('A joint fit of all six placements was accepted: their bottom normals and support plane are shared, '
                'with separate azimuths and positions. The common plane is inferred from the mug and is not the floor.'
                if metrics['joint_table_fit_used'] else
                'The joint support-plane fit did not meet the image-loss acceptance criterion; separate stationary fits were retained.')
    report=f"""# Object trajectory refinement

Output: `object_poses_world.npz`, key `T_world_mesh`, shape `({f.N}, 4, 4)`, float64. These are mesh-file-to-world transforms for every frame in the released world frame. No alignment, symmetry equivalence, or change to the released camera transform is applied.

The supplied reconstruction rendered the mug upside down, often with the handle on the opposite side, and drifted on the table. I replaced its orientations and re-estimated metric translation by fitting the exact textured mesh to the video and SAM2 masks. The old translations provide starting depths only; their scale correction is not retained as a constraint. A weak inclination prior uses the two intermediate tracker axes only when they agree with each other, the mask axis, and the visible-base cue; their yaw and depth are not trusted. No ground truth, external data, network requests, model weights, or later reference humans were used. Inputs were only read.

The script searches 16 azimuths at 85 keyframes, with two initial inclinations during motion. Its objective combines antialiased silhouettes, distances to foreground, robust RGB residuals with affine illumination compensation, and soft physical constraints. Pixels covered by the person are excluded from background penalties. A temporal dynamic program chooses a continuous rotation branch. All {f.N} frames then receive individual image refinement. Rotation is projected onto SO(3); translation is lightly smoothed in image-ray/depth coordinates, and rotation is smoothed before the stationary intervals are blended in.

Rest intervals are detected from mask bottom height: {intervals}, inclusive. Each receives a shared pose fit over interior frames. {table_text} The mesh bottom-face normal is estimated automatically by robust planar fitting; it is slightly oblique to the mesh z-axis. The released ground normal is a soft orientation prior, not an imposed object height.

Explicit asset/video observations are documented as constants and criteria in `solve.py`: mesh +z points toward the base, mesh +x toward the handle; the base remains below the lid in the image; the handle stays to image-left; appreciably tilted lifts expose the base/barcode toward the camera. The last observation is applied with a soft 10–16 degree mask-axis gate only when the mug is raised. These prevent axial, handle-side, and forward/backward tilt ambiguities from exploiting hand occlusion. No frame-by-frame poses were entered manually.

Validation: mean occlusion-aware silhouette IoU changed from {metrics['mean_visible_IoU_before']:.4f} to {metrics['mean_visible_IoU_after']:.4f}; median changed from {metrics['median_visible_IoU_before']:.4f} to {metrics['median_visible_IoU_after']:.4f}. This is a consistency measure against the provided masks, not ground-truth ADD. `pose_preview.jpg` shows sampled image overlays and textured renders; `silhouette_metrics.npz` contains per-frame comparisons. Finiteness, frame coverage, float64 storage, homogeneous rows, rotation orthogonality, and positive unit determinants are checked before saving.

Confidence is high in correcting the longitudinal flip, handle side, and static jitter; moderate in metric depth and inclination during motion. Remaining uncertainty comes from the small image size (roughly 30 by 70 pixels), hand occlusion, specular metal, motion blur, and estimated intrinsics/extrinsics. The physical constraints are soft and image evidence is still imperfect. No ground-truth accuracy is claimed.

Reproduce from inputs alone:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

Final execution took {metrics['runtime_seconds']:.1f} seconds ({minutes:.2f} minutes) on one selected CUDA GPU (device 0). Seeds are fixed at 0; PyTorch deterministic algorithms are enabled and TF32/autotuning are disabled. No scratch files or previous fit outputs are read. Auxiliary NPZ/JSON files are diagnostics regenerated by the script. Total development GPU use remained below the 45-minute limit.
"""
    (out/'REPORT.md').write_text(report)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inputs',default='/work/inputs');parser.add_argument('--out',default='/work/outputs/object_poses_world.npz');parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--keyframes-only',action='store_true');args=parser.parse_args()
    start_time=time.time()
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    out=Path(args.out).parent;out.mkdir(parents=True,exist_ok=True)
    f=Fitter(args.inputs)
    if args.diagnostic:diagnostic(f,out);return
    result=keyframes(f,out)
    if args.keyframes_only:return
    R,t=dense_fit(f,*result,out)
    T=np.tile(np.eye(4),(f.N,1,1));T[:,:3,:3]=R;T[:,:3,3]=t
    Tw=np.einsum('ij,fjk->fik',f.wc,T).astype(np.float64)
    expected=json.loads((Path(args.inputs)/'episode.json').read_text())['frames']
    assert Tw.shape==(expected,4,4) and Tw.dtype==np.float64 and np.isfinite(Tw).all()
    assert np.allclose(Tw[:,3],np.array([0,0,0,1]),atol=1e-10)
    assert np.max(np.abs(np.linalg.det(Tw[:,:3,:3])-1))<2e-5
    assert np.max(np.abs(Tw[:,:3,:3].transpose(0,2,1)@Tw[:,:3,:3]-np.eye(3)))<2e-5
    np.savez(args.out,T_world_mesh=Tw)
    before=evaluate(f,f.T0[:,:3,:3],f.T0[:,:3,3]);after=evaluate(f,R,t)
    metrics={'frames':f.N,'mean_visible_IoU_before':float(before.mean()),'mean_visible_IoU_after':float(after.mean()),'median_visible_IoU_before':float(np.median(before)),'median_visible_IoU_after':float(np.median(after)),'runtime_seconds':time.time()-start_time,'joint_table_fit_used':f.static_joint_used,'bottom_normal_mesh':f.bottom_normal.tolist(),'camera_depth_min_max':[float(t[:,2].min()),float(t[:,2].max())]}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    np.savez(out/'silhouette_metrics.npz',before=before,after=after)
    save_preview(f,R,t,out)
    write_report(f,metrics,result[3],out)
    print(json.dumps(metrics,indent=2),flush=True)
if __name__=='__main__':main()
