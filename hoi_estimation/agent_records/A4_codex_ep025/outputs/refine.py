import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json,time,cv2,h5py,numpy as np,torch
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation
from render_utils import Renderer

def make_data(inputs,ids,K,T,size=160):
    cap=cv2.VideoCapture(str(inputs/'video.mp4')); h=h5py.File(inputs/'masks.h5'); boxes=[];images=[];masks=[];people=[];ids=set(map(int,ids));frame=0;times=time.time()
    while cap.isOpened():
        ok,im=cap.read()
        if not ok:break
        if frame in ids:
            uv=K@T[frame,:3,3];uv=uv[:2]/uv[2]
            x,y=np.round(uv-size/2).astype(int);x=int(np.clip(x,0,im.shape[1]-size));y=int(np.clip(y,0,im.shape[0]-size))
            images.append(im[y:y+size,x:x+size,::-1].copy());masks.append(h['object'][frame,y:y+size,x:x+size]);people.append(h['person'][frame,y:y+size,x:x+size]);boxes.append([x,y,1.])
        frame+=1
    cap.release();h.close()
    images=np.stack(images);masks=np.stack(masks);people=np.stack(people);boxes=np.array(boxes)
    # Ignore people outside the provided object segmentation, permitting occlusion.
    valid=(~people)|masks
    # Material-color cues are used only where the released object mask is present.
    rgb=images.astype(float)/255; blue=(rgb[...,2]>1.15*rgb[...,1])&(rgb[...,2]>1.45*rgb[...,0])&(rgb[...,2]>.18)&masks
    cyan=(rgb[...,1]>1.22*rgb[...,0])&(rgb[...,2]>.92*rgb[...,0])&(rgb[...,1]>.84*rgb[...,2])&(rgb[...,1]>.23)&masks
    # Soft body probability from color; cap pixels are darker / non-cyan.
    nonbody=masks&(~cyan)
    dist=np.stack([distance_transform_edt(~m)/12 for m in masks]).clip(0,2).astype('float32')
    print('loaded crops',len(boxes),'seconds',round(time.time()-times,2),flush=True)
    return {'ids':np.array(sorted(ids)),'boxes':boxes,'images':images,'masks':masks,'valid':valid,'blue':blue,'cyan':cyan,'nonbody':nonbody,'dist':dist}

def to_torch(data):
    return {k:torch.as_tensor(data[k],device='cuda',dtype=torch.float32) for k in ['boxes','images','masks','valid','blue','cyan','nonbody','dist']}

def losses(renderer,R,t,d,texture=False):
    m,attr,col=renderer.render(R,t,d['boxes'],texture=texture,antialias=True)
    obs=d['masks'];v=d['valid'];area=obs.sum((1,2)).clamp(min=100)
    inter=(m*obs).sum((1,2));union=(m*v).sum((1,2))+area-inter
    iou=inter/union.clamp(min=1)
    outline=(m*d['dist']*v).sum((1,2))/area
    # Cap/body discrepancy, positive only on confidently cyan observed pixels.
    cap=torch.clamp((attr[...,0]-.108)/.010,0,1)*m
    cap_wrong=(cap*d['cyan']).sum((1,2))/area
    # The blue latch occupies the top half of the mesh.
    blue_wrong=(torch.clamp((.035-attr[...,0])/.06,0,1)*m*d['blue']).sum((1,2))/area
    l=1-iou+.12*outline+.9*cap_wrong+2*blue_wrong
    if texture:
        # Color-invariant blue ratio gives a stable asymmetric feature.
        br=torch.sigmoid((col[...,2]-1.16*col[...,1]-.015)*30)*torch.sigmoid((col[...,2]-1.4*col[...,0])*30)*m
        ob=d['blue'];both=(m*obs); n=ob.sum((1,2));
        berror=((br-ob).abs()*both).sum((1,2))/area
        blue_union=(br*v+ob-br*ob).sum((1,2)).clamp(min=20)
        blue_inter=(br*ob).sum((1,2))
        l=l+.50*(1-blue_inter/blue_union)*(n>12)
    return l,torch.stack([iou,cap_wrong,blue_wrong,outline],1),m

def rotmat(w):
    theta=torch.linalg.vector_norm(w,dim=-1,keepdim=True)
    a=torch.sinc(theta/torch.pi);b=.5*torch.sinc(theta/(2*torch.pi))**2
    x,y,z=w.unbind(-1);zero=torch.zeros_like(x);skew=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(-1,3,3)
    return torch.eye(3,device=w.device)[None]+a[...,None]*skew+b[...,None]*(skew@skew)

def fixed_radius_t(uv,rad,K):
    ray=torch.cat([(uv-torch.tensor(K[:2,2],device='cuda',dtype=torch.float32))/torch.tensor([K[0,0],K[1,1]],device='cuda',dtype=torch.float32),torch.ones((len(uv),1),device='cuda')],1)
    return ray/torch.linalg.vector_norm(ray,dim=-1,keepdim=True)*rad[:,None]

def optimize(renderer,Rs,ts,d,steps=80,texture=False,rot_reg=.006,shift_reg=.003):
    R0=torch.as_tensor(Rs,device='cuda',dtype=torch.float32);t0=torch.as_tensor(ts,device='cuda',dtype=torch.float32);w=torch.zeros((len(Rs),3),device='cuda',requires_grad=True);shift=torch.zeros((len(Rs),2),device='cuda',requires_grad=True)
    K=renderer.K;uv0=t0[:,:2]/t0[:,2:]*torch.tensor([K[0,0],K[1,1]],device='cuda')+torch.tensor(K[:2,2],device='cuda');rad=torch.linalg.vector_norm(t0,dim=-1)
    opt=torch.optim.Adam([{'params':[w],'lr':.025},{'params':[shift],'lr':.25}]);best=np.full(len(Rs),np.inf);bestR=Rs.copy();bestt=ts.copy();beststats=np.zeros((len(Rs),4))
    for step in range(steps+1):
        R=rotmat(w)@R0;t=fixed_radius_t((uv0+shift).float(),rad,K);l,stats,_=losses(renderer,R,t,d,texture=texture)
        reg=rot_reg*(w*w).sum(-1)+shift_reg*(shift/4).square().sum(-1)
        score=l+reg
        with torch.no_grad():
            ss=score.detach().cpu().numpy();take=ss<best;best[take]=ss[take];bestR[take]=R.detach().cpu().numpy()[take];bestt[take]=t.detach().cpu().numpy()[take];beststats[take]=stats.detach().cpu().numpy()[take]
        if step==steps:break
        opt.zero_grad();score.sum().backward();opt.step()
        with torch.no_grad():shift.clamp_(-10,10)
    return bestR,bestt,best,beststats

if __name__=='__main__':
    from pathlib import Path
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--mode',default='scan');p.add_argument('--ids',default='');p.add_argument('--steps',type=int,default=100);args=p.parse_args();torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4);cv2.setNumThreads(1)
    inputs=Path('/work/inputs');out=Path('/work/outputs');c=json.load(open(inputs/'camera.json'));K=np.array(c['K']);a=np.load(inputs/'reconstruction.npz');T=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];F=len(T);r=Renderer(inputs/'object_mesh.glb',K);
    ids=np.arange(F) if args.mode=='scan' else np.array([int(x) for x in args.ids.split(',')]);data=make_data(inputs,ids,K,T);d=to_torch(data);scores=[];stats=[]
    start=time.time()
    if args.mode=='scan':
        for j in range(0,len(ids),12):
            idx=ids[j:j+12];dd={k:v[j:j+12] for k,v in d.items()};
            with torch.no_grad():l,s,_=losses(r,T[idx,:3,:3],T[idx,:3,3],dd,texture=True)
            scores.extend(l.cpu().numpy());stats.extend(s.cpu().numpy())
        np.savez_compressed(out/'baseline_scores.npz',scores=scores,stats=stats,**data)
        scores=np.array(scores);stats=np.array(stats);print('iou percentiles',np.percentile(stats[:,0],[0,5,10,25,50,75,95,100]));print('worst',sorted(zip(ids,scores),key=lambda x:-x[1])[:45])
    else:
        Rs=[];ts=[];sc=[];st=[]
        for j in range(0,len(ids),8):
            idx=ids[j:j+8];n=len(idx);dd={k:torch.cat([v[j:j+n]]*2) for k,v in d.items()};R0=np.concatenate([T[idx,:3,:3],a['T_cam_mesh_foundationpose'][idx,:3,:3]]);t0=np.concatenate([T[idx,:3,3]]*2)
            R,t,s,ss=optimize(r,R0,t0,dd,steps=args.steps,texture=True);Rs.append(R.reshape(2,n,3,3));ts.append(t.reshape(2,n,3));sc.append(s.reshape(2,n));st.append(ss.reshape(2,n,4));print('optimized',idx.tolist(),np.round(s,3),flush=True)
        np.savez_compressed(out/'trial_results.npz',ids=ids,R=np.concatenate(Rs,1),t=np.concatenate(ts,1),scores=np.concatenate(sc,1),stats=np.concatenate(st,1),**{k:v for k,v in data.items() if k!='ids'})
    print('render/optimization seconds',time.time()-start)
