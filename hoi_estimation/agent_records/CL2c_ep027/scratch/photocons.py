import sys, numpy as np, json, h5py, torch, cv2, trimesh
sys.path.insert(0,'/work/outputs')
from hoilib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam'])
def load_cam_traj(p):
    if p=='base': T=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
    else: T=np.load(p)['T_world_mesh']
    return np.linalg.inv(Twc)[None]@T
fr=np.load('/work/scratch/frames_all.npy',mmap_mode='r'); H5=h5py.File('/work/inputs/masks.h5')
mesh=TexMesh('/work/inputs/object_mesh.glb')
tm=trimesh.Trimesh(mesh.vw.cpu().numpy(),mesh.fw.cpu().numpy(),process=False)
pts,fid=trimesh.sample.sample_surface_even(tm,20000,seed=0); nrm=tm.face_normals[fid]
print('points',len(pts))
def accumulate(Tc, frames):
    S=np.zeros((len(pts),3)); C=np.zeros(len(pts))
    P=torch.tensor(pts,dtype=torch.float32,device='cuda'); N=torch.tensor(nrm,dtype=torch.float32,device='cuda')
    for i in frames:
        om=H5['object'][i]
        if om.sum()<150: continue
        T=Tc[i]; p=K@T[:3,3]; u0,v0=p[:2]/p[2]
        ys,xs=np.nonzero(om); x0=max(0,xs.min()-20); y0=max(0,ys.min()-20); x1=min(1536,xs.max()+20); y1=min(1152,ys.max()+20)
        W,H=x1-x0,y1-y0
        Tt=torch.tensor(T,dtype=torch.float32,device='cuda')[None]
        _,msk,dep=render(mesh,Tt,K,H,W,crop=(x0,y0,W,H),need_rgb=False)
        dep=dep[0].cpu().numpy()
        pc=(P@Tt[0,:3,:3].T+Tt[0,:3,3]).cpu().numpy(); nc=(N@Tt[0,:3,:3].T).cpu().numpy()
        u=K[0,0]*pc[:,0]/pc[:,2]+K[0,2]-x0; v=K[1,1]*pc[:,1]/pc[:,2]+K[1,2]-y0
        ui=np.round(u).astype(int); vi=np.round(v).astype(int)
        ok=(ui>=0)&(ui<W)&(vi>=0)&(vi<H)
        ui=np.clip(ui,0,W-1); vi=np.clip(vi,0,H-1)
        view=pc/np.linalg.norm(pc,axis=1,keepdims=True)
        ok&=(np.abs(dep[vi,ui]-pc[:,2])<0.008)&((nc*view).sum(1)<-0.25)
        omc=cv2.erode(om[y0:y1,x0:x1].astype(np.uint8),np.ones((3,3)))>0
        pmc=cv2.dilate(H5['person'][i][y0:y1,x0:x1].astype(np.uint8),np.ones((5,5)))>0
        ok&=omc[vi,ui]&~pmc[vi,ui]
        img=cv2.GaussianBlur(np.asarray(fr[i,y0:y1,x0:x1]).astype(np.float32),(0,0),1.0)
        col=img[vi,ui]
        # per-frame normalisation of the object's colour (gain), to reduce lighting/exposure effects
        if ok.sum()<20: continue
        col=col/col[ok].mean(0,keepdims=True)
        S[ok]+=col[ok]; C[ok]+=1
    return S,C
if __name__=='__main__':
    traj=sys.argv[1]; Tc=load_cam_traj(traj)
    segs=[tuple(map(int,s.split(':'))) for s in sys.argv[2].split(',')]
    res={}
    for a,b in segs:
        S,C=accumulate(Tc,range(a,b)); res[(a,b)]=(S,C); print('seg',a,b,'points seen',(C>=3).sum(),flush=True)
    np.save('/work/scratch/pc_%s.npy'%traj.replace('/','_'),{'pts':pts,'res':res},allow_pickle=True)
