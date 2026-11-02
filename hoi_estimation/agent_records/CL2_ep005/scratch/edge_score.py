"""Held-out validation: chamfer distance from rendered INTERNAL crease edges (box face boundaries, not the silhouette)
to Canny edges of the video frame (person pixels excluded)."""
import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2, torch, h5py
import nvdiffrast.torch as dr
from hoi_lib import *
d=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs'); R=Renderer(mesh)
fn=torch.tensor(np.asarray(mesh.face_normals),dtype=torch.float32,device='cuda')
S, O = trimesh_obb = None, None
import trimesh
Sobb,_=trimesh.bounds.oriented_bounds(mesh); Robb=np.linalg.inv(Sobb)[:3,:3]  # obb axes in mesh frame (columns)
fr=None
def get_frames():
    global fr
    if fr is None: fr=np.load('/work/scratch/frames_full_gray.npy',mmap_mode='r')
    return fr
f5=h5py.File('/work/inputs/masks.h5')
def labels(Tc_batch):
    T=torch.tensor(Tc_batch,dtype=torch.float32,device='cuda'); B=len(T)
    K=d['K']; H,W=d['H'],d['W']
    P=torch.tensor([[2*K[0,0]/W,0,2*(K[0,2]+0.5)/W-1,0],[0,2*K[1,1]/H,2*(K[1,2]+0.5)/H-1,0],[0,0,1.002,-0.1],[0,0,1,0]],dtype=torch.float32,device='cuda')
    vh=torch.cat([R.v,torch.ones_like(R.v[:,:1])],1); vc=torch.einsum('bij,nj->bni',T,vh); clip=torch.einsum('ij,bnj->bni',P,vc).contiguous()
    rast,_=dr.rasterize(R.glctx,clip,R.f,(H,W))
    tri=rast[...,3].long()-1; ok=tri>=0
    A=torch.tensor(np.concatenate([Robb,-Robb],1),dtype=torch.float32,device='cuda')  # 3x6 obb face normals (mesh frame)
    lab=torch.argmax(fn[tri.clamp_min(0)]@A,-1)+1; lab[~ok]=0
    return lab.cpu().numpy()
def score(Tc_all, idx, trunc=8.0):
    out=[]
    for s in range(0,len(idx),8):
        ii=idx[s:s+8]; labs=labels(Tc_all[ii])
        for j,i in enumerate(ii):
            g=get_frames()[i]; lab=labs[j]; pm=f5['person'][i]
            ok=lab>0
            e=np.zeros(lab.shape,bool); e[:,1:]|=(lab[:,1:]!=lab[:,:-1])&ok[:,1:]&ok[:,:-1]; e[1:,:]|=(lab[1:,:]!=lab[:-1,:])&ok[1:,:]&ok[:-1,:]
            # only keep long internal edges: remove small label islands by erosion of each label region
            e&=~(cv2.dilate(pm.astype(np.uint8),np.ones((9,9)))>0)
            ys,xs=np.where(ok); y0,y1,x0,x1=max(ys.min()-20,0),ys.max()+20,max(xs.min()-20,0),xs.max()+20
            crop=g[y0:y1,x0:x1]; crop=cv2.equalizeHist(crop)
            ed=cv2.Canny(cv2.GaussianBlur(crop,(5,5),1.2),40,100)
            dt=cv2.distanceTransform((ed==0).astype(np.uint8),cv2.DIST_L2,3)
            ee=e[y0:y1,x0:x1]
            out.append(np.minimum(dt[ee],trunc).mean() if ee.sum()>20 else np.nan)
    return np.array(out)
if __name__=='__main__':
    import cv2
    if not __import__('os').path.exists('/work/scratch/frames_full_gray.npy'):
        c=cv2.VideoCapture('/work/inputs/video.mp4'); a=[]
        while True:
            ok,f=c.read()
            if not ok: break
            a.append(cv2.cvtColor(f,cv2.COLOR_BGR2GRAY))
        np.save('/work/scratch/frames_full_gray.npy',np.stack(a))
