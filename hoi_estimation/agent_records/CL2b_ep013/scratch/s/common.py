import numpy as np, torch, json, trimesh, h5py, cv2
import nvdiffrast.torch as dr
INP='/work/inputs'
W,H=1536,1152
def load_cam():
    c=json.load(open(f'{INP}/camera.json'))
    return np.array(c['K']),np.array(c['T_world_cam']),np.array(c['K_pipeline'])
def load_mesh():
    return trimesh.load(f'{INP}/object_mesh.glb',force='scene',process=False).to_geometry()

class Renderer:
    def __init__(self, mesh, dev='cuda', tex_scale=0.5):
        self.dev=dev
        self.glctx=dr.RasterizeCudaContext()
        self.v=torch.tensor(np.asarray(mesh.vertices),dtype=torch.float32,device=dev)
        self.f=torch.tensor(np.asarray(mesh.faces),dtype=torch.int32,device=dev)
        uv=np.asarray(mesh.visual.uv).astype(np.float32).copy(); uv[:,1]=1-uv[:,1]
        self.uv=torch.tensor(uv,device=dev)
        tex=np.asarray(mesh.visual.material.baseColorTexture.convert('RGB'))
        if tex_scale!=1: tex=cv2.resize(tex,None,fx=tex_scale,fy=tex_scale,interpolation=cv2.INTER_AREA)
        self.tex=torch.tensor(tex[...,::-1].copy(),dtype=torch.float32,device=dev)[None]/255.  # BGR
    def clip(self, T, K, w, h, s=1.0):
        # T: (B,4,4) cam_from_mesh torch; K for full res; s: scale of output vs full res
        v=torch.cat([self.v,torch.ones_like(self.v[:,:1])],1)
        pc=torch.einsum('bij,nj->bni',T,v)[...,:3]
        Z=pc[...,2].clamp(min=1e-4)
        u=(K[0,0]*pc[...,0]/Z+K[0,2])*s; vv=(K[1,1]*pc[...,1]/Z+K[1,2])*s
        nx=2*(u+0.5)/w-1; ny=2*(vv+0.5)/h-1
        n,f=0.05,20.
        zc=Z*(f+n)/(f-n)-2*f*n/(f-n)
        return torch.stack([nx*Z,ny*Z,zc,Z],-1), pc
    def render(self, T, K, w, h, s=1.0, color=True):
        T=torch.as_tensor(T,dtype=torch.float32,device=self.dev)
        if T.dim()==2: T=T[None]
        K=torch.as_tensor(K,dtype=torch.float32,device=self.dev)
        cl,pc=self.clip(T,K,w,h,s)
        rast,_=dr.rasterize(self.glctx,cl.contiguous(),self.f,(h,w))
        mask=rast[...,3]>0
        depth,_=dr.interpolate(pc[...,2:3].contiguous(),rast,self.f)
        out={'mask':mask,'depth':depth[...,0]*mask}
        if color:
            uvi,_=dr.interpolate(self.uv[None],rast,self.f)
            col=dr.texture(self.tex,uvi,filter_mode='linear')
            out['color']=col*mask[...,None]
        out['rast']=rast
        return out
def read_frames(idx=None):
    fr=np.load('/work/scratch/frames_half.npy',mmap_mode='r')
    return fr if idx is None else fr[idx]
