import numpy as np, torch, trimesh, nvdiffrast.torch as dr, json
class Renderer:
    def __init__(self, mesh_path, device='cuda'):
        m=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
        self.mesh=m
        self.dev=device
        # merge duplicated vertices (same position and uv) to cut per-vertex cost
        key=np.hstack([m.vertices,m.visual.uv]).round(7)
        _,first,inv=np.unique(key,axis=0,return_index=True,return_inverse=True)
        inv=inv.reshape(-1)
        self.v=torch.tensor(m.vertices[first],dtype=torch.float32,device=device)
        self.f=torch.tensor(inv[m.faces],dtype=torch.int32,device=device)
        uv=np.array(m.visual.uv[first],dtype=np.float32).copy(); uv[:,1]=1-uv[:,1]
        self.uv=torch.tensor(uv,device=device)
        tex=np.array(m.visual.material.baseColorTexture.convert('RGB'),dtype=np.float32)/255.
        self.tex=torch.tensor(tex,device=device)[None].contiguous()
        self.glctx=dr.RasterizeCudaContext()
    def render(self, T_cam_mesh, K, H, W, x0=0, y0=0):
        """T (B,4,4) torch cam<-mesh; renders crop window [x0,x0+W)x[y0,y0+H) of full image. returns rgb(B,H,W,3), mask, depth"""
        T=torch.as_tensor(T_cam_mesh,dtype=torch.float32,device=self.dev)
        if T.ndim==2: T=T[None]
        B=T.shape[0]
        K=torch.as_tensor(K,dtype=torch.float32,device=self.dev)
        pc=self.v@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3]
        z=pc[...,2]
        u=K[0,0]*pc[...,0]/z+K[0,2]-x0; v=K[1,1]*pc[...,1]/z+K[1,2]-y0
        n,f=0.05,20.
        xc=2*u/W-1; yc=2*v/H-1
        zc=(z*(f+n)/(f-n)-2*f*n/(f-n))
        clip=torch.stack([xc*z,yc*z,zc,z],-1)
        rast,_=dr.rasterize(self.glctx,clip.contiguous(),self.f,(H,W))
        texc,_=dr.interpolate(self.uv,rast,self.f)
        col=dr.texture(self.tex,texc,filter_mode='linear')
        mask=rast[...,3]>0
        dep,_=dr.interpolate(z[...,None].contiguous(),rast,self.f)
        col=torch.where(mask[...,None],col,torch.zeros_like(col))
        return col,mask,dep[...,0]
def load_cam():
    c=json.load(open('/work/inputs/camera.json'))
    return np.array(c['K']),np.array(c['T_world_cam']),np.array(c['K_pipeline'])
