import numpy as np, torch, nvdiffrast.torch as dr, trimesh
class Renderer:
    def __init__(self, mesh, K, H=1152, W=1536, device='cuda'):
        self.dev=device
        self.v=torch.tensor(np.asarray(mesh.vertices),dtype=torch.float32,device=device)
        self.f=torch.tensor(np.asarray(mesh.faces),dtype=torch.int32,device=device)
        uv=np.asarray(mesh.visual.uv).copy(); uv[:,1]=1-uv[:,1]
        self.uv=torch.tensor(uv,dtype=torch.float32,device=device)
        tex=np.asarray(mesh.visual.material.baseColorTexture.convert('RGB')).astype(np.float32)/255.
        self.tex=torch.tensor(tex,device=device)[None].contiguous()
        self.glctx=dr.RasterizeCudaContext()
        self.K=K; self.H=H; self.W=W
    def proj(self, K, x0, y0, w, h, n=0.05, f=20.):
        # OpenCV camera -> clip for crop window [x0,x0+w]x[y0,y0+h]
        fx,fy,cx,cy=K[0,0],K[1,1],K[0,2]-x0,K[1,2]-y0
        P=np.array([[2*fx/w,0,(2*cx-w)/w,0],[0,2*fy/h,(2*cy-h)/h,0],[0,0,(f+n)/(f-n),-2*f*n/(f-n)],[0,0,1,0]],dtype=np.float32)
        return torch.tensor(P,device=self.dev)
    def render(self, T_cam_mesh, crop=None, res=None, K=None):
        """T_cam_mesh: (B,4,4) torch or np. crop=(x0,y0,w,h) in pixels; res=(rh,rw). returns rgb (B,rh,rw,3), mask, depth"""
        K=self.K if K is None else K
        if crop is None: crop=(0,0,self.W,self.H)
        x0,y0,w,h=crop
        if res is None: res=(int(h),int(w))
        T=torch.as_tensor(T_cam_mesh,dtype=torch.float32,device=self.dev)
        if T.dim()==2: T=T[None]
        P=self.proj(K,x0,y0,w,h)
        vc=self.v@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3]
        vh=torch.cat([vc,torch.ones_like(vc[...,:1])],-1)
        clip=vh@P.T
        rast,_=dr.rasterize(self.glctx,clip.contiguous(),self.f,resolution=list(res))
        uvi,_=dr.interpolate(self.uv[None],rast,self.f)
        col=dr.texture(self.tex,uvi,filter_mode="linear")
        mask=(rast[...,3]>0).float()
        z,_=dr.interpolate(vc[...,2:3].contiguous(),rast,self.f)
        return col*mask[...,None], mask, z[...,0]*mask
