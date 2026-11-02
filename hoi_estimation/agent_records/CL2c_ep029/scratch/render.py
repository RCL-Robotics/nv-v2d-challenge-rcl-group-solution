import numpy as np, torch, trimesh, nvdiffrast.torch as dr
class Renderer:
    def __init__(self, mesh_path, device='cuda', tex_max=2048):
        m=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
        self.mesh=m
        self.dev=device
        self.v=torch.tensor(np.asarray(m.vertices),dtype=torch.float32,device=device)
        self.f=torch.tensor(np.asarray(m.faces),dtype=torch.int32,device=device)
        uv=np.asarray(m.visual.uv).copy(); uv[:,1]=1-uv[:,1]
        self.uv=torch.tensor(uv,dtype=torch.float32,device=device)
        img=m.visual.material.baseColorTexture.convert('RGB')
        s=tex_max/max(img.size)
        if s<1: img=img.resize((int(img.size[0]*s),int(img.size[1]*s)))
        self.tex=torch.tensor(np.asarray(img),dtype=torch.float32,device=device)[None]/255.
        self.glctx=dr.RasterizeCudaContext(device=device)
    def render(self, T_cam_mesh, K, x0, y0, W, H, near=0.05, far=20.):
        """T_cam_mesh (B,4,4) numpy/torch; renders crop [x0,x0+W)x[y0,y0+H) of the image with intrinsics K.
        returns rgb (B,H,W,3), mask (B,H,W), depth (B,H,W)"""
        T=torch.as_tensor(T_cam_mesh,dtype=torch.float32,device=self.dev)
        if T.ndim==2: T=T[None]
        B=T.shape[0]
        fx,fy,cx,cy=K[0,0],K[1,1],K[0,2]-x0,K[1,2]-y0
        # OpenCV -> clip; pixel centers at +0.5
        P=torch.tensor([[2*fx/W,0,2*cx/W-1,0],[0,2*fy/H,2*cy/H-1,0],[0,0,(far+near)/(far-near),-2*far*near/(far-near)],[0,0,1,0]],dtype=torch.float32,device=self.dev)
        vh=torch.cat([self.v,torch.ones_like(self.v[:,:1])],1)
        vc=torch.einsum('bij,nj->bni',T,vh)
        clip=torch.einsum('ij,bnj->bni',P,vc)
        rast,_=dr.rasterize(self.glctx,clip.contiguous(),self.f,(H,W))
        texc,_=dr.interpolate(self.uv[None],rast,self.f)
        rgb=dr.texture(self.tex.expand(B,-1,-1,-1).contiguous(),texc,filter_mode='linear')
        mask=rast[...,3]>0
        z,_=dr.interpolate(vc[...,2:3].contiguous(),rast,self.f)
        rgb=torch.where(mask[...,None],rgb,torch.zeros_like(rgb))
        return rgb,mask,z[...,0]*mask

class SilRenderer:
    """differentiable silhouette renderer for a vertex subset (faces list)"""
    def __init__(self, verts, faces, device='cuda'):
        self.dev=device
        self.v=torch.tensor(np.asarray(verts),dtype=torch.float32,device=device)
        self.f=torch.tensor(np.asarray(faces),dtype=torch.int32,device=device)
        self.glctx=dr.RasterizeCudaContext(device=device)
    def proj(self,K,x0,y0,W,H,near=0.05,far=20.):
        fx,fy,cx,cy=K[0,0],K[1,1],K[0,2]-x0,K[1,2]-y0
        return torch.tensor([[2*fx/W,0,2*cx/W-1,0],[0,2*fy/H,2*cy/H-1,0],[0,0,(far+near)/(far-near),-2*far*near/(far-near)],[0,0,1,0]],dtype=torch.float32,device=self.dev)
    def render(self, R, t, Ps, H, W):
        """R (B,3,3) t (B,3) torch; Ps (B,4,4) projection per item; returns soft mask (B,H,W), depth"""
        vc=torch.einsum('bij,nj->bni',R,self.v)+t[:,None,:]
        vh=torch.cat([vc,torch.ones_like(vc[...,:1])],-1)
        clip=torch.einsum('bij,bnj->bni',Ps,vh).contiguous()
        rast,_=dr.rasterize(self.glctx,clip,self.f,(H,W))
        m=(rast[...,3:4]>0).float()
        m=dr.antialias(m,rast,clip,self.f)
        z,_=dr.interpolate(vc[...,2:3].contiguous(),rast,self.f)
        return m[...,0], z[...,0]
