import numpy as np, torch, trimesh, json, nvdiffrast.torch as dr
dev='cuda'
def load_mesh(path='/work/inputs/object_mesh.glb'):
    m=trimesh.load(path,force='scene',process=False).to_geometry()
    v=np.asarray(m.vertices,np.float32); f=np.asarray(m.faces,np.int32)
    uv=np.asarray(m.visual.uv,np.float32)
    mat=m.visual.material
    tex=np.asarray(mat.baseColorTexture.convert('RGB'),np.float32)/255.
    return m,v,f,uv,tex
class Renderer:
    def __init__(self,K,W=1536,H=1152,path='/work/inputs/object_mesh.glb'):
        self.m,v,f,uv,tex=load_mesh(path)
        self.v=torch.tensor(v,device=dev); self.f=torch.tensor(f,device=dev)
        self.uv=torch.tensor(uv,device=dev); self.uv[:,1]=1-self.uv[:,1]
        self.tex=torch.tensor(tex,device=dev)[None].contiguous()
        self.glctx=dr.RasterizeCudaContext()
        self.K=np.asarray(K,np.float64); self.W=W; self.H=H
    def proj(self,K,W,H,n=0.05,fa=20.):
        fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
        return torch.tensor([[2*fx/W,0,2*cx/W-1,0],[0,2*fy/H,2*cy/H-1,0],[0,0,(fa+n)/(fa-n),-2*fa*n/(fa-n)],[0,0,1,0]],dtype=torch.float32,device=dev)
    def render(self,T_cam_mesh,K=None,W=None,H=None,crop=None):
        """T_cam_mesh (B,4,4) torch/np. crop=(x0,y0,w,h) render subwindow at full res scale. returns rgb(B,H,W,3), mask, depth"""
        K=self.K if K is None else np.asarray(K,np.float64); W=W or self.W; H=H or self.H
        if crop is not None:
            x0,y0,w,h=crop; K=K.copy(); K[0,2]-=x0; K[1,2]-=y0; W,H=w,h
        T=torch.as_tensor(np.asarray(T_cam_mesh) if not torch.is_tensor(T_cam_mesh) else T_cam_mesh,dtype=torch.float32,device=dev)
        if T.ndim==2: T=T[None]
        P=self.proj(K,W,H)
        vh=torch.cat([self.v,torch.ones_like(self.v[:,:1])],1)
        vc=torch.einsum('bij,nj->bni',T,vh)
        clip=torch.einsum('ij,bnj->bni',P,vc)
        # nvdiffrast: y up in clip -> image row 0 at bottom; flip
        clip=clip*torch.tensor([1,-1,1,1],device=dev,dtype=torch.float32) 
        rast,_=dr.rasterize(self.glctx,clip.contiguous(),self.f,(H,W))
        rast=torch.flip(rast,[1])
        texc,_=dr.interpolate(self.uv[None],rast,self.f)
        col=dr.texture(self.tex,texc,filter_mode='linear')
        mask=rast[...,3]>0
        z,_=dr.interpolate(vc[...,2:3].contiguous(),rast,self.f)
        return col*mask[...,None], mask, z[...,0]*mask
def load_all():
    cam=json.load(open('/work/inputs/camera.json'))
    r=np.load('/work/inputs/reconstruction.npz')
    return cam,r
