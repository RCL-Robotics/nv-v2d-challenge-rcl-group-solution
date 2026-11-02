import numpy as np, torch, trimesh, nvdiffrast.torch as dr, json
from PIL import Image
class Renderer:
    def __init__(self, mesh_path='/work/inputs/object_mesh.glb', device='cuda'):
        m=trimesh.load(mesh_path, force='scene', process=False).to_geometry()
        self.mesh=m
        self.dev=device
        self.v=torch.tensor(m.vertices,dtype=torch.float32,device=device)
        self.f=torch.tensor(m.faces,dtype=torch.int32,device=device)
        uv=np.array(m.visual.uv,dtype=np.float32).copy(); uv[:,1]=1-uv[:,1]
        self.uv=torch.tensor(uv,device=device)
        tex=np.asarray(m.visual.material.baseColorTexture.convert('RGB'),dtype=np.float32)/255.
        self.tex=torch.tensor(tex,device=device)[None].contiguous()
        self.glctx=dr.RasterizeCudaContext()
        vn=np.asarray(m.vertex_normals,dtype=np.float32); self.vn=torch.tensor(vn,device=device)
    def render(self, T_cam_mesh, K, H, W, near=0.05, far=20.0, want_rgb=True):
        """T_cam_mesh: (B,4,4) numpy/torch (OpenCV). Returns mask (B,H,W), depth (B,H,W), rgb (B,H,W,3)"""
        T=torch.as_tensor(np.asarray(T_cam_mesh),dtype=torch.float32,device=self.dev)
        if T.ndim==2: T=T[None]
        B=T.shape[0]
        K=np.asarray(K)
        fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
        # OpenCV -> clip: x_ndc = (fx*x/z+cx)/W*2-1 ; y_ndc = (fy*y/z+cy)/H*2-1 (we render with y down => flip later none, nvdiffrast row 0 = y_ndc -1)
        P=torch.tensor([[2*fx/W,0,(2*cx+1)/W-1,0],[0,2*fy/H,(2*cy+1)/H-1,0],[0,0,(far+near)/(far-near),-2*far*near/(far-near)],[0,0,1,0]],dtype=torch.float32,device=self.dev)
        vh=torch.cat([self.v,torch.ones_like(self.v[:,:1])],1)
        vc=torch.einsum('bij,nj->bni',T,vh)
        clip=torch.einsum('ij,bnj->bni',P,vc)
        rast,_=dr.rasterize(self.glctx,clip.contiguous(),self.f,(H,W))
        mask=rast[...,3]>0
        depth,_=dr.interpolate(vc[...,2:3].contiguous(),rast,self.f)
        depth=depth[...,0]*mask
        rgb=None
        if want_rgb:
            uvi,uvd=dr.interpolate(self.uv[None],rast,self.f,rast_db=None)
            rgb=dr.texture(self.tex,uvi,filter_mode='linear')*mask[...,None]
        return mask,depth,rgb
def load_cam():
    c=json.load(open('/work/inputs/camera.json'))
    return np.array(c['K']),np.array(c['K_pipeline']),np.array(c['T_world_cam'])
