"""Deterministic nvdiffrast rendering in OpenCV image coordinates."""
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

class Renderer:
    def __init__(self, mesh_path, device='cuda'):
        self.device=device
        self.mesh=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
        self.v=torch.as_tensor(np.array(self.mesh.vertices),dtype=torch.float32,device=device)
        self.f=torch.as_tensor(np.array(self.mesh.faces),dtype=torch.int32,device=device)
        self.uv=torch.as_tensor(np.array(self.mesh.visual.uv),dtype=torch.float32,device=device)
        tex=np.asarray(self.mesh.visual.material.baseColorTexture.convert('RGB'))[::-1].copy()/255.
        self.tex=torch.tensor(tex,dtype=torch.float32,device=device)[None]
        self.normals=torch.as_tensor(np.array(self.mesh.vertex_normals),dtype=torch.float32,device=device)
        self.ctx=dr.RasterizeCudaContext(device=device)
    def render(self,R,t,K,resolution=(160,160),texture=True):
        if not torch.is_tensor(R): R=torch.as_tensor(np.asarray(R),dtype=torch.float32,device=self.device)
        if not torch.is_tensor(t): t=torch.as_tensor(np.asarray(t),dtype=torch.float32,device=self.device)
        if not torch.is_tensor(K): K=torch.as_tensor(np.asarray(K),dtype=torch.float32,device=self.device)
        if R.ndim==2: R=R[None]
        if t.ndim==1: t=t[None]
        if K.ndim==2: K=K[None]
        v=self.v[None]@R.transpose(1,2)+t[:,None,:]
        z=v[:,:,2]
        h,w=resolution
        x=2*(K[:,0,0,None]*v[:,:,0]+(K[:,0,2,None]+0.5)*z)/w-z
        y=2*(K[:,1,1,None]*v[:,:,1]+(K[:,1,2,None]+0.5)*z)/h-z
        near,far=.01,20.
        clip=torch.stack([x,y,(far+near)/(far-near)*z-2*far*near/(far-near),z],-1).contiguous()
        rast,_=dr.rasterize(self.ctx,clip,self.f,resolution=resolution)
        mask=(rast[:,:,:,3:]>0).float()
        aa=dr.antialias(mask,rast,clip,self.f)
        if not texture: return aa[:,:,:,0]
        uv,_=dr.interpolate(self.uv[None],rast,self.f)
        rgb=dr.texture(self.tex,uv,filter_mode='linear')*mask
        normals,_=dr.interpolate(self.normals[None]@R.transpose(1,2),rast,self.f)
        normals=torch.nn.functional.normalize(normals,dim=-1)
        light=torch.tensor([-.3,-.5,-.8],device=self.device)
        diffuse=(normals*light).sum(-1,keepdim=True).clamp(0,1)
        rgb=rgb*(.65+.35*diffuse)
        rgb=dr.antialias(rgb,rast,clip,self.f)
        return rgb,aa[:,:,:,0]
