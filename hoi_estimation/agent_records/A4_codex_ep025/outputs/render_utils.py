"""Small OpenCV-camera mesh renderer, using the installed nvdiffrast."""
import numpy as np
import torch
import nvdiffrast.torch as dr
import trimesh

class Renderer:
 def __init__(self,mesh_path,K,size=160):
  self.device='cuda';self.K=np.asarray(K);self.size=size
  mesh=trimesh.load(mesh_path,force='scene',process=False).to_geometry()
  mesh.merge_vertices(merge_tex=False,merge_norm=True)
  self.mesh=mesh; self.v=torch.as_tensor(np.asarray(mesh.vertices).copy(),device=self.device,dtype=torch.float32)
  self.faces=torch.as_tensor(np.asarray(mesh.faces).copy(),device=self.device,dtype=torch.int32)
  uv=np.asarray(mesh.visual.uv).copy();uv[:,1]=1-uv[:,1]
  self.uv=torch.as_tensor(uv,device=self.device,dtype=torch.float32)[None]
  tex=np.asarray(mesh.visual.material.baseColorTexture.convert('RGB')).copy()/255.
  self.tex=torch.as_tensor(tex,device=self.device,dtype=torch.float32)[None].contiguous()
  self.ctx=dr.RasterizeCudaContext()
  # Smooth semantic label: cyan body below the cap; cap begins near z=0.10 m.
  self.attr=torch.stack([self.v[:,2],torch.clamp((self.v[:,2]-.097)/.012,0,1)],-1)[None].contiguous()
 def render(self,R,t,boxes,texture=False,antialias=False):
  if not torch.is_tensor(R):R=torch.as_tensor(R,device=self.device,dtype=torch.float32)
  if not torch.is_tensor(t):t=torch.as_tensor(t,device=self.device,dtype=torch.float32)
  boxes=torch.as_tensor(boxes,device=self.device,dtype=torch.float32)
  cam=self.v[None]@R.transpose(1,2)+t[:,None,:]
  x,y,z=cam.unbind(-1);x0,y0,scale=boxes.unbind(-1);S=self.size
  xx=2*(self.K[0,0]*x+(self.K[0,2]-x0[:,None])*z)/(scale[:,None]*S)-z
  yy=2*(self.K[1,1]*y+(self.K[1,2]-y0[:,None])*z)/(scale[:,None]*S)-z
  clip=torch.stack([xx,yy,1.00002*z-.002,z],-1).contiguous()
  rast,_=dr.rasterize(self.ctx,clip,self.faces,resolution=[S,S])
  mask=(rast[...,3:] > 0).float()
  attr,_=dr.interpolate(self.attr,rast,self.faces)
  if antialias: mask=dr.antialias(mask, rast,clip,self.faces)
  color=None
  if texture:
   uv,_=dr.interpolate(self.uv,rast,self.faces);color=dr.texture(self.tex,uv,filter_mode='linear')*mask
  return mask[...,0],attr,color
