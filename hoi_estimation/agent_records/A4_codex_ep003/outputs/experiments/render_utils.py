import numpy as np, torch, trimesh
import nvdiffrast.torch as dr
class Renderer:
 def __init__(self,path,K,size=128):
  self.device='cuda';self.size=size;self.K=np.array(K);m=trimesh.load(path,force='scene',process=False).to_geometry();self.mesh=m
  self.v=torch.tensor(np.asarray(m.vertices),device='cuda',dtype=torch.float32);self.faces=torch.tensor(np.array(m.faces,dtype=np.int32),device='cuda');self.uv=torch.tensor(np.array(m.visual.uv),device='cuda',dtype=torch.float32)[None]
  mat=m.visual.material; im=getattr(mat,'baseColorTexture',None)
  if im is None: im=mat.image
  self.tex=torch.tensor(np.array(im.convert('RGB'))[::-1].copy(),device='cuda',dtype=torch.float32)[None]/255
  self.ctx=dr.RasterizeCudaContext()
 def render(self,R,t,origin,aa=True,scale=None):
  if not torch.is_tensor(R): R=torch.tensor(R,device='cuda',dtype=torch.float32)
  if not torch.is_tensor(t): t=torch.tensor(t,device='cuda',dtype=torch.float32)
  if not torch.is_tensor(origin): origin=torch.tensor(origin,device='cuda',dtype=torch.float32)
  if R.ndim==2:R=R[None]
  if t.ndim==1:t=t[None]
  if origin.ndim==1:origin=origin[None]
  v=self.v[None]@R.transpose(1,2)+t[:,None];z=v[:,:,2];x=self.K[0,0]*v[:,:,0]+(self.K[0,2]-origin[:,0:1])*z;y=self.K[1,1]*v[:,:,1]+(self.K[1,2]-origin[:,1:2])*z
  if scale is not None:
   cx=(self.K[0,0]*t[:,0]/t[:,2]+self.K[0,2]-origin[:,0])[:,None]*z;cy=(self.K[1,1]*t[:,1]/t[:,2]+self.K[1,2]-origin[:,1])[:,None]*z
   x=cx+scale[:,None]*(x-cx);y=cy+scale[:,None]*(y-cy)
  clip=torch.stack([2*x/self.size-z,2*y/self.size-z,(10.01/9.99)*z-0.2002/9.99,z],-1)
  rast,_=dr.rasterize(self.ctx,clip,self.faces,resolution=[self.size,self.size]);uv,_=dr.interpolate(self.uv,rast,self.faces);color=dr.texture(self.tex,uv,filter_mode='linear');mask=(rast[:,:,:,3:]>0).float();color=color*mask
  if aa:mask=dr.antialias(mask,rast,clip,self.faces);color=dr.antialias(color,rast,clip,self.faces)
  return color,mask
