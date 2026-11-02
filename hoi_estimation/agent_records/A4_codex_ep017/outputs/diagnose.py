import os,json,cv2,numpy as np,trimesh,h5py,torch
import nvdiffrast.torch as dr
from PIL import Image,ImageDraw
class Renderer:
 def __init__(self,root='/work/inputs'):
  m=trimesh.load(root+'/object_mesh.glb',force='scene',process=False).to_geometry(); self.mesh=m
  self.v=torch.tensor(m.vertices,dtype=torch.float32,device='cuda'); self.f=torch.tensor(m.faces.astype(np.int32),device='cuda'); self.uv=torch.tensor(m.visual.uv,dtype=torch.float32,device='cuda')[None]
  self.uv[:,:,1]=1-self.uv[:,:,1]
  tex=np.array(m.visual.material.baseColorTexture.convert('RGB')); print('texture',tex.shape)
  self.tex=torch.tensor(tex.copy(),device='cuda',dtype=torch.float32)[None]/255
  self.ctx=dr.RasterizeCudaContext()
 def render(self,T,K,H,W):
  T=torch.as_tensor(T,dtype=torch.float32,device='cuda'); T=T[None] if T.ndim==2 else T
  v=self.v[None]@T[:,:3,:3].transpose(1,2)+T[:,None,:3,3]
  x=v[:,:,0]*2*K[0,0]/W+v[:,:,2]*(2*K[0,2]/W-1); y=v[:,:,1]*2*K[1,1]/H+v[:,:,2]*(2*K[1,2]/H-1)
  z=v[:,:,2]*1.002002-0.02002002
  clip=torch.stack([x,y,z,v[:,:,2]],-1)
  rast,_=dr.rasterize(self.ctx,clip.contiguous(),self.f,(H,W)); uv,_=dr.interpolate(self.uv,rast,self.f)
  color=dr.texture(self.tex,uv,filter_mode='linear',boundary_mode='clamp'); mask=rast[:,:,:,3]>0
  return (color*255).byte().cpu().numpy(),mask.cpu().numpy()
if __name__=='__main__':
 root='/work/inputs'; r=np.load(root+'/reconstruction.npz'); cam=json.load(open(root+'/camera.json')); K=np.array(cam['K']); C=np.linalg.inv(cam['T_world_cam']); T=C@r['T_world_mesh']; ren=Renderer()
 fs=[0,90,110,117,130,150,180,210,235,260,290,320,350,400,439]
 cap=cv2.VideoCapture(root+'/video.mp4'); h=h5py.File(root+'/masks.h5'); canvas=Image.new('RGB',(5*360,3*310),'white'); draw=ImageDraw.Draw(canvas)
 for j,f in enumerate(fs):
  cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,f); ok,bgr=cap.read(); rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
  col,mask=ren.render(T[f],K,1152,1536); col=col[0];mask=mask[0]
  obs=h['object'][f]; ys,xs=np.where(obs|mask); x0,x1=xs.min()-25,xs.max()+26; y0,y1=ys.min()-20,ys.max()+21
  original=rgb.copy();cv2.drawContours(original,cv2.findContours(obs.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,255,0),1)
  overlay=rgb.copy(); overlay[mask]=(rgb[mask]*.4+col[mask]*.6).astype('uint8')
  def crop(a):return Image.fromarray(a[max(0,y0):min(1152,y1),max(0,x0):min(1536,x1)]).resize((180,280))
  x=j%5*360;y=j//5*310;canvas.paste(crop(original),(x,y+25));canvas.paste(crop(overlay),(x+180,y+25));draw.text((x+5,y+5),f'{f}: observed / baseline overlay',fill='black')
 canvas.save('/work/outputs/baseline_review.jpg')
