import numpy as np,cv2,trimesh,torch,json,h5py
import nvdiffrast.torch as dr
from PIL import Image,ImageDraw
p='/work/inputs/'
cam=json.load(open(p+'camera.json'));K=np.array(cam['K']);a=np.load(p+'reconstruction.npz');T=np.linalg.inv(cam['T_world_cam'])@a['T_world_mesh'];mesh=trimesh.load(p+'object_mesh.glb',force='scene',process=False).to_geometry()
v=torch.tensor(mesh.vertices,dtype=torch.float32,device='cuda');tri=torch.tensor(mesh.faces.astype('int32'),device='cuda');ctx=dr.RasterizeCudaContext()
def render(t,H=576,W=768):
 t=torch.tensor(t,dtype=torch.float32,device='cuda');xyz=v@t[:3,:3].T+t[:3,3];x,y,z=xyz.unbind(-1);clip=torch.stack((2*K[0,0]/1536*x+(2*K[0,2]/1536-1)*z,2*K[1,1]/1152*y+(2*K[1,2]/1152-1)*z,1.002002*z-.020020,z),-1)[None];rast,_=dr.rasterize(ctx,clip,tri,(H,W));return (rast[0,:,:,3]>0).cpu().numpy()
if __name__=='__main__':
 ids=list(range(0,993,50))+[992];c=cv2.VideoCapture(p+'video.mp4');out=Image.new('RGB',(5*420,5*420),'white');d=ImageDraw.Draw(out)
 with h5py.File(p+'masks.h5') as h:
  for j,i in enumerate(ids):
   c.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=c.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);im=cv2.resize(im,(768,576));gt=h['object'][i,::2,::2];m=render(T[i]);inter=(m&gt).sum();print(i,'iou',inter/(m|gt).sum())
   im[m]=(.55*im[m]+.45*np.array([255,30,30])).astype('uint8');ys,xs=np.where(m|gt);x=int((xs.min()+xs.max())/2);y=int((ys.min()+ys.max())/2);im=im[max(0,y-100):min(576,y+100),max(0,x-100):min(768,x+100)];out.paste(Image.fromarray(im).resize((400,400)),(j%5*420,j//5*420));d.text((j%5*420,j//5*420+400),str(i),fill='black')
 out.save('/work/scratch/baseline.jpg')
