from inspect_baseline import *
from scipy.spatial.transform import Rotation
uv=torch.tensor(mesh.visual.uv,dtype=torch.float32,device='cuda')[None].contiguous();tex=torch.tensor(np.array(mesh.visual.material.baseColorTexture)[::-1].copy()/255,dtype=torch.float32,device='cuda')[None].contiguous()
def textured(t):
 t=torch.tensor(t,dtype=torch.float32,device='cuda');xyz=v@t[:3,:3].T+t[:3,3];x,y,z=xyz.unbind(-1);clip=torch.stack((2*K[0,0]/1536*x+(2*K[0,2]/1536-1)*z,2*K[1,1]/1152*y+(2*K[1,2]/1152-1)*z,1.002002*z-.020020,z),-1)[None].contiguous();rast,_=dr.rasterize(ctx,clip,tri,(1152,1536));tc,_=dr.interpolate(uv,rast,tri);color=dr.texture(tex,tc,filter_mode='linear');color=torch.where(rast[...,3:]>0,color,.6);return (color[0].cpu().numpy()*255).astype(np.uint8)
Tnew=np.linalg.inv(json.load(open(p+'camera.json'))['T_world_cam'])@np.load('/work/outputs/object_poses_world.npz')['T_world_mesh']
ids=[550,600,625,675,750,800];c=cv2.VideoCapture(p+'video.mp4');out=Image.new('RGB',(3*320,len(ids)*340),'white');d=ImageDraw.Draw(out)
with h5py.File(p+'masks.h5') as h:
 for j,i in enumerate(ids):
  c.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=c.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);m=h['object'][i];ys,xs=np.where(m);cx=(xs.min()+xs.max())//2;cy=(ys.min()+ys.max())//2;side=int(max(xs.max()-xs.min(),ys.max()-ys.min())*1.1);x=cx-side//2;y=cy-side//2
  for k,deg in enumerate([None,0,999]):
   if deg is None:pic=im
   else:
    t=T[i].copy() if deg==0 else Tnew[i].copy();pic=textured(t)
   out.paste(Image.fromarray(pic[y:y+side,x:x+side]).resize((320,320)),(k*320,j*340));d.text((k*320,j*340+320),f'{i}: '+('Video' if deg is None else ('Baseline' if deg==0 else 'Refined')),fill='black')
out.save('/work/outputs/orientation_evidence.jpg')
