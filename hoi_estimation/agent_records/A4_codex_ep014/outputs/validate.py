"""Validate the submitted arrays and produce visual evidence (optional)."""
import numpy as np,json,cv2,h5py,torch
from PIL import Image,ImageDraw
from inspect_baseline import render,T
from scipy.spatial.transform import Rotation
p='/work/inputs/';o='/work/outputs/'
base=np.load(p+'reconstruction.npz')['T_world_mesh'];new=np.load(o+'object_poses_world.npz')['T_world_mesh'];camera=json.load(open(p+'camera.json'));cam=np.linalg.inv(camera['T_world_cam'])@new
assert new.dtype==np.float64 and new.shape==base.shape==(993,4,4)
assert np.isfinite(new).all()
assert np.allclose(new[:,3,:],np.array([0,0,0,1]))
assert np.allclose(np.linalg.det(new[:,:3,:3]),1,atol=1e-5)
# The released float32 rotations already have orthogonality residuals up to
# 2.16e-5. Preserve baseline frames exactly, and check against that precision.
base_error=np.max(np.abs(base[:,:3,:3]@base[:,:3,:3].transpose(0,2,1)-np.eye(3)))
assert np.max(np.abs(new[:,:3,:3]@new[:,:3,:3].transpose(0,2,1)-np.eye(3)))<=base_error+1e-6
ids=[0,200,300,450,550,650,750,800,900,992];c=cv2.VideoCapture(p+'video.mp4');out=Image.new('RGB',(3*330,len(ids)*280),'white');d=ImageDraw.Draw(out)
with h5py.File(p+'masks.h5') as h:
 for row,i in enumerate(ids):
  c.set(cv2.CAP_PROP_POS_FRAMES,i);ok,im=c.read();im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);im=cv2.resize(im,(768,576));gt=h['object'][i,::2,::2];mb=render(T[i]);mn=render(cam[i]);ys,xs=np.where(gt|mb|mn);cx=(xs.max()+xs.min())//2;cy=(ys.max()+ys.min())//2;side=int(max(xs.max()-xs.min(),ys.max()-ys.min())*1.24);x=max(0,min(768-side,cx-side//2));y=max(0,min(576-side,cy-side//2))
  for col,mask,label in [(0,None,'Video'),(1,mb,'Baseline'),(2,mn,'Refined')]:
   image=im.copy()
   if mask is not None:
    image[mask]=(.6*image[mask]+.4*np.array([255,35,25])).astype('uint8')
   crop=image[y:y+side,x:x+side];out.paste(Image.fromarray(crop).resize((270,250)),(col*330,row*280));d.text((col*330,row*280+252),f'{label}: frame {i}',fill='black')
out.save(o+'evidence.jpg',quality=92)
dg=np.rad2deg(Rotation.from_matrix(new[:,:3,:3]@base[:,:3,:3].transpose(0,2,1)).magnitude());dt=np.linalg.norm(new[:,:3,3]-base[:,:3,3],axis=-1)
print('Valid trajectory; rotation degrees median/max',np.median(dg),dg.max(),'mesh-origin translation median/max metres',np.median(dt),dt.max())
