import cv2, numpy as np,json,h5py,trimesh
from PIL import Image,ImageDraw
p='/work/inputs/'
K=np.array(json.load(open(p+'camera.json'))['K']); W=np.array(json.load(open(p+'camera.json'))['T_world_cam']);r=np.load(p+'reconstruction.npz');T=np.linalg.inv(W)@r['T_world_mesh'];m=trimesh.load(p+'object_mesh.glb',force='scene',process=False).to_geometry()
v=m.vertices[::30]
c=cv2.VideoCapture(p+'video.mp4'); masks=h5py.File(p+'masks.h5')
frames=list(range(0,781,40)); result=Image.new('RGB',(5*400,4*310),'white'); draw=ImageDraw.Draw(result)
for i,f in enumerate(frames):
 c.set(cv2.CAP_PROP_POS_FRAMES,f);_,im=c.read(); mask=masks['object'][f];ys,xs=np.where(mask);x,y=int(np.median(xs)),int(np.median(ys));x0=max(x-100,0);y0=max(y-70,0);crop=im[y0:y0+155,x0:x0+200].copy();pts=v@T[f,:3,:3].T+T[f,:3,3];uv=pts@K.T;uv=uv[:,:2]/uv[:,2:];overlay=crop.copy()
 for xx,yy in uv.astype(int):
  if x0<=xx<x0+200 and y0<=yy<y0+155: overlay[yy-y0,xx-x0]=(0,255,0)
 contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 cv2.drawContours(overlay,[q-np.array([[[x0,y0]]]) for q in contours],-1,(255,0,255),1)
 both=np.concatenate([crop,overlay],axis=1);pil=Image.fromarray(cv2.cvtColor(both,cv2.COLOR_BGR2RGB)).resize((400,285)); result.paste(pil,((i%5)*400,(i//5)*310+25));draw.text(((i%5)*400+5,(i//5)*310+5),str(f),fill='black')
result.save('/work/outputs/baseline_contact_sheet.jpg')
