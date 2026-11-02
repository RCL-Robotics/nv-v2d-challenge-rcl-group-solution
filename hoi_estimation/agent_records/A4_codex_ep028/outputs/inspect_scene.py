import os,json,cv2,h5py,trimesh,numpy as np
from PIL import Image,ImageDraw
p='/work/inputs/'; out='/work/outputs/'
a=np.load(p+'reconstruction.npz'); c=json.load(open(p+'camera.json')); K=np.array(c['K']); W=np.linalg.inv(c['T_world_cam']); T=W@a['T_world_mesh']; m=trimesh.load(p+'object_mesh.glb',force='scene',process=False).to_geometry(); v=np.asarray(m.vertices); v=v[np.linspace(0,len(v)-1,18000).astype(int)]
cap=cv2.VideoCapture(p+'video.mp4'); fs=[0,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,987]
canvas=Image.new('RGB',(5*480,5*420),'white'); f=h5py.File(p+'masks.h5')
for j,i in enumerate(fs):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); mask=f['object'][i]; yy,xx=np.where(mask); pts=v@T[i,:3,:3].T+T[i,:3,3]; uv=pts@K.T; uv=uv[:,:2]/uv[:,2:];
 lo=np.minimum(uv.min(0),[xx.min(),yy.min()]);hi=np.maximum(uv.max(0),[xx.max(),yy.max()]); cen=(lo+hi)/2; width=max(hi[0]-lo[0]+100, (hi[1]-lo[1]+80)*480/390); left=max(0,int(cen[0]-width/2));top=max(0,int(cen[1]-width*390/480/2)); right=min(1536,left+int(width));bottom=min(1152,top+int(width*390/480));
 contours,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,(0,255,0),1)
 for x,y in uv.astype(int):
  if 0<=x<1536 and 0<=y<1152: im[y,x]=(255,80,255)
 crop=cv2.cvtColor(im[top:bottom,left:right],cv2.COLOR_BGR2RGB); tile=Image.fromarray(crop).resize((480,390));canvas.paste(tile,((j%5)*480,(j//5)*420+30));ImageDraw.Draw(canvas).text(((j%5)*480+8,(j//5)*420+6),f'frame {i} | magenta baseline / green SAM',fill='black')
 if i in [0,200,500,800]:
  cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,orig=cap.read();cv2.imwrite(out+f'raw_{i:04d}.jpg',orig)
canvas.save(out+'baseline_contact_sheet.jpg')
print('pose ranges',T[:,:3,3].min(0),T[:,:3,3].max(0));print('saved montage')
