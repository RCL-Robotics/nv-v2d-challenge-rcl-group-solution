import cv2,json,trimesh,h5py,numpy as np
from PIL import Image,ImageDraw
p='/work/inputs/';m=trimesh.load(p+'object_mesh.glb',force='scene',process=False).to_geometry();v=m.vertices;col=m.visual.to_color().vertex_colors; print('color',col.shape,col.min(0),col.max(0));hsv=cv2.cvtColor(col[:,:3].reshape(1,-1,3),cv2.COLOR_RGB2HSV)[0];orange=(hsv[:,0]<24)&(hsv[:,1]>130)&(hsv[:,2]>60);print('orange',orange.sum());
for lo,hi in [(-.52,-.35),(.88,1.03)]:
 sel=orange&(v[:,2]>lo)&(v[:,2]<hi); print('orange range',lo,hi,sel.sum(),np.mean(v[sel],axis=0),np.median(v[sel],axis=0))
np.savez('/work/outputs/mesh_colors.npz',vertices=v,colors=col,orange=orange)
a=np.load(p+'reconstruction.npz');c=json.load(open(p+'camera.json'));K=np.array(c['K']);T=np.linalg.inv(c['T_world_cam'])@a['T_world_mesh'];cap=cv2.VideoCapture(p+'video.mp4');f=h5py.File(p+'masks.h5');fs=[0,150,200,400,550,650,700,725,750,775,800,850,900,950,987];out=Image.new('RGB',(1500,1800),'white');draw=ImageDraw.Draw(out)
for j,i in enumerate(fs):
 cap.set(0 if False else cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();mask=f['object'][i];h=cv2.cvtColor(im,cv2.COLOR_BGR2HSV);sel=(h[:,:,0]<24)&(h[:,:,1]>130)&(h[:,:,2]>60)&cv2.dilate(mask.astype('uint8'),np.ones((5,5),np.uint8)).astype(bool);n,lab,stats,cent=cv2.connectedComponentsWithStats(sel.astype('uint8'));print('frame',i,'orange',[(s[4],np.round(c,1).tolist()) for s,c in zip(stats[1:],cent[1:]) if s[4]>3]);
 yy,xx=np.where(mask);x0,x1=max(0,xx.min()-50),min(1536,xx.max()+50);y0,y1=max(0,yy.min()-50),min(1152,yy.max()+50);im[sel]=(0,0,255);
 for s,ce in zip(stats[1:],cent[1:]):
  if s[4]>3:cv2.circle(im,tuple(ce.astype(int)),8,(0,255,255),1)
 crop=im[y0:y1,x0:x1];hh,ww=crop.shape[:2];ss=min(500/ww,330/hh);crop=cv2.resize(crop,(int(ww*ss),int(hh*ss)));out.paste(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)),((j%3)*500,(j//3)*360+30));draw.text(((j%3)*500+5,(j//3)*360+5),str(i),fill='black')
out.save('/work/outputs/orange_features.jpg')
