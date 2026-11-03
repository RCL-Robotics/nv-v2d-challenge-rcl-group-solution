import numpy as np,cv2
from PIL import Image,ImageDraw
a=np.load('/work/outputs/feature_tracks.npz');head=a['head'];cap=cv2.VideoCapture('/work/inputs/video.mp4');det=cv2.createLineSegmentDetector();out=Image.new('RGB',(1800,1200),'white');draw=ImageDraw.Draw(out)
for j,i in enumerate([760,770,780,790,800,810,820,830,840,850,860,870]):
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);_,im=cap.read();hp=head[i];hp=hp if np.isfinite(hp).all() else np.array([780,620]);x,y=hp.astype(int);roi=im[y-80:y+80,x-100:x+100].copy();gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY);lines=det.detect(gray)[0];h=hp-[x-100,y-80];cand=[]
 if lines is not None:
  for ll in lines[:,0]:
   aa,bb=ll[:2],ll[2:];le=np.linalg.norm(bb-aa);vv=(bb-aa)/le;dis=abs(np.cross(vv,h-aa));dd=min(np.linalg.norm(h-aa),np.linalg.norm(h-bb));
   if le>10 and dis<14 and dd<65:
    cand.append([round(le,1),round(dis,1),round(dd,1),np.round(vv,2).tolist()]);cv2.line(roi,tuple(aa.astype(int)),tuple(bb.astype(int)),(0,255,0),1)
 print(i,cand);roi=cv2.resize(roi,(450,360));out.paste(Image.fromarray(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)),((j%4)*450,(j//4)*400+30));draw.text(((j%4)*450+5,(j//4)*400+5),str(i),fill='black')
out.save('/work/outputs/blade_lines.jpg')
