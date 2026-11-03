import sys,json,cv2,torch,numpy as np
from pathlib import Path
from solve import Renderer
p=Path('/work/inputs');o=Path('/work/outputs')
cam=json.loads((p/'camera.json').read_text());K=np.array(cam['K']);C=np.linalg.inv(cam['T_world_cam'])
base=C@np.load(p/'reconstruction.npz')['T_world_mesh'];final=C@np.load(o/'object_poses_world.npz')['T_world_mesh']
r=Renderer(p/'object_mesh.glb',K);cap=cv2.VideoCapture(str(p/'video.mp4'));frames=[200,240,250,400,550,600,655,670,680,695,700,750];panels=[]
for f in frames:
 cap.set(cv2.CAP_PROP_POS_FRAMES,f);ok,bgr=cap.read();assert ok
 uv=K@base[f,:3,3];uv=uv[:2]/uv[2];xy=np.round(uv-80).astype(int);x,y=xy;im=bgr[y:y+160,x:x+160,::-1].copy();box=np.array([x,y,1.])
 with torch.no_grad():mask,_,rgb=r.render(np.array([base[f,:3,:3],final[f,:3,:3]]),np.array([base[f,:3,3],final[f,:3,3]]),np.tile(box,(2,1)),texture=True)
 mask=mask.cpu().numpy();rgb=rgb.cpu().numpy();ims=[im]
 for j in range(2):
  rendered=im.copy();visible=mask[j]>.5;rendered[visible]=rgb[j][visible]*255;ims.append(rendered)
 panel=np.concatenate(ims,axis=1)[...,::-1].copy();bar=np.full((25,480,3),35,np.uint8)
 for j,label in enumerate(['Video '+str(f),'Baseline','Final']):cv2.putText(bar,label,(j*160+5,18),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
 panels.append(np.concatenate([bar,panel],axis=0))
rows=[np.concatenate(panels[j:j+2],axis=1) for j in range(0,len(panels),2)];assert cv2.imwrite(str(o/'final_evidence.jpg'),np.concatenate(rows,axis=0))
print('Rendered 12 saved-pose comparisons')
