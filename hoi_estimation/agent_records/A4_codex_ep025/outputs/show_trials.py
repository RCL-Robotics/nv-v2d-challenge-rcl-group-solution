import numpy as np,cv2,json,torch
from render_utils import Renderer
a=np.load('/work/outputs/trial_results.npz');base=np.load('/work/inputs/reconstruction.npz');cam=json.load(open('/work/inputs/camera.json'));T=np.linalg.inv(cam['T_world_cam'])@base['T_world_mesh'];r=Renderer('/work/inputs/object_mesh.glb',cam['K']);panels=[]
for j,f in enumerate(a['ids']):
 best=np.argmin(a['scores'][:,j]);Rs=np.array([T[f,:3,:3],a['R'][best,j]]);ts=np.array([T[f,:3,3],a['t'][best,j]])
 with torch.no_grad():m,attr,c=r.render(Rs,ts,np.tile(a['boxes'][j],(2,1)),texture=True)
 m=m.cpu().numpy();c=c.cpu().numpy();im=a['images'][j].copy();imgs=[]
 for i in range(2):
  b=im.copy();b[m[i]>.5]=c[i][m[i]>.5]*255;imgs.append(b)
 panel=np.concatenate([im]+imgs,axis=1)[...,::-1].copy();cv2.putText(panel,str(f),(4,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(panel)
for page in range(2):
 rows=[]
 for j in range(page*16,(page+1)*16,2):rows.append(np.concatenate(panels[j:j+2],axis=1))
 cv2.imwrite(f'/work/outputs/trials_{page}.jpg',np.concatenate(rows,axis=0))
