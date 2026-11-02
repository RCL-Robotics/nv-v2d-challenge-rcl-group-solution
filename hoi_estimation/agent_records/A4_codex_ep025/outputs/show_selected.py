import numpy as np,cv2,json,torch,sys
from render_utils import Renderer
c=np.load(sys.argv[1] if len(sys.argv)>1 else '/work/outputs/det_candidates.npz');a=np.load('/work/outputs/conservative_selected.npz');r=Renderer('/work/inputs/object_mesh.glb',json.load(open('/work/inputs/camera.json'))['K']);images=c['images'];boxes=c['boxes'];baseR=c['R'][:,0];baset=c['t'][:,0];panels=[]
frames=[240,250,275,295,315,400,410,470,545,550,555,600,605,610,650,655,660,665,670,675,680,685,690,695,700,705,710,715,740,745]
for f in frames:
 Rs=np.array([baseR[f],a['R'][f]]);ts=np.array([baset[f],a['t'][f]])
 with torch.no_grad():m,_,rgb=r.render(Rs,ts,np.tile(boxes[f],(2,1)),texture=True)
 m=m.cpu().numpy();rgb=rgb.cpu().numpy();im=images[f];imgs=[]
 for j in range(2):
  b=im.copy();b[m[j]>.5]=rgb[j][m[j]>.5]*255;imgs.append(b)
 panel=np.concatenate([im]+imgs,axis=1)[...,::-1].copy();cv2.putText(panel,str(f)+' / '+str(a['chosen'][f]),(4,19),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);panels.append(panel)
for page in range(3):
 rows=[np.concatenate(panels[i:i+2],axis=1) for i in range(page*10,(page+1)*10,2)];cv2.imwrite(f'/work/outputs/selected_{page}.jpg',np.concatenate(rows,axis=0))
