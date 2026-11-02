import numpy as np,torch,json
from refine import Renderer,to_torch
from conservative_select import angle
c=np.load('/work/outputs/det_candidates.npz');data={k:c[k] for k in c.files};K=np.array(json.load(open('/work/inputs/camera.json'))['K']);r=Renderer('/work/inputs/object_mesh.glb',K);d=to_torch(data);allstats=[]
with torch.no_grad():
 for f in [650,655,660,665,670,675,680,685,690,695,700,705]:
  R=data['R'][f];t=data['t'][f];n=len(R);m,at,col=r.render(R,t,np.repeat(data['boxes'][f:f+1],n,axis=0),texture=True,antialias=True)
  br=torch.sigmoid((col[...,2]-1.16*col[...,1]-.015)*30)*torch.sigmoid((col[...,2]-1.4*col[...,0])*30)*m;ob=d['blue'][f:f+1];v=d['valid'][f:f+1];iou=(br*ob).sum((1,2))/(br*v+ob-br*ob).sum((1,2)).clamp(min=20);ang=angle(R,R[0])*180/np.pi
  print('frame',f,'observed blue',ob.sum().item());print(np.round(np.stack([data['scores'][f],data['stats'][f,:,0],iou.cpu().numpy(),ang],1),3))
