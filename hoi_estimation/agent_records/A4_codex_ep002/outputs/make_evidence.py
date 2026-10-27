"""Optional visual audit; not used by solve.py."""
from solve import *
p = Path('/work/inputs'); out = Path('/work/outputs')
c = json.loads((p/'camera.json').read_text()); K=np.array(c['K']); inv=np.linalg.inv(c['T_world_cam'])
a=inv@np.load(p/'reconstruction.npz')['T_world_mesh'];b=inv@np.load(out/'object_poses_world.npz')['T_world_mesh']
mesh=trimesh.load(p/'object_mesh.glb',force='scene',process=False).to_geometry();r=Renderer(mesh,K)
cap=cv2.VideoCapture(str(p/'video.mp4'));rows=[]
for i in [0,200,300,638,700,723]:
 center=K@b[i,:3,3];center=center[:2]/center[2];left,top=np.rint(center-64).astype(int)
 cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,img=cap.read();assert ok
 crop=cv2.resize(img[top:top+128,left:left+128],(256,256),interpolation=cv2.INTER_NEAREST)
 panels=[]
 for label,pose,col in [('Video',None,None),('Baseline',a[i],(30,30,255)),('Corrected',b[i],(40,255,40))]:
  panel=crop.copy()
  if pose is not None:
   mask=r.render(pose,left,top,128,2);contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(panel,contours,-1,col,1)
  cv2.rectangle(panel,(0,0),(256,25),(20,20,20),-1)
  cv2.putText(panel,f'{i}: {label}',(7,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
  panels.append(panel)
 rows.append(np.concatenate(panels,1))
cv2.imwrite(str(out/'before_after.jpg'),np.concatenate(rows,0))
