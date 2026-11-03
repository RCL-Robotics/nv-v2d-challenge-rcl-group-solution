import sys; sys.path.insert(0,'/work/outputs')
from common import *
K,Kp,Twc,rec=load_inputs('inputs')
mesh=load_mesh('inputs/object_mesh.glb'); R=Renderer(mesh)
Tw=rec['T_world_mesh'].astype(np.float64)
Tc=np.linalg.inv(Twc)[None]@Tw
want=[int(a) for a in sys.argv[1].split(',')]
src=sys.argv[2] if len(sys.argv)>2 else 'base'
if src=='fp': Tc=rec['T_cam_mesh_foundationpose'].astype(np.float64); K=Kp
if src=='cc': Tc=rec['T_cam_mesh_coconet'].astype(np.float64); K=Kp
cap=cv2.VideoCapture('inputs/video.mp4'); frames={}; i=0
while True:
    ok,f=cap.read()
    if not ok or i>max(want): break
    if i in want: frames[i]=f
    i+=1
roi=(560,400,880,720)
tiles=[]
for i in want:
    rgb,m,d=R.render(Tc[i:i+1],K,roi,(480,480))
    crop=cv2.resize(frames[i][400:720,560:880],(480,480))
    r=(rgb[0].cpu().numpy()[...,::-1]*255).astype(np.uint8); mm=m[0].cpu().numpy()
    ov=crop.copy(); ov[mm]=(0.5*crop[mm]+0.5*r[mm]).astype(np.uint8)
    cnt,_=cv2.findContours(mm.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(ov,cnt,-1,(0,255,0),1)
    cv2.putText(ov,str(i),(5,25),0,0.8,(0,255,255),2)
    tiles.append(np.hstack([crop,ov]))
while len(tiles)%3: tiles.append(np.zeros_like(tiles[0]))
rows=[np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]
cv2.imwrite(f'scratch/viz/{src}_{sys.argv[3] if len(sys.argv)>3 else "x"}.jpg',np.vstack(rows))
