import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, cv2
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp); mesh=load_mesh(inp)
Tcw=np.linalg.inv(Twc); o=Twc[:3,3]
fm=FloorModel(mesh,plane,np.array([-0.28,-0.02,-0.96]),h0=0.014)
R=Renderer(mesh,K,1536,1152)
T=rec['T_world_mesh'].astype(np.float64)
ids=[int(x) for x in sys.argv[1].split(',')]; offs=[float(x) for x in sys.argv[2].split(',')]
fr=read_video(inp+'/video.mp4',ids); rows=[]
for i,f in zip(ids,fr):
    th,a,b=fm.from_pose(T[i],o); tiles=[]
    Pc=Tcw@T[i]; c=Pc[:3,3]; uv=K@c; uv=uv[:2]/uv[2]; s=int(800/c[2])
    x0,y0=int(uv[0]-s/2),int(uv[1]-s/2)
    def cr(a): return cv2.resize(a[y0:y0+s,x0:x0+s],(300,300))
    tiles.append(cr(f))
    for off in offs:
        P=Tcw@fm.pose(th+np.radians(off),a,b); o_=R.render(P)
        m=o_['mask'][0].cpu().numpy(); rgb=(o_['rgb'][0].cpu().numpy()*255).astype(np.uint8)
        img=f.copy(); img[m]=rgb[m]; t=cr(img); cv2.putText(t,f'{i} {off:.0f}',(5,20),0,0.6,(255,255,0),2); tiles.append(t)
    rows.append(np.concatenate(tiles,1))
cv2.imwrite('/work/scratch/frames/flip.jpg',np.concatenate(rows,0)[:,:,::-1])
