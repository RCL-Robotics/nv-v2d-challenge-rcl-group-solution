import sys, numpy as np, json, torch
sys.path.insert(0,'/work/outputs')
from hoilib import *
cam=json.load(open('/work/inputs/camera.json')); K=np.array(cam['K']); Twc=np.array(cam['T_world_cam']); Tcw=np.linalg.inv(Twc)
gp=json.load(open('/work/inputs/ground_plane.json'))['plane']; down=-np.array(gp[:3]); down/=np.linalg.norm(down)
Tw=np.load('/work/inputs/reconstruction.npz')['T_world_mesh'].astype(float)
cr=np.load('/work/scratch/crops.npz'); mesh=TexMesh('/work/inputs/object_mesh.glb'); yf=YawFrame(down)
psi_b=np.array([yf.decompose(Tw[i,:3,:3])[1] for i in range(len(Tw))])
for (a,b) in [(0,150),(452,565)]:
    fr=list(range(a,b,4)); pm=np.arctan2(np.sin(psi_b[a:b]).mean(),np.cos(psi_b[a:b]).mean())
    tc=np.median((Tcw[None]@Tw[a:b])[:,:3,3],axis=0)
    offs=np.radians(np.arange(-30,31,3)); res=[]
    for o in offs:
        Tm=np.eye(4); Tm[:3,:3]=Tcw[:3,:3]@yf.compose(down,pm+o); Tm[:3,3]=tc
        T,I=fit_silhouettes(mesh,cr['Kc'][fr],cr['obs'][fr],cr['val'][fr],np.repeat(Tm[None],len(fr),0),iters=40,lr_r=0.0)
        res.append(I.mean())
    print((a,b),'baseline mean yaw %.1f'%np.degrees(pm),' '.join('%+d:%.3f'%(np.degrees(o),r) for o,r in zip(offs,res)))
