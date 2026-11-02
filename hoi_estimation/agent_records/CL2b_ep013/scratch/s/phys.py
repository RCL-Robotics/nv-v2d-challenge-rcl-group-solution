import numpy as np, json, trimesh
from scipy.spatial.transform import Rotation as Rot
c=json.load(open('/work/inputs/camera.json')); K=np.array(c['K']); Twc=np.array(c['T_world_cam'])
pl=json.load(open('/work/inputs/ground_plane.json'))['plane']; n=np.array(pl[:3]); d0=pl[3]
rec=np.load('/work/inputs/reconstruction.npz'); Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(float)
res=np.load('/work/scratch/static_res.npy',allow_pickle=True).item()
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=np.asarray(m.vertices)[::20]
ph={'A':(0,131),'C':(312,386),'E':(484,701),'G':(817,921)}
Rw={}
for k,T in res.items():
    Tc=T['T']; a,b=ph[k]; zb=np.median(Tcb[a:b+1,2,3])
    for lab,s in [('sil',1.0),('base',zb/Tc[2,3])]:
        T2=Tc.copy(); T2[:3,3]*=s; Tw=Twc@T2
        pw=v@Tw[:3,:3].T+Tw[:3,3]; dist=pw@n+d0
        print(k,lab,'z %.3f'%T2[2,3],'plane dist min %.3f max %.3f'%(dist.min(),dist.max()), 'up axis in mesh',np.round(Tw[:3,:3].T@n,3))
    Rw[k]=(Twc@Tc)[:3,:3]
for a,b in [('A','C'),('C','E'),('E','G'),('A','G')]:
    Rr=Rw[b]@Rw[a].T; rv=Rot.from_matrix(Rr).as_rotvec(); ang=np.linalg.norm(rv)
    print(a,b,'angle %.1f'%np.degrees(ang),'axis',np.round(rv/ang,3),'angle between axis and normal %.1f'%np.degrees(np.arccos(abs(rv@n)/ang)))
