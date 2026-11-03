import numpy as np, json, trimesh
c=json.load(open('/work/inputs/camera.json')); Twc=np.array(c['T_world_cam'])
rec=np.load('/work/inputs/reconstruction.npz'); hv=rec['human_vertices_world']
Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(float)
res=np.load('/work/scratch/static_res.npy',allow_pickle=True).item()
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
lo,hi=m.bounds; ctr=(lo+hi)/2; he=(hi-lo)/2
def sdf(p):
    q=np.abs(p-ctr)-he
    return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
ph={'C':(312,386),'E':(484,701)}
for k,frames in [('C',range(370,395)),('E',range(470,500)),('E',range(580,660))]:
    Tc=res[k]['T']; a,b=ph[k]; zb=np.median(Tcb[a:b+1,2,3])
    print(k,'sil z %.3f base z %.3f'%(Tc[2,3],zb))
    for z in np.arange(3.0,3.55,0.05):
        T=Tc.copy(); T[:3,3]*=z/Tc[2,3]; Ti=np.linalg.inv(Twc@T)
        mins=[sdf(hv[t].astype(float)@Ti[:3,:3].T+Ti[:3,3]).min() for t in frames]
        print('  z %.2f  min sdf over frames: min %.3f median %.3f'%(z,np.min(mins),np.median(mins)))
