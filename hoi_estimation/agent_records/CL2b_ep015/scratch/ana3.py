import numpy as np, json
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64); Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam'])
Tcw=np.linalg.inv(Twc)
ref=np.load('/work/inputs/frame0_reference_human.npz'); print({k:ref[k].shape for k in ref})
hv=r['human_vertices_world'].astype(np.float64); kp=r['human_keypoints_world'].astype(np.float64)
rv=ref['vertices_world']
print('ref human centroid', rv.mean(0), 'recon f0 centroid (subsampled)', hv[0].mean(0), rv[::4].mean(0))
c_ref=(Tcw[:3,:3]@rv.mean(0))+Tcw[:3,3]; c_rec=(Tcw[:3,:3]@hv[0].mean(0))+Tcw[:3,3]
print('cam ref', c_ref, 'cam rec', c_rec)
if rv.shape[0]//4+1>=hv.shape[1]:
    d=hv[0]-rv[::4][:hv.shape[1]]; print('per-vertex diff mean', d.mean(0), np.linalg.norm(d,axis=1).mean())
handle=np.array([0,0,0.33]); hw=(T[:,:3,:3]@handle)+T[:,:3,3]
for i in [200,300,350,400,450,550]:
    dd=np.linalg.norm(hv[i]-hw[i],axis=1); j=dd.argmin()
    print(i,'nearest human vert to handle %.3f'%dd[j], 'idx',j, 'rh %.3f lh %.3f'%(np.linalg.norm(kp[i,21:42].mean(0)-hw[i]),np.linalg.norm(kp[i,42:63].mean(0)-hw[i])),
      'all kp min %.3f at %d'%(np.linalg.norm(kp[i]-hw[i],axis=1).min(), np.linalg.norm(kp[i]-hw[i],axis=1).argmin()))
