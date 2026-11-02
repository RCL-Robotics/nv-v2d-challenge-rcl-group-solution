import numpy as np, trimesh, json
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
v=m.vertices
for z0 in np.linspace(-0.21,0.33,12):
    s=np.abs(v[:,2]-z0)<0.02
    if s.sum(): print('z %.2f r %.3f  cx %.3f cy %.3f'%(z0,np.sqrt(v[s,0]**2+v[s,1]**2).max(),v[s,0].mean(),v[s,1].mean()))
print(m.visual.material)
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64)
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
ax=T[:,:3,2]; ctr=T[:,:3,3]
F=len(T)
def ang(R1,R2): return np.degrees(np.arccos(np.clip((np.trace(R1.T@R2)-1)/2,-1,1)))
for i in range(0,F,10):
    print(i, 'pos', np.round(ctr[i],3),'zc %.3f'%Tc[i,2,3],'axis',np.round(ax[i],2), 'dR %.1f'%ang(T[i,:3,:3],T[min(i+1,F-1),:3,:3]), 'cp',np.round(r['contact_prob_right_left'][i],2))
