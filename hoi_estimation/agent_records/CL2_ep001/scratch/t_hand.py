exec(open('/work/scratch/common_exp.py').read())
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
V=np.asarray(m.vertices); top=V[V[:,2]>0.60].mean(0); print('top',top)
kp=rec['human_keypoints_world'].astype(np.float64)
Tw=rec['T_world_mesh'].astype(np.float64)
gw=np.einsum('fij,j->fi',Tw[:,:3,:3],top)+Tw[:,:3,3]
for name,rng in [('R',range(21,42)),('L',range(42,63))]:
    h=kp[:,list(rng)].mean(1)
    dd=np.linalg.norm(h-gw,axis=1)
    # camera frame depth comparison
    hc=(np.linalg.inv(Twc)[:3,:3]@h.T).T+np.linalg.inv(Twc)[:3,3]; gc=(np.linalg.inv(Twc)[:3,:3]@gw.T).T+np.linalg.inv(Twc)[:3,3]
    print(name,[ (i,round(dd[i],3), round(hc[i,2]-gc[i,2],3)) for i in range(100,660,20)])
