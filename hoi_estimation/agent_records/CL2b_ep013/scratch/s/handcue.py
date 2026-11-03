import numpy as np, json, trimesh
c=json.load(open('/work/inputs/camera.json')); Twc=np.array(c['T_world_cam'])
rec=np.load('/work/inputs/reconstruction.npz'); kp=rec['human_keypoints_world'].astype(float); cp=rec['contact_prob_right_left']
Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(float)
res=np.load('/work/scratch/static_res.npy',allow_pickle=True).item()
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry()
lo,hi=m.bounds; ctr=(lo+hi)/2; he=(hi-lo)/2
def sdf(p):  # p in mesh frame (N,3)
    q=np.abs(p-ctr)-he
    return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)
ph={'A':(0,131),'C':(312,386),'E':(484,701),'G':(817,921)}
hands={'R':list(range(21,42)),'L':list(range(42,63))}
for k,(a,b) in ph.items():
    Tc=res[k]['T']; zb=np.median(Tcb[a:b+1,2,3]); zs=Tc[2,3]
    fr=[t for t in range(a,b+1) if cp[t].max()>0.5]
    print(k,'contact frames',len(fr), fr[:3],fr[-3:] if fr else '')
    if not fr: continue
    scales=np.linspace(0.9,1.1,201); best=[]
    for s in scales:
        T=Tc.copy(); T[:3,3]*=s; Tw=Twc@T; Ti=np.linalg.inv(Tw)
        ds=[]
        for t in fr:
            for h,(idx) in zip([0,1],hands.values()):
                if cp[t,h]<0.5: continue
                p=kp[t,idx]@Ti[:3,:3].T+Ti[:3,3]
                ds.append(np.abs(sdf(p)).min())   # closest hand keypoint to surface
        best.append(np.median(ds))
    best=np.array(best); i=np.argmin(best)
    print('  sil z %.3f base med z %.3f  hand-best z %.3f (median min-dist %.3f; at sil %.3f, at base %.3f)'%(zs,zb,zs*scales[i],best[i],best[100],best[np.argmin(abs(scales*zs-zb))]))
