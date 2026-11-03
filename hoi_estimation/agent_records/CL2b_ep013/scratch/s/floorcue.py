import numpy as np, json, trimesh
c=json.load(open('/work/inputs/camera.json')); Twc=np.array(c['T_world_cam'])
gp=json.load(open('/work/inputs/ground_plane.json')); pl=gp['plane']; n=np.array(pl[:3]); d0=pl[3]
rec=np.load('/work/inputs/reconstruction.npz'); Tcb=np.linalg.inv(Twc)[None]@rec['T_world_mesh'].astype(float)
res=np.load('/work/scratch/static_res.npy',allow_pickle=True).item()
m=trimesh.load('/work/inputs/object_mesh.glb',force='scene',process=False).to_geometry(); v=np.asarray(m.vertices)[::10]
ph={'A':(0,131),'C':(312,386),'E':(484,701),'G':(817,921)}
# human foot heights: lowest human vertices per frame
hv=rec['human_vertices_world']
fh=np.array([np.sort(hv[t].astype(float)@n+d0)[:20].mean() for t in range(0,922,5)])
print('human lowest-vertex height above plane: median %.3f'%np.median(fh))
for k in ['A','C','E','G']:
    Tc=res[k]['T']; a,b=ph[k]; zb=np.median(Tcb[a:b+1,2,3])
    zs=np.arange(2.9,3.6,0.001); h=[]
    for z in zs:
        T=Tc.copy(); T[:3,3]*=z/Tc[2,3]; Tw=Twc@T
        h.append(((v@Tw[:3,:3].T+Tw[:3,3])@n+d0).min())
    h=np.array(h)
    def zat(target): return zs[np.argmin(abs(h-target))]
    print(k,'sil z %.3f base z %.3f | bottom height at sil %.3f at base %.3f | z for bottom=0: %.3f, bottom=%.3f(foot level): %.3f | dh/dz %.3f'%(Tc[2,3],zb,h[np.argmin(abs(zs-Tc[2,3]))],h[np.argmin(abs(zs-zb))],zat(0),np.median(fh),zat(np.median(fh)),(h[-1]-h[0])/(zs[-1]-zs[0])))
